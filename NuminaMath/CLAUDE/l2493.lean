import Mathlib

namespace haunted_mansion_entry_exit_l2493_249346

/-- The number of windows in the haunted mansion -/
def num_windows : ℕ := 8

/-- The number of ways to enter and exit the haunted mansion through different windows -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem: The number of ways to enter and exit the haunted mansion through different windows is 56 -/
theorem haunted_mansion_entry_exit : num_ways = 56 := by
  sorry

end haunted_mansion_entry_exit_l2493_249346


namespace inverse_of_A_l2493_249357

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -1; 4, 3]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![3/10, 1/10; -2/5, 1/5]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end inverse_of_A_l2493_249357


namespace binary_to_quaternary_conversion_l2493_249344

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal [true, false, true, true, true, true, false, false]) = [2, 3, 3, 0] := by
  sorry

end binary_to_quaternary_conversion_l2493_249344


namespace sum_seventeen_terms_l2493_249318

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference
  sum_first_fifth : a + (a + 4 * d) = 5 / 3
  product_third_fourth : (a + 2 * d) * (a + 3 * d) = 65 / 72

/-- Sum of the first n terms of an arithmetic progression -/
def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  n / 2 * (2 * ap.a + (n - 1) * ap.d)

/-- The main theorem to prove -/
theorem sum_seventeen_terms (ap : ArithmeticProgression) :
  sum_n_terms ap 17 = 119 / 3 := by
  sorry

end sum_seventeen_terms_l2493_249318


namespace range_of_a_l2493_249345

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a * x + 1 else Real.log x

-- Define symmetry about the origin
def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∃ x > 0, f (-x) = -f x

-- Theorem statement
theorem range_of_a (a : ℝ) :
  symmetric_about_origin (f a) → a ∈ Set.Iic 1 :=
sorry

end range_of_a_l2493_249345


namespace cube_equation_solution_l2493_249390

theorem cube_equation_solution :
  ∃ x : ℝ, (2*x - 8)^3 = 64 ∧ x = 6 :=
by sorry

end cube_equation_solution_l2493_249390


namespace ellipse_equation_hyperbola_equation_l2493_249363

-- Ellipse problem
theorem ellipse_equation (f c a b : ℝ) (h1 : f = 8) (h2 : c = 4) (h3 : a = 5) (h4 : b = 3) (h5 : c / a = 0.8) :
  (∀ x y : ℝ, x^2 / 25 + y^2 / 9 = 1) ∨ (∀ x y : ℝ, y^2 / 25 + x^2 / 9 = 1) :=
sorry

-- Hyperbola problem
theorem hyperbola_equation (a b m : ℝ) 
  (h1 : ∀ x y : ℝ, y^2 / 4 - x^2 / 3 = 1 → y^2 / (4*m) - x^2 / (3*m) = 1) 
  (h2 : 3^2 / (6*m) - 2^2 / (8*m) = 1) :
  (∀ x y : ℝ, x^2 / 6 - y^2 / 8 = 1) :=
sorry

end ellipse_equation_hyperbola_equation_l2493_249363


namespace jake_fewer_peaches_indeterminate_peach_difference_l2493_249316

-- Define the number of apples and peaches for Steven
def steven_apples : ℕ := 52
def steven_peaches : ℕ := 13

-- Define Jake's apples in terms of Steven's
def jake_apples : ℕ := steven_apples + 84

-- Define a variable for Jake's peaches (unknown, but less than Steven's)
variable (jake_peaches : ℕ)

-- Theorem stating that Jake's peaches are fewer than Steven's
theorem jake_fewer_peaches : jake_peaches < steven_peaches := by sorry

-- Theorem stating that the exact difference in peaches cannot be determined
theorem indeterminate_peach_difference :
  ¬ ∃ (diff : ℕ), ∀ (jake_peaches : ℕ), jake_peaches < steven_peaches →
    steven_peaches - jake_peaches = diff := by sorry

end jake_fewer_peaches_indeterminate_peach_difference_l2493_249316


namespace school_pet_ownership_l2493_249378

theorem school_pet_ownership (total_students : ℕ) (cat_owners : ℕ) (bird_owners : ℕ)
  (h_total : total_students = 500)
  (h_cats : cat_owners = 80)
  (h_birds : bird_owners = 120) :
  (cat_owners : ℚ) / total_students * 100 = 16 ∧
  (bird_owners : ℚ) / total_students * 100 = 24 :=
by sorry

end school_pet_ownership_l2493_249378


namespace valid_sequence_probability_l2493_249359

/-- Recursive function to calculate the number of valid sequences of length n -/
def b : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 0
| 3 => 0
| 4 => 1
| 5 => 1
| 6 => 2
| 7 => 3
| n + 8 => b (n + 5) + b (n + 4)

/-- The probability of generating a valid sequence of length 12 -/
def prob : ℚ := 5 / 1024

theorem valid_sequence_probability :
  b 12 = 5 ∧ 2^10 = 1024 ∧ prob = (b 12 : ℚ) / 2^10 := by sorry

end valid_sequence_probability_l2493_249359


namespace craig_total_commissions_l2493_249383

/-- Represents the commission structure for an appliance brand -/
structure CommissionStructure where
  refrigerator_base : ℝ
  refrigerator_rate : ℝ
  washing_machine_base : ℝ
  washing_machine_rate : ℝ
  oven_base : ℝ
  oven_rate : ℝ

/-- Represents the sales data for an appliance brand -/
structure SalesData where
  refrigerators : ℕ
  refrigerators_price : ℝ
  washing_machines : ℕ
  washing_machines_price : ℝ
  ovens : ℕ
  ovens_price : ℝ

/-- Calculates the commission for a single appliance type -/
def calculate_commission (base : ℝ) (rate : ℝ) (quantity : ℕ) (total_price : ℝ) : ℝ :=
  (base + rate * total_price) * quantity

/-- Calculates the total commission for a brand -/
def total_brand_commission (cs : CommissionStructure) (sd : SalesData) : ℝ :=
  calculate_commission cs.refrigerator_base cs.refrigerator_rate sd.refrigerators sd.refrigerators_price +
  calculate_commission cs.washing_machine_base cs.washing_machine_rate sd.washing_machines sd.washing_machines_price +
  calculate_commission cs.oven_base cs.oven_rate sd.ovens sd.ovens_price

/-- Main theorem: Craig's total commissions for the week -/
theorem craig_total_commissions :
  let brand_a_cs : CommissionStructure := {
    refrigerator_base := 75,
    refrigerator_rate := 0.08,
    washing_machine_base := 50,
    washing_machine_rate := 0.10,
    oven_base := 60,
    oven_rate := 0.12
  }
  let brand_b_cs : CommissionStructure := {
    refrigerator_base := 90,
    refrigerator_rate := 0.06,
    washing_machine_base := 40,
    washing_machine_rate := 0.14,
    oven_base := 70,
    oven_rate := 0.10
  }
  let brand_a_sales : SalesData := {
    refrigerators := 3,
    refrigerators_price := 5280,
    washing_machines := 4,
    washing_machines_price := 2140,
    ovens := 5,
    ovens_price := 4620
  }
  let brand_b_sales : SalesData := {
    refrigerators := 2,
    refrigerators_price := 3780,
    washing_machines := 3,
    washing_machines_price := 2490,
    ovens := 4,
    ovens_price := 3880
  }
  total_brand_commission brand_a_cs brand_a_sales + total_brand_commission brand_b_cs brand_b_sales = 9252.60 := by
  sorry

end craig_total_commissions_l2493_249383


namespace power_equation_solution_l2493_249393

theorem power_equation_solution : 
  ∃! x : ℤ, (10 : ℝ) ^ x * (10 : ℝ) ^ 652 = 1000 ∧ x = -649 := by
  sorry

end power_equation_solution_l2493_249393


namespace sin_2alpha_values_l2493_249366

theorem sin_2alpha_values (α : Real) 
  (h1 : 2 * (Real.tan α)^2 - 7 * Real.tan α + 3 = 0) :
  (π < α ∧ α < 5*π/4 → Real.sin (2*α) = 4/5) ∧
  (5*π/4 < α ∧ α < 3*π/2 → Real.sin (2*α) = 3/5) := by
  sorry

end sin_2alpha_values_l2493_249366


namespace xiao_ming_reading_inequality_l2493_249335

/-- Represents Xiao Ming's reading situation -/
def reading_situation (total_pages : ℕ) (total_days : ℕ) (initial_pages_per_day : ℕ) (initial_days : ℕ) (remaining_pages_per_day : ℝ) : Prop :=
  (initial_pages_per_day * initial_days : ℝ) + (remaining_pages_per_day * (total_days - initial_days)) ≥ total_pages

/-- The inequality correctly represents Xiao Ming's reading situation -/
theorem xiao_ming_reading_inequality :
  reading_situation 72 10 5 2 x ↔ 10 + 8 * x ≥ 72 := by
  sorry

end xiao_ming_reading_inequality_l2493_249335


namespace series_convergence_implies_k_value_l2493_249315

/-- Given a real number k > 1 such that the infinite series
    Σ(n=1 to ∞) (7n-3)/k^n converges to 5, prove that k = 1.2 + 0.2√46. -/
theorem series_convergence_implies_k_value (k : ℝ) 
  (h1 : k > 1) 
  (h2 : ∑' n, (7 * n - 3) / k^n = 5) : 
  k = 1.2 + 0.2 * Real.sqrt 46 := by
  sorry

end series_convergence_implies_k_value_l2493_249315


namespace find_a_and_b_l2493_249336

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 + 7 * x - 15 < 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}

-- State the theorem
theorem find_a_and_b :
  ∃ (a b : ℝ),
    (A ∩ B a b = ∅) ∧
    (A ∪ B a b = {x | -5 < x ∧ x ≤ 2}) ∧
    (a = -7/2) ∧
    (b = 3) := by
  sorry

end find_a_and_b_l2493_249336


namespace units_digit_of_product_l2493_249314

theorem units_digit_of_product (n : ℕ) : 
  (2^2023 * 5^2024 * 11^2025) % 10 = 0 :=
by sorry

end units_digit_of_product_l2493_249314


namespace max_writers_is_fifty_l2493_249343

/-- Represents the number of people at a newspaper conference --/
structure ConferenceAttendees where
  total : Nat
  editors : Nat
  both : Nat
  neither : Nat
  hTotal : total = 90
  hEditors : editors > 38
  hNeither : neither = 2 * both
  hBothMax : both ≤ 6

/-- The maximum number of writers at the conference --/
def maxWriters (c : ConferenceAttendees) : Nat :=
  c.total - c.editors - c.both

/-- Theorem stating that the maximum number of writers is 50 --/
theorem max_writers_is_fifty (c : ConferenceAttendees) : maxWriters c ≤ 50 ∧ ∃ c', maxWriters c' = 50 := by
  sorry

#eval maxWriters { total := 90, editors := 39, both := 1, neither := 2, hTotal := rfl, hEditors := by norm_num, hNeither := rfl, hBothMax := by norm_num }

end max_writers_is_fifty_l2493_249343


namespace parabola_coefficients_l2493_249351

/-- A parabola with a vertical axis of symmetry -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℚ) : ℚ :=
  p.a * x^2 + p.b * x + p.c

/-- The vertex of a parabola -/
def Parabola.vertex (p : Parabola) : ℚ × ℚ :=
  (- p.b / (2 * p.a), p.c - p.b^2 / (4 * p.a))

theorem parabola_coefficients :
  ∃ (p : Parabola),
    p.vertex = (5, -3) ∧
    p.y_coord 3 = 7 ∧
    p.a = 5/2 ∧
    p.b = -25 ∧
    p.c = 119/2 := by
  sorry

end parabola_coefficients_l2493_249351


namespace range_of_k_l2493_249379

def p (k : ℝ) : Prop := k^2 - 8*k - 20 ≤ 0

def q (k : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
  (∀ (x y : ℝ), x^2 / (4 - k) + y^2 / (1 - k) = 1 ↔ 
    x^2 / a^2 - y^2 / b^2 = 1) ∧ 
  (4 - k > 0) ∧ (1 - k < 0)

theorem range_of_k (k : ℝ) : 
  ((p k ∨ q k) ∧ ¬(p k ∧ q k)) → 
  ((-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10)) :=
by sorry

end range_of_k_l2493_249379


namespace triangle_property_l2493_249388

-- Define the binary operation ★
noncomputable def star (A B : ℂ) : ℂ := 
  let ζ : ℂ := Complex.exp (Complex.I * Real.pi / 3)
  ζ * (B - A) + A

-- Define the theorem
theorem triangle_property (I M O : ℂ) :
  star I (star M O) = star (star O I) M →
  -- Triangle IMO is positively oriented
  (Complex.arg ((I - O) / (M - O)) > 0) ∧
  -- Triangle IMO is isosceles with OI = OM
  Complex.abs (I - O) = Complex.abs (M - O) ∧
  -- ∠IOM = 2π/3
  Complex.arg ((I - O) / (M - O)) = 2 * Real.pi / 3 :=
by sorry

end triangle_property_l2493_249388


namespace intersection_of_A_and_B_l2493_249385

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end intersection_of_A_and_B_l2493_249385


namespace boat_speed_in_still_water_l2493_249396

/-- Proves that a boat traveling 54 km downstream in 3 hours and 54 km upstream in 9 hours has a speed of 12 km/hr in still water. -/
theorem boat_speed_in_still_water : 
  ∀ (v_b v_r : ℝ), 
    v_b > 0 → 
    v_r > 0 → 
    v_b + v_r = 54 / 3 → 
    v_b - v_r = 54 / 9 → 
    v_b = 12 := by
  sorry

#check boat_speed_in_still_water

end boat_speed_in_still_water_l2493_249396


namespace elder_age_problem_l2493_249313

theorem elder_age_problem (y e : ℕ) : 
  e = y + 20 →                 -- The ages differ by 20 years
  e - 4 = 5 * (y - 4) →        -- 4 years ago, elder was 5 times younger's age
  e = 29                       -- Elder's present age is 29
  := by sorry

end elder_age_problem_l2493_249313


namespace complement_intersection_M_N_l2493_249362

-- Define the sets M and N
def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x > -1}

-- Define the universal set U
def U : Type := ℝ

-- State the theorem
theorem complement_intersection_M_N :
  (M ∩ N)ᶜ = {x : ℝ | x ≤ -1 ∨ x ≥ 3} :=
by sorry

end complement_intersection_M_N_l2493_249362


namespace building_cleaning_earnings_l2493_249302

/-- Calculates the total earnings for cleaning a building -/
def total_earnings (floors : ℕ) (rooms_per_floor : ℕ) (hours_per_room : ℕ) (hourly_rate : ℕ) : ℕ :=
  floors * rooms_per_floor * hours_per_room * hourly_rate

/-- Proves that the total earnings for cleaning the specified building is $3600 -/
theorem building_cleaning_earnings :
  total_earnings 4 10 6 15 = 3600 := by
  sorry

#eval total_earnings 4 10 6 15

end building_cleaning_earnings_l2493_249302


namespace points_on_opposite_sides_l2493_249331

-- Define the line
def line (x y : ℝ) : ℝ := 2*y - 6*x + 1

-- Define the points
def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem points_on_opposite_sides :
  line origin.1 origin.2 * line point.1 point.2 < 0 := by sorry

end points_on_opposite_sides_l2493_249331


namespace two_numbers_product_l2493_249312

theorem two_numbers_product (x y : ℕ) : 
  x ∈ Finset.range 33 ∧ 
  y ∈ Finset.range 33 ∧ 
  x ≠ y ∧
  (Finset.sum (Finset.range 33) id - x - y = x * y) →
  x * y = 484 := by
  sorry

end two_numbers_product_l2493_249312


namespace exists_n_good_not_n_plus_one_good_l2493_249376

/-- Sum of digits of a natural number -/
def S (k : ℕ) : ℕ := sorry

/-- A natural number is n-good if there exists a sequence satisfying the given condition -/
def is_n_good (a n : ℕ) : Prop :=
  ∃ (seq : Fin (n + 1) → ℕ), seq ⟨n, sorry⟩ = a ∧
    ∀ (i : Fin n), seq ⟨i.val + 1, sorry⟩ = seq i - S (seq i)

/-- For any n, there exists a number that is n-good but not (n+1)-good -/
theorem exists_n_good_not_n_plus_one_good :
  ∀ n : ℕ, ∃ a : ℕ, is_n_good a n ∧ ¬is_n_good a (n + 1) := by sorry

end exists_n_good_not_n_plus_one_good_l2493_249376


namespace expand_and_simplify_l2493_249332

theorem expand_and_simplify (x y : ℝ) : (x + 6) * (x + 8 + y) = x^2 + 14*x + x*y + 48 + 6*y := by
  sorry

end expand_and_simplify_l2493_249332


namespace range_of_a_l2493_249394

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 9*x + a^2/x + 7
  else if x > 0 then 9*x + a^2/x - 7
  else 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) ∧  -- f is odd
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) →  -- condition for x ≥ 0
  a ≤ -8/7 :=
by sorry

end range_of_a_l2493_249394


namespace spatial_relations_l2493_249352

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelPlaneLine : Plane → Line → Prop)
variable (perpendicularPlaneLine : Plane → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem spatial_relations 
  (m n l : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n ∧ m ≠ l ∧ n ≠ l) 
  (h_distinct_planes : α ≠ β) :
  -- Define the propositions
  let p1 := ∀ m n α, parallel m n → contains α n → parallelPlaneLine α m
  let p2 := ∀ l m α β, perpendicularPlaneLine α l → perpendicularPlaneLine β m → perpendicular l m → perpendicularPlanes α β
  let p3 := ∀ l m n, perpendicular l n → perpendicular m n → parallel l m
  let p4 := ∀ m n α β, perpendicularPlanes α β → intersect α β m → contains β n → perpendicular n m → perpendicularPlaneLine α n
  -- The theorem statement
  (¬p1 ∧ p2 ∧ ¬p3 ∧ p4) :=
by
  sorry

end spatial_relations_l2493_249352


namespace number_problem_l2493_249389

theorem number_problem : ∃ x : ℝ, 1.3333 * x = 4.82 ∧ abs (x - 3.615) < 0.001 := by
  sorry

end number_problem_l2493_249389


namespace factor_x4_minus_81_l2493_249372

theorem factor_x4_minus_81 (x : ℝ) : x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) := by
  sorry

end factor_x4_minus_81_l2493_249372


namespace hexagon_area_from_triangle_l2493_249340

/-- Given a triangle XYZ with circumcircle radius R and perimeter P, 
    the area of the hexagon formed by the intersection points of 
    the perpendicular bisectors with the circumcircle is (P * R) / 4 -/
theorem hexagon_area_from_triangle (R P : ℝ) (hR : R = 10) (hP : P = 45) :
  let hexagon_area := (P * R) / 4
  hexagon_area = 112.5 := by sorry

end hexagon_area_from_triangle_l2493_249340


namespace infinitely_many_solutions_iff_abs_a_gt_one_l2493_249327

-- Define the equation
def equation (a x y : ℤ) : Prop := x^2 + a*x*y + y^2 = 1

-- Define the property of having infinitely many integer solutions
def has_infinitely_many_solutions (a : ℤ) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), equation a x y ∧ x.natAbs + y.natAbs > n

-- Theorem statement
theorem infinitely_many_solutions_iff_abs_a_gt_one (a : ℤ) :
  has_infinitely_many_solutions a ↔ a.natAbs > 1 := by sorry

end infinitely_many_solutions_iff_abs_a_gt_one_l2493_249327


namespace probability_correct_l2493_249334

/-- Represents a runner on a circular track -/
structure Runner where
  direction : Bool  -- true for counterclockwise, false for clockwise
  lap_time : ℝ      -- time to complete one lap in seconds

/-- Represents the track and race setup -/
structure RaceSetup where
  track_length : ℝ           -- length of the track in meters
  focus_start : ℝ            -- start of focus area in meters from start line
  focus_length : ℝ           -- length of focus area in meters
  alice : Runner
  bob : Runner
  race_start_time : ℝ        -- start time of the race in seconds
  photo_start_time : ℝ       -- start time of photo opportunity in seconds
  photo_end_time : ℝ         -- end time of photo opportunity in seconds

def setup : RaceSetup := {
  track_length := 500
  focus_start := 50
  focus_length := 150
  alice := { direction := true, lap_time := 120 }
  bob := { direction := false, lap_time := 75 }
  race_start_time := 0
  photo_start_time := 15 * 60
  photo_end_time := 16 * 60
}

/-- Calculates the probability of both runners being in the focus area -/
def probability_both_in_focus (s : RaceSetup) : ℚ :=
  11/60

theorem probability_correct (s : RaceSetup) :
  s = setup → probability_both_in_focus s = 11/60 := by sorry

end probability_correct_l2493_249334


namespace product_of_fractions_l2493_249305

theorem product_of_fractions :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end product_of_fractions_l2493_249305


namespace left_of_kolya_l2493_249387

/-- The number of people in a physical education class line-up -/
def ClassSize : ℕ := 29

/-- The number of people to the right of Kolya -/
def RightOfKolya : ℕ := 12

/-- The number of people to the left of Sasha -/
def LeftOfSasha : ℕ := 20

/-- The number of people to the right of Sasha -/
def RightOfSasha : ℕ := 8

/-- Theorem: The number of people to the left of Kolya is 16 -/
theorem left_of_kolya : ClassSize - RightOfKolya - 1 = 16 := by
  sorry

end left_of_kolya_l2493_249387


namespace school_travel_time_l2493_249381

/-- Given a boy who walks at 7/6 of his usual rate and reaches school 6 minutes early,
    his usual time to reach the school is 42 minutes. -/
theorem school_travel_time (usual_rate : ℝ) (usual_time : ℝ) 
    (h1 : usual_rate > 0) 
    (h2 : usual_time > 0)
    (h3 : usual_rate * usual_time = (7/6 * usual_rate) * (usual_time - 6)) : 
  usual_time = 42 := by
sorry

end school_travel_time_l2493_249381


namespace max_value_x_plus_y_l2493_249392

/-- Given plane vectors OA, OB, OC satisfying certain conditions, 
    the maximum value of x + y is √2. -/
theorem max_value_x_plus_y (OA OB OC : ℝ × ℝ) (x y : ℝ) : 
  (norm OA = 1) → 
  (norm OB = 1) → 
  (norm OC = 1) → 
  (OA.1 * OB.1 + OA.2 * OB.2 = 0) → 
  (OC = (x * OA.1 + y * OB.1, x * OA.2 + y * OB.2)) → 
  (∃ (x y : ℝ), x + y ≤ Real.sqrt 2 ∧ 
    ∀ (x' y' : ℝ), (OC = (x' * OA.1 + y' * OB.1, x' * OA.2 + y' * OB.2)) → 
      x' + y' ≤ x + y) :=
by sorry

end max_value_x_plus_y_l2493_249392


namespace carwash_donation_percentage_l2493_249325

/-- Proves that the percentage of carwash proceeds donated is 90%, given the conditions of Hank's fundraising activities. -/
theorem carwash_donation_percentage
  (carwash_amount : ℝ)
  (bake_sale_amount : ℝ)
  (lawn_mowing_amount : ℝ)
  (bake_sale_donation_percentage : ℝ)
  (lawn_mowing_donation_percentage : ℝ)
  (total_donation : ℝ)
  (h1 : carwash_amount = 100)
  (h2 : bake_sale_amount = 80)
  (h3 : lawn_mowing_amount = 50)
  (h4 : bake_sale_donation_percentage = 0.75)
  (h5 : lawn_mowing_donation_percentage = 1)
  (h6 : total_donation = 200)
  (h7 : total_donation = carwash_amount * x + bake_sale_amount * bake_sale_donation_percentage + lawn_mowing_amount * lawn_mowing_donation_percentage)
  : x = 0.9 := by
  sorry

#check carwash_donation_percentage

end carwash_donation_percentage_l2493_249325


namespace smallest_a1_l2493_249307

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = 7 * a (n - 1) - n

theorem smallest_a1 (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  sequence_property a →
  ∀ a1 : ℝ, (a 1 = a1 ∧ ∀ n, a n > 0) → a1 ≥ 13/36 :=
sorry

end smallest_a1_l2493_249307


namespace group_size_problem_l2493_249371

theorem group_size_problem (total_collection : ℕ) 
  (h1 : total_collection = 3249) 
  (h2 : ∃ n : ℕ, n * n = total_collection) : 
  ∃ n : ℕ, n = 57 ∧ n * n = total_collection :=
sorry

end group_size_problem_l2493_249371


namespace zoo_animal_ratio_l2493_249338

/-- Given a zoo with penguins and polar bears, prove the ratio of polar bears to penguins -/
theorem zoo_animal_ratio (num_penguins num_total : ℕ) 
  (h1 : num_penguins = 21)
  (h2 : num_total = 63) :
  (num_total - num_penguins) / num_penguins = 2 := by
  sorry

end zoo_animal_ratio_l2493_249338


namespace distance_between_cities_l2493_249399

/-- The distance between two cities given specific conditions of bus and car travel --/
theorem distance_between_cities (bus_speed car_speed : ℝ) 
  (h1 : bus_speed = 40)
  (h2 : car_speed = 50)
  (h3 : 0 < bus_speed ∧ 0 < car_speed)
  : ∃ (s : ℝ), s = 160 ∧ 
    (s - 10) / car_speed + 1/4 = (s - 30) / bus_speed := by
  sorry

end distance_between_cities_l2493_249399


namespace subcommittee_count_l2493_249348

theorem subcommittee_count : 
  let total_members : ℕ := 12
  let coach_count : ℕ := 5
  let subcommittee_size : ℕ := 5
  let total_subcommittees := Nat.choose total_members subcommittee_size
  let non_coach_count := total_members - coach_count
  let all_non_coach_subcommittees := Nat.choose non_coach_count subcommittee_size
  total_subcommittees - all_non_coach_subcommittees = 771 := by
sorry

end subcommittee_count_l2493_249348


namespace rose_garden_delivery_l2493_249368

theorem rose_garden_delivery (red yellow white : ℕ) : 
  red + yellow = 120 →
  red + white = 105 →
  yellow + white = 45 →
  red + yellow + white = 135 →
  (red = 90 ∧ white = 15 ∧ yellow = 30) := by
  sorry

end rose_garden_delivery_l2493_249368


namespace nineteenth_replacement_in_july_l2493_249337

/-- Represents the months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Calculates the number of months between two replacements -/
def monthsBetweenReplacements : ℕ := 7

/-- Calculates the total number of months after a given number of replacements -/
def totalMonthsAfter (replacements : ℕ) : ℕ :=
  monthsBetweenReplacements * (replacements - 1)

/-- Determines the month after a given number of months from January -/
def monthAfter (months : ℕ) : Month :=
  match months % 12 with
  | 0 => Month.January
  | 1 => Month.February
  | 2 => Month.March
  | 3 => Month.April
  | 4 => Month.May
  | 5 => Month.June
  | 6 => Month.July
  | 7 => Month.August
  | 8 => Month.September
  | 9 => Month.October
  | 10 => Month.November
  | _ => Month.December

/-- Theorem: The 19th replacement occurs in July -/
theorem nineteenth_replacement_in_july :
  monthAfter (totalMonthsAfter 19) = Month.July := by
  sorry

end nineteenth_replacement_in_july_l2493_249337


namespace parallelogram_EFGH_area_l2493_249326

-- Define the parallelogram EFGH
def E : ℝ × ℝ := (1, 3)
def F : ℝ × ℝ := (5, 3)
def G : ℝ × ℝ := (6, 1)
def H : ℝ × ℝ := (2, 1)

-- Define the area function for a parallelogram
def parallelogram_area (a b c d : ℝ × ℝ) : ℝ :=
  let base := abs (b.1 - a.1)
  let height := abs (a.2 - d.2)
  base * height

-- Theorem statement
theorem parallelogram_EFGH_area :
  parallelogram_area E F G H = 8 := by sorry

end parallelogram_EFGH_area_l2493_249326


namespace cubic_equation_product_l2493_249321

theorem cubic_equation_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2010) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2009)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2010) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2009)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2010) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2009) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1 := by
  sorry

end cubic_equation_product_l2493_249321


namespace price_adjustment_percentage_l2493_249364

theorem price_adjustment_percentage (P : ℝ) (x : ℝ) (h : P > 0) :
  P * (1 + x / 100) * (1 - x / 100) = 0.75 * P →
  x = 50 := by
sorry

end price_adjustment_percentage_l2493_249364


namespace specific_eighth_term_l2493_249398

/-- An arithmetic sequence is defined by its second and fourteenth terms -/
structure ArithmeticSequence where
  second_term : ℚ
  fourteenth_term : ℚ

/-- The eighth term of an arithmetic sequence -/
def eighth_term (seq : ArithmeticSequence) : ℚ :=
  (seq.second_term + seq.fourteenth_term) / 2

/-- Theorem stating the eighth term of the specific arithmetic sequence -/
theorem specific_eighth_term :
  let seq := ArithmeticSequence.mk (8/11) (9/13)
  eighth_term seq = 203/286 := by sorry

end specific_eighth_term_l2493_249398


namespace min_value_sum_fractions_l2493_249360

theorem min_value_sum_fractions (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) ≥ 7 ∧
  ((x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) = 7 ↔ x = y ∧ y = z) :=
by sorry

end min_value_sum_fractions_l2493_249360


namespace log_expression_equality_complex_expression_equality_l2493_249311

-- Part 1
theorem log_expression_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

-- Part 2
theorem complex_expression_equality : (8 / 125) ^ (-(1 / 3)) - (-3 / 5) ^ 0 + 16 ^ 0.75 = 19 / 2 := by
  sorry

end log_expression_equality_complex_expression_equality_l2493_249311


namespace candle_height_shadow_relation_l2493_249356

/-- Given two positions of a gnomon and the shadows cast, we can relate the height of the object to the shadow lengths and distance between positions. -/
theorem candle_height_shadow_relation 
  (h : ℝ) -- height of the candle
  (d : ℝ) -- distance between gnomon positions
  (a : ℝ) -- length of shadow in first position
  (b : ℝ) -- length of shadow in second position
  (x : ℝ) -- length from base of candle at first position to end of shadow in second position plus d
  (h_pos : h > 0)
  (d_pos : d > 0)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (x_def : x = d + b) -- definition of x
  : x = h * (1 + d / (a + b)) := by
  sorry


end candle_height_shadow_relation_l2493_249356


namespace solution_set_of_inequality_l2493_249342

theorem solution_set_of_inequality (x : ℝ) :
  (6 * x^2 + 5 * x < 4) ↔ (-4/3 < x ∧ x < 1/2) := by
  sorry

end solution_set_of_inequality_l2493_249342


namespace solution_set_l2493_249347

/-- A decreasing function f: ℝ → ℝ that passes through (0, 3) and (3, -1) -/
def f : ℝ → ℝ :=
  sorry

/-- f is a decreasing function -/
axiom f_decreasing : ∀ x y, x < y → f y < f x

/-- f(0) = 3 -/
axiom f_at_zero : f 0 = 3

/-- f(3) = -1 -/
axiom f_at_three : f 3 = -1

/-- The solution set of |f(x+1) - 1| < 2 is (-1, 2) -/
theorem solution_set : 
  {x : ℝ | |f (x + 1) - 1| < 2} = Set.Ioo (-1) 2 :=
sorry

end solution_set_l2493_249347


namespace marble_probability_correct_l2493_249370

def marble_probability (initial_red : ℕ) (initial_blue : ℕ) (initial_green : ℕ) (initial_white : ℕ)
                       (removed_red : ℕ) (removed_blue : ℕ) (added_green : ℕ) :
  (ℚ × ℚ × ℚ) :=
  let final_red : ℕ := initial_red - removed_red
  let final_blue : ℕ := initial_blue - removed_blue
  let final_green : ℕ := initial_green + added_green
  let total : ℕ := final_red + final_blue + final_green + initial_white
  ((final_red : ℚ) / total, (final_blue : ℚ) / total, (final_green : ℚ) / total)

theorem marble_probability_correct :
  marble_probability 12 10 8 5 5 4 3 = (7/29, 6/29, 11/29) := by
  sorry

end marble_probability_correct_l2493_249370


namespace solve_for_b_l2493_249322

theorem solve_for_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 35 * 45 * b) : b = 1 := by
  sorry

end solve_for_b_l2493_249322


namespace mushroom_soup_total_l2493_249328

theorem mushroom_soup_total (team1 team2 team3 : ℕ) 
  (h1 : team1 = 90) 
  (h2 : team2 = 120) 
  (h3 : team3 = 70) : 
  team1 + team2 + team3 = 280 := by
sorry

end mushroom_soup_total_l2493_249328


namespace flour_per_new_crust_l2493_249353

/-- Amount of flour per pie crust in cups -/
def flour_per_crust : ℚ := 1 / 8

/-- Number of pie crusts made daily -/
def daily_crusts : ℕ := 40

/-- Total flour used daily in cups -/
def total_flour : ℚ := daily_crusts * flour_per_crust

/-- Number of new pie crusts -/
def new_crusts : ℕ := 50

/-- Number of cakes -/
def cakes : ℕ := 10

/-- Flour used for cakes in cups -/
def cake_flour : ℚ := 1

/-- Theorem stating the amount of flour per new pie crust -/
theorem flour_per_new_crust : 
  (total_flour - cake_flour) / new_crusts = 2 / 25 := by sorry

end flour_per_new_crust_l2493_249353


namespace pigs_joined_l2493_249304

theorem pigs_joined (initial_pigs final_pigs : ℕ) (h : initial_pigs ≤ final_pigs) :
  final_pigs - initial_pigs = final_pigs - initial_pigs :=
by sorry

end pigs_joined_l2493_249304


namespace coin_and_die_prob_l2493_249330

/-- A fair coin -/
def FairCoin : Type := Bool

/-- A regular eight-sided die -/
def EightSidedDie : Type := Fin 8

/-- The event of getting heads on a fair coin -/
def headsEvent (c : FairCoin) : Prop := c = true

/-- The event of getting an even number on an eight-sided die -/
def evenDieEvent (d : EightSidedDie) : Prop := d.val % 2 = 0

/-- The probability of an event on a fair coin -/
axiom probCoin (event : FairCoin → Prop) : ℚ

/-- The probability of an event on an eight-sided die -/
axiom probDie (event : EightSidedDie → Prop) : ℚ

/-- The probability of getting heads on a fair coin -/
axiom prob_heads : probCoin headsEvent = 1/2

/-- The probability of getting an even number on an eight-sided die -/
axiom prob_even_die : probDie evenDieEvent = 1/2

/-- The main theorem: The probability of getting heads on a fair coin and an even number
    on a regular eight-sided die when flipped and rolled once is 1/4 -/
theorem coin_and_die_prob :
  probCoin headsEvent * probDie evenDieEvent = 1/4 := by sorry

end coin_and_die_prob_l2493_249330


namespace additional_men_count_l2493_249384

theorem additional_men_count (initial_men : ℕ) (initial_days : ℕ) (final_days : ℕ) :
  initial_men = 600 →
  initial_days = 20 →
  final_days = 15 →
  ∃ (additional_men : ℕ), 
    initial_men * initial_days = (initial_men + additional_men) * final_days ∧
    additional_men = 200 := by
  sorry

end additional_men_count_l2493_249384


namespace largest_integer_problem_l2493_249333

theorem largest_integer_problem (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  (a + b + c + d) / 4 = 72 ∧  -- Average is 72
  a = 21  -- Smallest integer is 21
  → d = 222 := by  -- Largest integer is 222
  sorry

end largest_integer_problem_l2493_249333


namespace circle_properties_l2493_249324

/-- A circle in the xy-plane is defined by the equation x^2 + y^2 - 6x = 0. -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

/-- The center of the circle is the point (h, k) in ℝ² -/
def circle_center : ℝ × ℝ := (3, 0)

/-- The radius of the circle is r -/
def circle_radius : ℝ := 3

/-- Theorem stating that the given equation describes a circle with center (3, 0) and radius 3 -/
theorem circle_properties :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
sorry

end circle_properties_l2493_249324


namespace ellipse_axis_endpoint_distance_l2493_249329

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis of the ellipse 4(x+2)^2 + 16y^2 = 64 is 2√5. -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ),
    (∀ (x y : ℝ), 4 * (x + 2)^2 + 16 * y^2 = 64 → 
      ((x = C.1 ∧ y = C.2) ∨ (x = -C.1 ∧ y = -C.2)) ∨ 
      ((x = D.1 ∧ y = D.2) ∨ (x = -D.1 ∧ y = -D.2))) →
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end ellipse_axis_endpoint_distance_l2493_249329


namespace parabola_tangent_lines_l2493_249369

/-- The parabola defined by x^2 = 4y with focus (0, 1) -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4 * p.2}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (0, 1)

/-- The line perpendicular to the y-axis passing through the focus -/
def PerpendicularLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0}

/-- The intersection points of the parabola and the perpendicular line -/
def IntersectionPoints : Set (ℝ × ℝ) :=
  Parabola ∩ PerpendicularLine

/-- The tangent line at a point on the parabola -/
def TangentLine (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.1 + p.2 * q.2 + (p.1^2 / 4 + 1) = 0}

theorem parabola_tangent_lines :
  ∀ p ∈ IntersectionPoints,
    TangentLine p = {q : ℝ × ℝ | q.1 + q.2 + 1 = 0} ∨
    TangentLine p = {q : ℝ × ℝ | q.1 - q.2 - 1 = 0} :=
by sorry

end parabola_tangent_lines_l2493_249369


namespace pure_imaginary_equation_l2493_249375

-- Define the complex number i
def i : ℂ := Complex.I

-- Define a pure imaginary number
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem pure_imaginary_equation (z : ℂ) (b : ℝ) 
  (h1 : isPureImaginary z) 
  (h2 : (2 - i) * z = 4 - b * i) : 
  b = -8 := by sorry

end pure_imaginary_equation_l2493_249375


namespace fifteenth_term_of_sequence_l2493_249317

/-- Given an arithmetic sequence where the first term is 5 and the common difference is 2,
    prove that the 15th term is equal to 33. -/
theorem fifteenth_term_of_sequence (a : ℕ → ℕ) :
  a 1 = 5 →
  (∀ n : ℕ, a (n + 1) = a n + 2) →
  a 15 = 33 := by
  sorry

end fifteenth_term_of_sequence_l2493_249317


namespace cubic_roots_inequality_l2493_249350

theorem cubic_roots_inequality (a b : ℝ) 
  (h : ∀ x : ℝ, a * x^3 - x^2 + b * x - 1 = 0 → x > 0) : 
  0 < 3 * a * b ∧ 3 * a * b ≤ 1 ∧ b ≥ Real.sqrt 3 := by
sorry

end cubic_roots_inequality_l2493_249350


namespace curve_not_hyperbola_l2493_249374

/-- The curve equation -/
def curve_equation (m : ℝ) (x y : ℝ) : Prop :=
  (m - 1) * x^2 + (3 - m) * y^2 = (m - 1) * (3 - m)

/-- Definition of a non-hyperbola based on the coefficient condition -/
def is_not_hyperbola (m : ℝ) : Prop :=
  (m - 1) * (3 - m) ≥ 0

/-- Theorem stating that for m in [1,3], the curve is not a hyperbola -/
theorem curve_not_hyperbola (m : ℝ) (h : 1 ≤ m ∧ m ≤ 3) : is_not_hyperbola m := by
  sorry

end curve_not_hyperbola_l2493_249374


namespace sea_glass_collection_l2493_249303

/-- Sea glass collection problem -/
theorem sea_glass_collection (blanche_green blanche_red rose_blue dorothy_total : ℕ) 
  (h1 : blanche_green = 12)
  (h2 : blanche_red = 3)
  (h3 : rose_blue = 11)
  (h4 : dorothy_total = 57) :
  ∃ (rose_red : ℕ),
    dorothy_total = 2 * (blanche_red + rose_red) + 3 * rose_blue ∧ 
    rose_red = 9 := by
  sorry

end sea_glass_collection_l2493_249303


namespace marys_income_percentage_l2493_249320

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = 0.4 * juan) 
  (h2 : mary = 1.6 * tim) : 
  mary = 0.64 * juan := by
sorry

end marys_income_percentage_l2493_249320


namespace number_division_problem_l2493_249365

theorem number_division_problem : ∃ N : ℕ, N = (555 + 445) * (2 * (555 - 445)) + 30 := by
  sorry

end number_division_problem_l2493_249365


namespace wife_account_percentage_l2493_249358

def income : ℝ := 800000

def children_percentage : ℝ := 0.2
def num_children : ℕ := 3
def orphan_donation_percentage : ℝ := 0.05
def final_amount : ℝ := 40000

theorem wife_account_percentage :
  let children_total := children_percentage * num_children * income
  let after_children := income - children_total
  let orphan_donation := orphan_donation_percentage * after_children
  let after_donation := after_children - orphan_donation
  let wife_deposit := after_donation - final_amount
  (wife_deposit / income) * 100 = 33 := by sorry

end wife_account_percentage_l2493_249358


namespace three_algorithms_among_four_l2493_249341

/-- A statement describing a process or task -/
structure Statement where
  description : String

/-- Predicate to determine if a statement is an algorithm -/
def is_algorithm (s : Statement) : Prop :=
  -- This definition would typically include formal criteria for what constitutes an algorithm
  sorry

/-- The set of given statements -/
def given_statements : Finset Statement := sorry

theorem three_algorithms_among_four :
  ∃ (alg_statements : Finset Statement),
    alg_statements ⊆ given_statements ∧
    (∀ s ∈ alg_statements, is_algorithm s) ∧
    Finset.card alg_statements = 3 ∧
    Finset.card given_statements = 4 := by
  sorry

end three_algorithms_among_four_l2493_249341


namespace parabola_hyperbola_intersection_l2493_249355

/-- Given a hyperbola and a parabola with specific properties, prove that the parameter p of the parabola equals 4. -/
theorem parabola_hyperbola_intersection (a b p k : ℝ) : 
  a > 0 → b > 0 → p > 0 →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  a = 2*Real.sqrt 2 →  -- Real axis length
  b = p/2 →  -- Imaginary axis endpoint coincides with parabola focus
  (∀ x y, x^2 = 2*p*y) →  -- Parabola equation
  (∃ x y, y = k*x - 1 ∧ x^2 = 2*p*y) →  -- Line is tangent to parabola
  k = p/(4*Real.sqrt 2) →  -- Line is parallel to hyperbola asymptote
  p = 4 := by
sorry

end parabola_hyperbola_intersection_l2493_249355


namespace survey_respondents_l2493_249354

/-- Prove that the number of customers who responded to a survey is 50, given the following conditions:
  1. The average income of all customers is $45,000
  2. There are 10 wealthiest customers
  3. The average income of the 10 wealthiest customers is $55,000
  4. The average income of the remaining customers is $42,500
-/
theorem survey_respondents (N : ℕ) : 
  (10 * 55000 + (N - 10) * 42500 = N * 45000) → N = 50 := by
  sorry

end survey_respondents_l2493_249354


namespace equation_equivalence_l2493_249361

theorem equation_equivalence (x : ℝ) : 
  (x + 1) / 0.3 - (2 * x - 1) / 0.7 = 1 ↔ (10 * x + 10) / 3 - (20 * x - 10) / 7 = 1 :=
by sorry

end equation_equivalence_l2493_249361


namespace system_solutions_correct_l2493_249308

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℝ, 2*x + y = 3 ∧ 3*x - 5*y = 11 ∧ x = 2 ∧ y = -1) ∧
  -- System 2
  (∃ a b c : ℝ, a + b + c = 0 ∧ a - b + c = -4 ∧ 4*a + 2*b + c = 5 ∧
                a = 1 ∧ b = 2 ∧ c = -3) :=
by sorry

end system_solutions_correct_l2493_249308


namespace lcm_of_three_numbers_l2493_249391

theorem lcm_of_three_numbers (A B C : ℕ+) 
  (h_product : A * B * C = 185771616)
  (h_hcf_abc : Nat.gcd A (Nat.gcd B C) = 121)
  (h_hcf_ab : Nat.gcd A B = 363) :
  Nat.lcm A (Nat.lcm B C) = 61919307 := by
  sorry

end lcm_of_three_numbers_l2493_249391


namespace repeating_decimal_sum_difference_l2493_249309

theorem repeating_decimal_sum_difference (x y z : ℚ) :
  x = 5/9 ∧ y = 1/9 ∧ z = 3/9 → x + y - z = 1/3 := by
  sorry

end repeating_decimal_sum_difference_l2493_249309


namespace cube_cutting_problem_l2493_249382

theorem cube_cutting_problem :
  ∃! (n : ℕ), ∃ (s : ℕ), s < n ∧ n^3 - s^3 = 152 := by
  sorry

end cube_cutting_problem_l2493_249382


namespace max_value_h_exists_m_for_inequality_l2493_249349

open Real

/-- The natural logarithm function -/
noncomputable def f (x : ℝ) : ℝ := log x

/-- The square function -/
def g (x : ℝ) : ℝ := x^2

/-- The function h(x) = ln x - x + 1 -/
noncomputable def h (x : ℝ) : ℝ := f x - x + 1

theorem max_value_h :
  ∀ x > 0, h x ≤ 0 ∧ ∃ x₀ > 0, h x₀ = 0 :=
sorry

theorem exists_m_for_inequality (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hlt : x₁ < x₂) :
  ∃ m ≤ (-1/2), m * (g x₂ - g x₁) - x₂ * f x₂ + x₁ * f x₁ > 0 :=
sorry

end max_value_h_exists_m_for_inequality_l2493_249349


namespace exam_pass_probability_l2493_249301

/-- The probability of answering a single question correctly -/
def p : ℝ := 0.4

/-- The number of questions in the exam -/
def n : ℕ := 4

/-- The minimum number of correct answers required to pass -/
def k : ℕ := 3

/-- The probability of passing the exam -/
def pass_probability : ℝ := 
  (Nat.choose n k * p^k * (1-p)^(n-k)) + (p^n)

theorem exam_pass_probability : pass_probability = 112/625 := by
  sorry

end exam_pass_probability_l2493_249301


namespace max_value_of_sum_products_l2493_249310

theorem max_value_of_sum_products (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) 
  (h4 : a^2 + b^2 + c^2 = 3) : 
  ∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x^2 + y^2 + z^2 = 3 → 
  a * b + b * c + c * a ≥ x * y + y * z + z * x ∧
  a * b + b * c + c * a ≤ 3 :=
sorry

end max_value_of_sum_products_l2493_249310


namespace count_valid_pairs_l2493_249386

def satisfies_equation (x y : ℤ) : Prop :=
  2 * x^2 - 2 * x * y + y^2 = 289

def valid_pair (p : ℤ × ℤ) : Prop :=
  satisfies_equation p.1 p.2 ∧ p.1 ≥ 0

theorem count_valid_pairs :
  ∃ (S : Finset (ℤ × ℤ)), (∀ p ∈ S, valid_pair p) ∧ S.card = 7 ∧
  ∀ p : ℤ × ℤ, valid_pair p → p ∈ S :=
sorry

end count_valid_pairs_l2493_249386


namespace perfect_cube_in_range_l2493_249319

theorem perfect_cube_in_range (Y J : ℤ) : 
  (150 < Y) → (Y < 300) → (Y = J^5) → (∃ n : ℤ, Y = n^3) → J = 3 := by
  sorry

end perfect_cube_in_range_l2493_249319


namespace factory_machines_capping_l2493_249339

/-- Represents a machine in the factory -/
structure Machine where
  capping_rate : ℕ  -- bottles capped per minute
  working_time : ℕ  -- working time in minutes

/-- Calculates the total number of bottles capped by a machine -/
def total_capped (m : Machine) : ℕ := m.capping_rate * m.working_time

theorem factory_machines_capping (machine_a machine_b machine_c machine_d machine_e : Machine) :
  machine_a.capping_rate = 24 ∧
  machine_a.working_time = 10 ∧
  machine_b.capping_rate = machine_a.capping_rate - 3 ∧
  machine_b.working_time = 12 ∧
  machine_c.capping_rate = machine_b.capping_rate + 6 ∧
  machine_c.working_time = 15 ∧
  machine_d.capping_rate = machine_c.capping_rate - 4 ∧
  machine_d.working_time = 8 ∧
  machine_e.capping_rate = machine_d.capping_rate + 5 ∧
  machine_e.working_time = 5 →
  total_capped machine_a = 240 ∧
  total_capped machine_b = 252 ∧
  total_capped machine_c = 405 ∧
  total_capped machine_d = 184 ∧
  total_capped machine_e = 140 := by
  sorry

#check factory_machines_capping

end factory_machines_capping_l2493_249339


namespace no_solution_implies_m_equals_negative_five_l2493_249397

theorem no_solution_implies_m_equals_negative_five (m : ℝ) : 
  (∀ x : ℝ, x ≠ -1 → (3*x - 2)/(x + 1) ≠ 2 + m/(x + 1)) → m = -5 := by
  sorry

end no_solution_implies_m_equals_negative_five_l2493_249397


namespace exists_non_regular_triangle_with_similar_median_triangle_l2493_249306

/-- Represents a triangle with sides a, b, c and medians s_a, s_b, s_c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  s_a : ℝ
  s_b : ℝ
  s_c : ℝ
  h_order : a ≤ b ∧ b ≤ c
  h_median_a : 4 * s_a^2 = 2 * b^2 + 2 * c^2 - a^2
  h_median_b : 4 * s_b^2 = 2 * c^2 + 2 * a^2 - b^2
  h_median_c : 4 * s_c^2 = 2 * a^2 + 2 * b^2 - c^2

/-- Two triangles are similar if the ratios of their corresponding sides are equal -/
def similar (t1 t2 : Triangle) : Prop :=
  (t1.a / t2.a)^2 = (t1.b / t2.b)^2 ∧ (t1.b / t2.b)^2 = (t1.c / t2.c)^2

/-- A triangle is regular if all its sides are equal -/
def regular (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

theorem exists_non_regular_triangle_with_similar_median_triangle :
  ∃ t : Triangle, ¬regular t ∧ similar t ⟨t.s_a, t.s_b, t.s_c, 0, 0, 0, sorry, sorry, sorry, sorry⟩ :=
sorry

end exists_non_regular_triangle_with_similar_median_triangle_l2493_249306


namespace vertex_x_coordinate_l2493_249367

def f (x : ℝ) := 3 * x^2 + 9 * x + 5

theorem vertex_x_coordinate (x : ℝ) :
  x = -1.5 ↔ ∀ y : ℝ, f y ≥ f x :=
sorry

end vertex_x_coordinate_l2493_249367


namespace factorial_of_factorial_divided_by_factorial_l2493_249377

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end factorial_of_factorial_divided_by_factorial_l2493_249377


namespace meeting_problem_solution_l2493_249323

/-- Represents the problem of two people moving towards each other --/
structure MeetingProblem where
  total_distance : ℝ
  meeting_time : ℝ
  distance_difference : ℝ
  time_to_b_after_meeting : ℝ

/-- The solution to the meeting problem --/
structure MeetingSolution where
  speed_xiaogang : ℝ
  speed_xiaoqiang : ℝ
  time_to_a_after_meeting : ℝ

/-- Theorem stating the solution to the meeting problem --/
theorem meeting_problem_solution (p : MeetingProblem) 
  (h1 : p.meeting_time = 2)
  (h2 : p.distance_difference = 24)
  (h3 : p.time_to_b_after_meeting = 0.5) :
  ∃ (s : MeetingSolution),
    s.speed_xiaogang = 16 ∧
    s.speed_xiaoqiang = 4 ∧
    s.time_to_a_after_meeting = 8 := by
  sorry

end meeting_problem_solution_l2493_249323


namespace max_intersection_points_l2493_249395

/-- Given 20 points on the positive x-axis and 10 points on the positive y-axis,
    the maximum number of intersection points in the first quadrant formed by
    the segments connecting these points is equal to the product of
    combinations C(20,2) and C(10,2). -/
theorem max_intersection_points (x_points y_points : ℕ) 
  (hx : x_points = 20) (hy : y_points = 10) :
  (x_points.choose 2) * (y_points.choose 2) = 8550 := by
  sorry

end max_intersection_points_l2493_249395


namespace line_transformation_l2493_249300

-- Define the original line
def original_line (x : ℝ) : ℝ := x

-- Define rotation by 90 degrees counterclockwise
def rotate_90 (x y : ℝ) : ℝ × ℝ := (-y, x)

-- Define vertical shift by 1 unit
def shift_up (y : ℝ) : ℝ := y + 1

-- Theorem statement
theorem line_transformation :
  ∀ x : ℝ, 
  let (x', y') := rotate_90 x (original_line x)
  shift_up y' = -x' + 1 := by
  sorry

end line_transformation_l2493_249300


namespace max_value_is_six_range_of_m_l2493_249373

-- Define the problem setup
def problem_setup (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 6

-- Define the maximum value function
def max_value (a b c : ℝ) : ℝ := a + 2*b + c

-- Theorem for the maximum value
theorem max_value_is_six (a b c : ℝ) (h : problem_setup a b c) :
  ∃ (M : ℝ), (∀ (a' b' c' : ℝ), problem_setup a' b' c' → max_value a' b' c' ≤ M) ∧
             M = 6 :=
sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ (x : ℝ), |x + 1| + |x + m| ≥ 6) ↔ (m ≥ 7 ∨ m ≤ -5) :=
sorry

end max_value_is_six_range_of_m_l2493_249373


namespace simplify_radical_l2493_249380

theorem simplify_radical (a b : ℝ) (h : b > 0) :
  Real.sqrt (20 * a^3 * b^2) = 2 * a * b * Real.sqrt (5 * a) :=
by sorry

end simplify_radical_l2493_249380
