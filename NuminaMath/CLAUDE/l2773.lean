import Mathlib

namespace integral_equality_l2773_277366

theorem integral_equality : ∫ (x : ℝ) in (0 : ℝ)..(1 : ℝ), (Real.sqrt (1 - (x - 1)^2) - x) = (Real.pi - 2) / 4 := by
  sorry

end integral_equality_l2773_277366


namespace circles_intersect_l2773_277331

/-- Two circles are intersecting if the distance between their centers is greater than the absolute
    difference of their radii and less than the sum of their radii. -/
def are_circles_intersecting (r1 r2 d : ℝ) : Prop :=
  abs (r1 - r2) < d ∧ d < r1 + r2

/-- Given two circles with radii 4 and 3, and a distance of 5 between their centers,
    prove that they are intersecting. -/
theorem circles_intersect : are_circles_intersecting 4 3 5 := by
  sorry

end circles_intersect_l2773_277331


namespace license_plate_difference_l2773_277399

/-- The number of possible license plates for Sunland -/
def sunland_plates : ℕ := 1 * (10^3) * (26^2)

/-- The number of possible license plates for Moonland -/
def moonland_plates : ℕ := (10^2) * (26^2) * (10^2)

/-- The theorem stating the difference in the number of license plates -/
theorem license_plate_difference : moonland_plates - sunland_plates = 6084000 := by
  sorry

end license_plate_difference_l2773_277399


namespace population_growth_l2773_277333

theorem population_growth (p q : ℕ) (h1 : p^2 + 180 = q^2 + 16) 
  (h2 : ∃ r : ℕ, p^2 + 360 = r^2) : 
  abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 21) < 
  min (abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 18))
      (min (abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 24))
           (min (abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 27))
                (abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 30)))) :=
by sorry

end population_growth_l2773_277333


namespace clara_older_than_alice_l2773_277355

/-- Represents a person with their age and number of pens -/
structure Person where
  age : ℕ
  pens : ℕ

/-- The problem statement -/
theorem clara_older_than_alice (alice clara : Person)
  (h1 : alice.pens = 60)
  (h2 : clara.pens = 2 * alice.pens / 5)
  (h3 : alice.pens - clara.pens = alice.age - clara.age)
  (h4 : alice.age = 20)
  (h5 : clara.age + 5 = 61) :
  clara.age > alice.age := by
  sorry

#check clara_older_than_alice

end clara_older_than_alice_l2773_277355


namespace x_value_l2773_277397

theorem x_value (w y z x : ℤ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 12)
  (hx : x = y + 7) : x = 134 := by
  sorry

end x_value_l2773_277397


namespace arithmetic_geometric_sequence_closed_form_l2773_277384

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a b : ℝ) (u₀ : ℝ) : ℕ → ℝ
  | 0 => u₀
  | n + 1 => a * ArithmeticGeometricSequence a b u₀ n + b

/-- Theorem for the closed form of an arithmetic-geometric sequence -/
theorem arithmetic_geometric_sequence_closed_form (a b u₀ : ℝ) (ha : a ≠ 1) :
  ∀ n : ℕ, ArithmeticGeometricSequence a b u₀ n = a^n * u₀ + b * (a^n - 1) / (a - 1) :=
by sorry

end arithmetic_geometric_sequence_closed_form_l2773_277384


namespace chords_from_nine_points_l2773_277308

/-- The number of different chords that can be drawn by connecting two points 
    out of n points on the circumference of a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem stating that the number of chords from 9 points is 36 -/
theorem chords_from_nine_points : num_chords 9 = 36 := by
  sorry

end chords_from_nine_points_l2773_277308


namespace equal_tape_length_l2773_277329

def minyoung_tape : ℕ := 1748
def yoojung_tape : ℕ := 850
def tape_to_give : ℕ := 449

theorem equal_tape_length : 
  minyoung_tape - tape_to_give = yoojung_tape + tape_to_give :=
by sorry

end equal_tape_length_l2773_277329


namespace arithmetic_geometric_properties_l2773_277386

-- Define the arithmetic-geometric sequence
def arithmetic_geometric (a b : ℝ) (u : ℕ → ℝ) : Prop :=
  ∀ n, u (n + 1) = a * u n + b

-- Define another sequence satisfying the same recurrence relation
def same_recurrence (a b : ℝ) (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 1) = a * v n + b

-- Define the sequence w as the difference of u and v
def w (u v : ℕ → ℝ) : ℕ → ℝ :=
  λ n => u n - v n

-- State the theorem
theorem arithmetic_geometric_properties
  (a b : ℝ)
  (u v : ℕ → ℝ)
  (hu : arithmetic_geometric a b u)
  (hv : same_recurrence a b v)
  (ha : a ≠ 1) :
  (∀ n, w u v (n + 1) = a * w u v n) ∧
  (∃ c : ℝ, ∀ n, v n = c ∧ c = b / (1 - a)) ∧
  (∀ n, u n = a^n * (u 0 - b/(1-a)) + b/(1-a)) :=
sorry

end arithmetic_geometric_properties_l2773_277386


namespace house_rent_fraction_l2773_277332

theorem house_rent_fraction (salary : ℕ) (food_fraction : ℚ) (clothes_fraction : ℚ) (remaining : ℕ) 
  (h1 : salary = 160000)
  (h2 : food_fraction = 1/5)
  (h3 : clothes_fraction = 3/5)
  (h4 : remaining = 16000)
  (h5 : ∃ (house_rent_fraction : ℚ), salary * (1 - food_fraction - clothes_fraction - house_rent_fraction) = remaining) :
  ∃ (house_rent_fraction : ℚ), house_rent_fraction = 1/10 := by
sorry

end house_rent_fraction_l2773_277332


namespace symmetry_properties_l2773_277354

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetry_x_axis (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, -p.z⟩

def symmetry_yOz_plane (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, p.z⟩

def symmetry_y_axis (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, -p.z⟩

def symmetry_origin (p : Point3D) : Point3D :=
  ⟨-p.x, -p.y, -p.z⟩

theorem symmetry_properties (p : Point3D) :
  (symmetry_x_axis p = ⟨p.x, -p.y, -p.z⟩) ∧
  (symmetry_yOz_plane p = ⟨-p.x, p.y, p.z⟩) ∧
  (symmetry_y_axis p = ⟨-p.x, p.y, -p.z⟩) ∧
  (symmetry_origin p = ⟨-p.x, -p.y, -p.z⟩) := by
  sorry

end symmetry_properties_l2773_277354


namespace arctan_equation_solution_l2773_277369

theorem arctan_equation_solution (y : ℝ) :
  2 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4 →
  y = -121/60 := by
  sorry

end arctan_equation_solution_l2773_277369


namespace cycle_original_price_l2773_277347

/-- Given a cycle sold at a 25% loss for Rs. 1350, prove its original price was Rs. 1800. -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) :
  selling_price = 1350 →
  loss_percentage = 25 →
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧
    original_price = 1800 :=
by sorry

end cycle_original_price_l2773_277347


namespace regular_pay_is_three_l2773_277375

/-- Calculates the regular hourly pay rate given total pay, regular hours, overtime hours, and overtime pay rate multiplier. -/
def regularHourlyPay (totalPay : ℚ) (regularHours : ℚ) (overtimeHours : ℚ) (overtimeMultiplier : ℚ) : ℚ :=
  totalPay / (regularHours + overtimeHours * overtimeMultiplier)

/-- Proves that the regular hourly pay is $3 given the problem conditions. -/
theorem regular_pay_is_three :
  let totalPay : ℚ := 192
  let regularHours : ℚ := 40
  let overtimeHours : ℚ := 12
  let overtimeMultiplier : ℚ := 2
  regularHourlyPay totalPay regularHours overtimeHours overtimeMultiplier = 3 := by
  sorry

#eval regularHourlyPay 192 40 12 2

end regular_pay_is_three_l2773_277375


namespace data_instances_eq_720_l2773_277307

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The interval between recordings in seconds -/
def recording_interval : ℕ := 5

/-- The number of data instances recorded in one hour by a device that records every 5 seconds -/
def data_instances : ℕ := 
  (seconds_per_minute * minutes_per_hour) / recording_interval

/-- Theorem: The number of data instances recorded in one hour is 720 -/
theorem data_instances_eq_720 : data_instances = 720 := by
  sorry

end data_instances_eq_720_l2773_277307


namespace parallel_condition_l2773_277314

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the subset relation for lines and planes
variable (line_in_plane : Line → Plane → Prop)

-- Main theorem
theorem parallel_condition 
  (α β : Plane) 
  (m : Line) 
  (h_distinct : α ≠ β) 
  (h_m_in_α : line_in_plane m α) :
  (∀ α β : Plane, plane_parallel α β → line_parallel_plane m β) ∧ 
  (∃ α β : Plane, line_parallel_plane m β ∧ ¬plane_parallel α β) :=
sorry

end parallel_condition_l2773_277314


namespace composite_sequence_existence_l2773_277363

theorem composite_sequence_existence (m : ℕ) (hm : m > 0) :
  ∃ n : ℕ, ∀ i : ℤ, -m ≤ i ∧ i ≤ m → 
    (2 : ℕ)^n + i > 0 ∧ ¬(Nat.Prime ((2 : ℕ)^n + i).toNat) := by
  sorry

end composite_sequence_existence_l2773_277363


namespace ali_fish_weight_l2773_277339

theorem ali_fish_weight (peter_weight joey_weight ali_weight : ℝ) 
  (h1 : ali_weight = 2 * peter_weight)
  (h2 : joey_weight = peter_weight + 1)
  (h3 : peter_weight + joey_weight + ali_weight = 25) :
  ali_weight = 12 := by
sorry

end ali_fish_weight_l2773_277339


namespace share_calculation_l2773_277385

/-- The amount y gets for each rupee x gets -/
def a : ℝ := 0.45

/-- The share of y in rupees -/
def y : ℝ := 63

/-- The total amount in rupees -/
def total : ℝ := 273

theorem share_calculation (x : ℝ) :
  x > 0 →
  x + a * x + 0.5 * x = total ∧
  a * x = y →
  a = 0.45 := by
  sorry

end share_calculation_l2773_277385


namespace integral_reciprocal_one_plus_x_squared_l2773_277320

theorem integral_reciprocal_one_plus_x_squared : 
  ∫ (x : ℝ) in (0)..(Real.sqrt 3), 1 / (1 + x^2) = π / 3 := by
  sorry

end integral_reciprocal_one_plus_x_squared_l2773_277320


namespace max_min_x_squared_l2773_277326

def f (x : ℝ) : ℝ := x^2

theorem max_min_x_squared :
  ∃ (max min : ℝ), 
    (∀ x, -3 ≤ x ∧ x ≤ 1 → f x ≤ max) ∧
    (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = max) ∧
    (∀ x, -3 ≤ x ∧ x ≤ 1 → min ≤ f x) ∧
    (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = min) ∧
    max = 9 ∧ min = 0 := by
  sorry

end max_min_x_squared_l2773_277326


namespace arithmetic_geometric_means_l2773_277318

theorem arithmetic_geometric_means (a b c x y : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- a, b, c are positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧  -- a, b, c are distinct
  2 * b = a + c ∧          -- a, b, c form an arithmetic sequence
  x^2 = a * b ∧            -- x is the geometric mean of a and b
  y^2 = b * c →            -- y is the geometric mean of b and c
  (2 * b^2 = x^2 + y^2) ∧  -- x^2, b^2, y^2 form an arithmetic sequence
  (b^4 ≠ x^2 * y^2)        -- x^2, b^2, y^2 do not form a geometric sequence
  := by sorry

end arithmetic_geometric_means_l2773_277318


namespace E_is_top_leftmost_l2773_277346

-- Define the structure for a rectangle
structure Rectangle where
  w : Int
  x : Int
  y : Int
  z : Int

-- Define the five rectangles
def A : Rectangle := { w := 4, x := 1, y := 6, z := 9 }
def B : Rectangle := { w := 1, x := 0, y := 3, z := 6 }
def C : Rectangle := { w := 3, x := 8, y := 5, z := 2 }
def D : Rectangle := { w := 7, x := 5, y := 4, z := 8 }
def E : Rectangle := { w := 9, x := 2, y := 7, z := 0 }

-- Define the placement rules
def isLeftmost (r : Rectangle) : Bool :=
  r.w = 1 ∨ r.w = 9

def isRightmost (r : Rectangle) : Bool :=
  r.y = 6 ∨ r.y = 5

def isCenter (r : Rectangle) : Bool :=
  ¬(isLeftmost r) ∧ ¬(isRightmost r)

-- Theorem to prove
theorem E_is_top_leftmost :
  isLeftmost E ∧ 
  isRightmost A ∧ 
  isRightmost C ∧ 
  isLeftmost B ∧ 
  isCenter D :=
sorry

end E_is_top_leftmost_l2773_277346


namespace hyperbola_eccentricity_specific_hyperbola_eccentricity_l2773_277383

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) / a -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  ∀ x y, hyperbola x y → e = Real.sqrt 5 / 2 :=
by
  sorry

/-- The eccentricity of the hyperbola x²/4 - y² = 1 is √5/2 -/
theorem specific_hyperbola_eccentricity :
  let e := Real.sqrt 5 / 2
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 4 - y^2 = 1
  ∀ x y, hyperbola x y → e = Real.sqrt 5 / 2 :=
by
  sorry

end hyperbola_eccentricity_specific_hyperbola_eccentricity_l2773_277383


namespace cube_digit_sum_l2773_277391

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for nine-digit numbers -/
def is_nine_digit (n : ℕ) : Prop := sorry

theorem cube_digit_sum (N : ℕ) (h1 : is_nine_digit N) (h2 : sum_of_digits N = 3) :
  sum_of_digits (N^3) = 9 ∨ sum_of_digits (N^3) = 18 ∨ sum_of_digits (N^3) = 27 := by sorry

end cube_digit_sum_l2773_277391


namespace cubic_sum_from_system_l2773_277398

theorem cubic_sum_from_system (x y : ℝ) 
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) : 
  x^3 + y^3 = 416000 / 729 := by
sorry

end cubic_sum_from_system_l2773_277398


namespace max_pencils_is_13_l2773_277389

def john_money : ℚ := 10
def regular_price : ℚ := 0.75
def discount_price : ℚ := 0.65
def discount_threshold : ℕ := 10

def cost (n : ℕ) : ℚ :=
  if n ≤ discount_threshold then
    n * regular_price
  else
    discount_threshold * regular_price + (n - discount_threshold) * discount_price

def can_afford (n : ℕ) : Prop :=
  cost n ≤ john_money

theorem max_pencils_is_13 :
  ∀ n : ℕ, can_afford n → n ≤ 13 ∧
  ∃ m : ℕ, m = 13 ∧ can_afford m :=
by sorry

end max_pencils_is_13_l2773_277389


namespace ariel_fencing_start_year_l2773_277300

def birth_year : ℕ := 1992
def current_age : ℕ := 30
def fencing_years : ℕ := 16

theorem ariel_fencing_start_year :
  birth_year + current_age - fencing_years = 2006 :=
by sorry

end ariel_fencing_start_year_l2773_277300


namespace power_product_eq_four_l2773_277379

theorem power_product_eq_four (a b : ℕ+) (h : (3 ^ a.val) ^ b.val = 3 ^ 3) :
  3 ^ a.val * 3 ^ b.val = 3 ^ 4 := by
  sorry

end power_product_eq_four_l2773_277379


namespace total_books_calculation_l2773_277316

theorem total_books_calculation (joan_books tom_books lisa_books steve_books : ℕ) 
  (h1 : joan_books = 10)
  (h2 : tom_books = 38)
  (h3 : lisa_books = 27)
  (h4 : steve_books = 45) :
  joan_books + tom_books + lisa_books + steve_books = 120 := by
sorry

end total_books_calculation_l2773_277316


namespace sqrt_equality_implies_one_five_l2773_277382

theorem sqrt_equality_implies_one_five (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) :
  (Real.sqrt (1 + Real.sqrt (45 + 20 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b) →
  (a = 1 ∧ b = 5) := by
sorry

end sqrt_equality_implies_one_five_l2773_277382


namespace inequality_proof_l2773_277340

theorem inequality_proof (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  (a * b + b * c + c * a ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) := by
  sorry

end inequality_proof_l2773_277340


namespace problem_statement_l2773_277387

theorem problem_statement (a b x y : ℝ) 
  (eq1 : a * x + b * y = 3)
  (eq2 : a * x^2 + b * y^2 = 7)
  (eq3 : a * x^3 + b * y^3 = 16)
  (eq4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
  sorry

end problem_statement_l2773_277387


namespace berrys_friday_temperature_l2773_277305

/-- Given Berry's temperatures for 6 days and the average for a week, 
    prove that his temperature on Friday was 99 degrees. -/
theorem berrys_friday_temperature 
  (temps : List ℝ) 
  (h_temps : temps = [99.1, 98.2, 98.7, 99.3, 99.8, 98.9]) 
  (h_avg : (temps.sum + x) / 7 = 99) : x = 99 := by
  sorry

end berrys_friday_temperature_l2773_277305


namespace random_walk_prob_4_in_3_to_9_l2773_277394

/-- A one-dimensional random walk on integers -/
def RandomWalk := ℕ → ℤ

/-- The probability of a random walk reaching a specific distance -/
def prob_reach_distance (w : RandomWalk) (d : ℕ) (steps : ℕ) : ℚ :=
  sorry

/-- The probability of a random walk reaching a specific distance at least once within a range of steps -/
def prob_reach_distance_in_range (w : RandomWalk) (d : ℕ) (min_steps max_steps : ℕ) : ℚ :=
  sorry

/-- The main theorem: probability of reaching distance 4 at least once in 3 to 9 steps is 47/224 -/
theorem random_walk_prob_4_in_3_to_9 (w : RandomWalk) :
  prob_reach_distance_in_range w 4 3 9 = 47 / 224 := by
  sorry

end random_walk_prob_4_in_3_to_9_l2773_277394


namespace part_one_part_two_l2773_277312

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part_one (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 4 → f a x ≤ 2) : a = 2 := by
  sorry

-- Part 2
theorem part_two (a : ℝ) (h : 0 ≤ a ∧ a ≤ 3) :
  ∀ x : ℝ, f a (x + a) + f a (x - a) ≥ f a (a * x) - a * f a x := by
  sorry

end part_one_part_two_l2773_277312


namespace ellipse_foci_on_y_axis_l2773_277388

theorem ellipse_foci_on_y_axis (k : ℝ) : 
  (∀ x y : ℝ, x^2 / (2 - k) + y^2 / (2*k - 1) = 1 → 
    (∃ c : ℝ, c > 0 ∧ 
      ∀ p : ℝ × ℝ, 
        (p.1 = 0 → (p.2 = c ∨ p.2 = -c)) ∧ 
        (p.2 = c ∨ p.2 = -c → p.1 = 0))) → 
  1 < k ∧ k < 2 :=
by sorry

end ellipse_foci_on_y_axis_l2773_277388


namespace constant_kill_time_l2773_277361

/-- Represents the time taken for lions to kill deers -/
def killTime (numLions : ℕ) : ℕ := 13

/-- The assumption that 13 lions can kill 13 deers in 13 minutes -/
axiom base_case : killTime 13 = 13

/-- Theorem stating that for any number of lions (equal to deers), 
    the time taken to kill all deers is always 13 minutes -/
theorem constant_kill_time (n : ℕ) : killTime n = 13 := by
  sorry

end constant_kill_time_l2773_277361


namespace six_digit_numbers_with_zero_l2773_277370

theorem six_digit_numbers_with_zero (total_six_digit : Nat) (six_digit_no_zero : Nat) :
  total_six_digit = 900000 →
  six_digit_no_zero = 531441 →
  total_six_digit - six_digit_no_zero = 368559 := by
  sorry

end six_digit_numbers_with_zero_l2773_277370


namespace least_perimeter_triangle_l2773_277368

/-- 
Given a triangle with two sides of 27 units and 34 units, and the third side having an integral length,
the least possible perimeter is 69 units.
-/
theorem least_perimeter_triangle : 
  ∀ z : ℕ, 
  z > 0 → 
  z + 27 > 34 → 
  34 + 27 > z → 
  27 + z > 34 → 
  ∀ w : ℕ, 
  w > 0 → 
  w + 27 > 34 → 
  34 + 27 > w → 
  27 + w > 34 → 
  w ≥ z → 
  27 + 34 + w ≥ 69 :=
by sorry

end least_perimeter_triangle_l2773_277368


namespace original_number_l2773_277302

theorem original_number (x : ℝ) (h : 5 * x - 9 = 51) : x = 12 := by
  sorry

end original_number_l2773_277302


namespace arithmetic_mean_of_first_three_composite_reciprocals_l2773_277334

/-- The arithmetic mean of the reciprocals of the first three composite numbers is 13/72. -/
theorem arithmetic_mean_of_first_three_composite_reciprocals :
  (1 / 4 + 1 / 6 + 1 / 8) / 3 = 13 / 72 := by
  sorry

end arithmetic_mean_of_first_three_composite_reciprocals_l2773_277334


namespace inequality_solution_l2773_277358

theorem inequality_solution (a : ℝ) :
  (a = 1/2 → ∀ x, (x - a) * (x + a - 1) > 0 ↔ x ≠ 1/2) ∧
  (a < 1/2 → ∀ x, (x - a) * (x + a - 1) > 0 ↔ x > a ∨ x < 1 - a) ∧
  (a > 1/2 → ∀ x, (x - a) * (x + a - 1) > 0 ↔ x > a ∨ x < 1 - a) :=
by sorry

end inequality_solution_l2773_277358


namespace meeting_time_and_distance_l2773_277348

/-- Represents the time in hours since 7:45 AM -/
def time_since_start : ℝ → ℝ := λ t => t

/-- Samantha's speed in miles per hour -/
def samantha_speed : ℝ := 15

/-- Adam's speed in miles per hour -/
def adam_speed : ℝ := 20

/-- Time difference between Samantha's and Adam's start times in hours -/
def start_time_diff : ℝ := 0.5

/-- Total distance between Town A and Town B in miles -/
def total_distance : ℝ := 75

/-- Calculates Samantha's traveled distance at time t -/
def samantha_distance (t : ℝ) : ℝ := samantha_speed * t

/-- Calculates Adam's traveled distance at time t -/
def adam_distance (t : ℝ) : ℝ := adam_speed * (t - start_time_diff)

/-- Theorem stating the meeting time and Samantha's traveled distance -/
theorem meeting_time_and_distance :
  ∃ t : ℝ, 
    samantha_distance t + adam_distance t = total_distance ∧
    time_since_start t = 2.4333333333333 ∧ 
    samantha_distance t = 36 := by
  sorry

end meeting_time_and_distance_l2773_277348


namespace z_range_is_closed_interval_l2773_277396

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define z as a function of x and y
def z (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem z_range_is_closed_interval :
  ∃ (a b : ℝ), a = -5 ∧ b = 5 ∧
  (∀ (x y : ℝ), ellipse_equation x y → a ≤ z x y ∧ z x y ≤ b) ∧
  (∀ t : ℝ, a ≤ t ∧ t ≤ b → ∃ (x y : ℝ), ellipse_equation x y ∧ z x y = t) :=
sorry

end z_range_is_closed_interval_l2773_277396


namespace hockey_arena_rows_l2773_277395

/-- The minimum number of rows required in a hockey arena -/
def min_rows (seats_per_row : ℕ) (total_students : ℕ) (max_students_per_school : ℕ) : ℕ :=
  let schools_per_row := seats_per_row / max_students_per_school
  let total_schools := (total_students + max_students_per_school - 1) / max_students_per_school
  (total_schools + schools_per_row - 1) / schools_per_row

/-- Theorem stating the minimum number of rows required for the given conditions -/
theorem hockey_arena_rows :
  min_rows 168 2016 45 = 16 := by
  sorry

#eval min_rows 168 2016 45

end hockey_arena_rows_l2773_277395


namespace f_composition_quarter_l2773_277324

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 4
  else 2^x

theorem f_composition_quarter : f (f (1/4)) = 1/2 := by
  sorry

end f_composition_quarter_l2773_277324


namespace point_movement_l2773_277373

/-- Represents a point on a number line -/
structure Point where
  value : ℤ

/-- Moves a point on the number line -/
def movePoint (p : Point) (distance : ℤ) : Point :=
  { value := p.value + distance }

theorem point_movement :
  let a : Point := { value := -3 }
  let b : Point := movePoint a 7
  b.value = 4 := by sorry

end point_movement_l2773_277373


namespace min_distance_to_origin_l2773_277343

/-- The minimum distance from any point on the line x + y - 4 = 0 to the origin (0, 0) is 2√2 -/
theorem min_distance_to_origin : 
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  ∀ p ∈ line, Real.sqrt ((p.1 ^ 2) + (p.2 ^ 2)) ≥ 2 * Real.sqrt 2 :=
by sorry

end min_distance_to_origin_l2773_277343


namespace six_balls_three_boxes_l2773_277341

/-- Number of partitions of n into at most k parts -/
def num_partitions (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to put 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : num_partitions 6 3 = 7 := by sorry

end six_balls_three_boxes_l2773_277341


namespace consecutive_divisibility_l2773_277303

theorem consecutive_divisibility (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  ∀ (start : ℕ), ∃ (x y z : ℕ), 
    (x ∈ Finset.range (2 * c) ∧ y ∈ Finset.range (2 * c) ∧ z ∈ Finset.range (2 * c)) ∧
    (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
    ((a * b * c) ∣ (x * y * z)) :=
by sorry

end consecutive_divisibility_l2773_277303


namespace coffee_shop_run_time_l2773_277337

/-- Represents the time in minutes to run a given distance at a constant pace -/
def runTime (distance : ℝ) (pace : ℝ) : ℝ := distance * pace

theorem coffee_shop_run_time :
  let parkDistance : ℝ := 5
  let parkTime : ℝ := 30
  let coffeeShopDistance : ℝ := 2
  let pace : ℝ := parkTime / parkDistance
  runTime coffeeShopDistance pace = 12 := by sorry

end coffee_shop_run_time_l2773_277337


namespace savings_percentage_l2773_277380

-- Define the original prices and discount rates
def coat_price : ℝ := 120
def hat_price : ℝ := 30
def gloves_price : ℝ := 50

def coat_discount : ℝ := 0.20
def hat_discount : ℝ := 0.40
def gloves_discount : ℝ := 0.30

-- Define the total original cost
def total_original_cost : ℝ := coat_price + hat_price + gloves_price

-- Define the savings for each item
def coat_savings : ℝ := coat_price * coat_discount
def hat_savings : ℝ := hat_price * hat_discount
def gloves_savings : ℝ := gloves_price * gloves_discount

-- Define the total savings
def total_savings : ℝ := coat_savings + hat_savings + gloves_savings

-- Theorem to prove
theorem savings_percentage :
  (total_savings / total_original_cost) * 100 = 25.5 := by
  sorry

end savings_percentage_l2773_277380


namespace stratified_sample_size_l2773_277317

/-- Represents the number of athletes in a sample -/
structure Sample where
  male : ℕ
  female : ℕ

/-- Represents the total population of athletes -/
structure Population where
  male : ℕ
  female : ℕ

/-- Checks if a sample is stratified with respect to a population -/
def isStratifiedSample (pop : Population) (samp : Sample) : Prop :=
  samp.male * pop.female = samp.female * pop.male

/-- The main theorem to prove -/
theorem stratified_sample_size 
  (pop : Population) 
  (samp : Sample) 
  (h1 : pop.male = 42)
  (h2 : pop.female = 30)
  (h3 : samp.female = 5)
  (h4 : isStratifiedSample pop samp) :
  samp.male + samp.female = 12 := by
  sorry


end stratified_sample_size_l2773_277317


namespace geometric_sequence_general_term_l2773_277344

/-- A geometric sequence {a_n} satisfying given conditions has the general term formula a_n = 1 / (2^(n-4)) -/
theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_sum1 : a 1 + a 3 = 10) 
  (h_sum2 : a 4 + a 6 = 5/4) : 
  ∀ n : ℕ, n ≥ 1 → a n = 1 / (2^(n-4)) :=
sorry

end geometric_sequence_general_term_l2773_277344


namespace expand_product_l2773_277381

theorem expand_product (x : ℝ) : (x + 3) * (x - 8) = x^2 - 5*x - 24 := by
  sorry

end expand_product_l2773_277381


namespace f_2021_2_l2773_277374

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_2021_2 (f : ℝ → ℝ) 
  (h1 : is_even_function f)
  (h2 : ∀ x, f (x + 2) = -f x)
  (h3 : ∀ x ∈ Set.Ioo 1 2, f x = 2^x) :
  f (2021/2) = 2 * Real.sqrt 2 := by
  sorry

end f_2021_2_l2773_277374


namespace closure_of_M_union_N_l2773_277350

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem closure_of_M_union_N :
  closure (M ∪ N) = {x : ℝ | x ≤ 1} := by sorry

end closure_of_M_union_N_l2773_277350


namespace decimal_equivalent_one_fourth_power_one_l2773_277362

theorem decimal_equivalent_one_fourth_power_one :
  (1 / 4 : ℚ) ^ (1 : ℕ) = 0.25 := by sorry

end decimal_equivalent_one_fourth_power_one_l2773_277362


namespace new_average_weight_l2773_277365

/-- Given 19 students with an average weight of 15 kg and a new student weighing 11 kg,
    the new average weight of all 20 students is 14.8 kg. -/
theorem new_average_weight (initial_students : ℕ) (initial_avg_weight : ℝ) 
  (new_student_weight : ℝ) : 
  initial_students = 19 → 
  initial_avg_weight = 15 → 
  new_student_weight = 11 → 
  (initial_students * initial_avg_weight + new_student_weight) / (initial_students + 1) = 14.8 := by
  sorry

end new_average_weight_l2773_277365


namespace first_player_can_ensure_non_trivial_solution_l2773_277309

-- Define the system of equations
structure LinearSystem :=
  (eq1 eq2 eq3 : ℝ → ℝ → ℝ → ℝ)

-- Define the game state
structure GameState :=
  (system : LinearSystem)
  (player_turn : Bool)

-- Define a strategy for the first player
def FirstPlayerStrategy : GameState → GameState := sorry

-- Define a strategy for the second player
def SecondPlayerStrategy : GameState → GameState := sorry

-- Theorem statement
theorem first_player_can_ensure_non_trivial_solution :
  ∀ (initial_state : GameState),
  ∃ (x y z : ℝ), 
    (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    (initial_state.system.eq1 x y z = 0) ∧
    (initial_state.system.eq2 x y z = 0) ∧
    (initial_state.system.eq3 x y z = 0) :=
sorry

end first_player_can_ensure_non_trivial_solution_l2773_277309


namespace infinitely_many_inequality_holds_l2773_277356

theorem infinitely_many_inequality_holds (a : ℕ → ℝ) (h : ∀ n, a n > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, 1 + a n > a (n - 1) * (2 : ℝ) ^ (1 / n) :=
sorry

end infinitely_many_inequality_holds_l2773_277356


namespace fraction_inequality_solution_l2773_277338

theorem fraction_inequality_solution (x : ℝ) : 
  (x ≠ 5) → (x / (x - 5) ≥ 0 ↔ x ∈ Set.Ici 5 ∪ Set.Iic 0) :=
by sorry

end fraction_inequality_solution_l2773_277338


namespace right_triangle_to_square_l2773_277349

theorem right_triangle_to_square (a b : ℝ) : 
  b = 10 → -- longer leg is 10
  a * b / 2 = a^2 → -- area of triangle equals area of square
  b = 2 * a → -- longer leg is twice the shorter leg
  a = 5 := by
sorry

end right_triangle_to_square_l2773_277349


namespace retail_profit_calculation_l2773_277310

/-- Represents the pricing and profit calculations for a retail scenario -/
def RetailScenario (costPrice : ℝ) : Prop :=
  let markupPercentage : ℝ := 65
  let discountPercentage : ℝ := 25
  let actualProfitPercentage : ℝ := 23.75
  let markedPrice : ℝ := costPrice * (1 + markupPercentage / 100)
  let sellingPrice : ℝ := markedPrice * (1 - discountPercentage / 100)
  let actualProfit : ℝ := sellingPrice - costPrice
  let intendedProfit : ℝ := markedPrice - costPrice
  (actualProfit / costPrice * 100 = actualProfitPercentage) ∧
  (intendedProfit / costPrice * 100 = markupPercentage)

/-- Theorem stating that under the given retail scenario, the initially expected profit percentage is 65% -/
theorem retail_profit_calculation (costPrice : ℝ) (h : costPrice > 0) :
  RetailScenario costPrice → 65 = (65 : ℝ) :=
by
  sorry

end retail_profit_calculation_l2773_277310


namespace test_results_problem_l2773_277322

/-- Represents the number of questions a person got wrong on a test. -/
structure TestResult where
  wrong : Nat

/-- Represents the test results for Emily, Felix, Grace, and Henry. -/
structure GroupTestResults where
  emily : TestResult
  felix : TestResult
  grace : TestResult
  henry : TestResult

/-- The theorem statement for the test results problem. -/
theorem test_results_problem (results : GroupTestResults) : 
  (results.emily.wrong + results.felix.wrong + 4 = results.grace.wrong + results.henry.wrong) →
  (results.emily.wrong + results.henry.wrong = results.felix.wrong + results.grace.wrong + 8) →
  (results.grace.wrong = 6) →
  (results.emily.wrong = 8) := by
  sorry

#check test_results_problem

end test_results_problem_l2773_277322


namespace gcd_problem_l2773_277301

theorem gcd_problem (a : ℕ+) : (Nat.gcd (Nat.gcd a 16) (Nat.gcd 18 a) = 2) → (a = 2) :=
by sorry

end gcd_problem_l2773_277301


namespace mr_mcpherson_contribution_l2773_277335

def total_rent : ℝ := 1200
def mrs_mcpherson_percentage : ℝ := 30

theorem mr_mcpherson_contribution :
  total_rent - (mrs_mcpherson_percentage / 100 * total_rent) = 840 := by
  sorry

end mr_mcpherson_contribution_l2773_277335


namespace ceiling_sqrt_200_l2773_277390

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by sorry

end ceiling_sqrt_200_l2773_277390


namespace student_count_l2773_277377

theorem student_count : ∃ S : ℕ, 
  (S / 3 : ℚ) + 10 = S - 6 ∧ S = 24 := by sorry

end student_count_l2773_277377


namespace triangle_side_lengths_l2773_277359

theorem triangle_side_lengths 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : c = 10)
  (h2 : Real.cos A / Real.cos B = b / a)
  (h3 : b / a = 4 / 3) :
  a = 6 ∧ b = 8 := by
sorry

end triangle_side_lengths_l2773_277359


namespace magnitude_of_complex_fourth_power_l2773_277364

theorem magnitude_of_complex_fourth_power :
  Complex.abs ((5 : ℂ) + (2 * Complex.I * Real.sqrt 3)) ^ 4 = 1369 := by
  sorry

end magnitude_of_complex_fourth_power_l2773_277364


namespace smallest_angle_satisfying_trig_equation_l2773_277306

theorem smallest_angle_satisfying_trig_equation :
  ∃ y : ℝ, y > 0 ∧ y < (π / 180) * 360 ∧
  (∀ θ : ℝ, 0 < θ ∧ θ < y → ¬(Real.sin (4 * θ) * Real.sin (5 * θ) = Real.cos (4 * θ) * Real.cos (5 * θ))) ∧
  Real.sin (4 * y) * Real.sin (5 * y) = Real.cos (4 * y) * Real.cos (5 * y) ∧
  y = (π / 180) * 10 :=
sorry

end smallest_angle_satisfying_trig_equation_l2773_277306


namespace expression_evaluation_l2773_277328

theorem expression_evaluation (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h : y - z / x ≠ 0) : 
  (x - z / y) / (y - z / x) = x / y := by
  sorry

end expression_evaluation_l2773_277328


namespace new_ratio_after_removing_clothing_l2773_277378

/-- Represents the ratio of books to clothes to electronics -/
structure Ratio :=
  (books : ℕ)
  (clothes : ℕ)
  (electronics : ℕ)

/-- Calculates the new ratio of books to clothes after removing some clothing -/
def newRatio (initial : Ratio) (electronicsWeight : ℕ) (clothingRemoved : ℕ) : Ratio :=
  sorry

/-- Theorem stating the new ratio after removing clothing -/
theorem new_ratio_after_removing_clothing 
  (initial : Ratio)
  (electronicsWeight : ℕ)
  (clothingRemoved : ℕ)
  (h1 : initial = ⟨7, 4, 3⟩)
  (h2 : electronicsWeight = 9)
  (h3 : clothingRemoved = 6) :
  (newRatio initial electronicsWeight clothingRemoved).books = 7 ∧
  (newRatio initial electronicsWeight clothingRemoved).clothes = 2 :=
by sorry

end new_ratio_after_removing_clothing_l2773_277378


namespace nearest_multiple_21_l2773_277325

theorem nearest_multiple_21 (x : ℤ) : 
  (∀ y : ℤ, y % 21 = 0 → |x - 2319| ≤ |x - y|) → x = 2318 :=
sorry

end nearest_multiple_21_l2773_277325


namespace perpendicular_tangents_intersection_y_coord_l2773_277319

/-- The y-coordinate of the intersection point of perpendicular tangents to y = 4x^2 -/
theorem perpendicular_tangents_intersection_y_coord (c d : ℝ) : 
  (c ≠ d) →                                  -- Ensure C and D are distinct points
  (4 * c^2 = (4 : ℝ) * c^2) →                -- C is on the parabola y = 4x^2
  (4 * d^2 = (4 : ℝ) * d^2) →                -- D is on the parabola y = 4x^2
  ((8 : ℝ) * c * (8 * d) = -1) →             -- Tangent lines are perpendicular
  (4 : ℝ) * c * d = -(1/16) :=               -- y-coordinate of intersection point Q is -1/16
by sorry

end perpendicular_tangents_intersection_y_coord_l2773_277319


namespace opposite_numbers_equation_l2773_277311

theorem opposite_numbers_equation (x : ℝ) : 2 * (x - 3) = -(4 * (1 - x)) → x = 1 := by
  sorry

end opposite_numbers_equation_l2773_277311


namespace overtime_calculation_l2773_277357

/-- Calculates the number of overtime hours worked given the total gross pay, regular hourly rate, overtime hourly rate, and regular hours limit. -/
def overtime_hours (gross_pay : ℚ) (regular_rate : ℚ) (overtime_rate : ℚ) (regular_hours_limit : ℕ) : ℕ :=
  sorry

/-- The number of overtime hours worked is 10 given the specified conditions. -/
theorem overtime_calculation :
  let gross_pay : ℚ := 622
  let regular_rate : ℚ := 11.25
  let overtime_rate : ℚ := 16
  let regular_hours_limit : ℕ := 40
  overtime_hours gross_pay regular_rate overtime_rate regular_hours_limit = 10 := by
  sorry

end overtime_calculation_l2773_277357


namespace cube_sum_theorem_l2773_277367

theorem cube_sum_theorem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) :
  x^3 + y^3 = 65 := by
  sorry

end cube_sum_theorem_l2773_277367


namespace infinite_solutions_l2773_277371

theorem infinite_solutions (k : ℝ) : 
  (∀ x : ℝ, 3 * (5 + k * x) = 15 * x + 15) ↔ k = 5 := by
  sorry

end infinite_solutions_l2773_277371


namespace sphere_surface_area_relation_l2773_277353

theorem sphere_surface_area_relation (R₁ R₂ R₃ S₁ S₂ S₃ : ℝ) 
  (h₁ : R₁ + 2 * R₂ = 3 * R₃)
  (h₂ : S₁ = 4 * Real.pi * R₁^2)
  (h₃ : S₂ = 4 * Real.pi * R₂^2)
  (h₄ : S₃ = 4 * Real.pi * R₃^2) :
  Real.sqrt S₁ + 2 * Real.sqrt S₂ = 3 * Real.sqrt S₃ := by
  sorry

#check sphere_surface_area_relation

end sphere_surface_area_relation_l2773_277353


namespace magnitude_of_2_plus_i_l2773_277345

theorem magnitude_of_2_plus_i : Complex.abs (2 + Complex.I) = Real.sqrt 5 := by sorry

end magnitude_of_2_plus_i_l2773_277345


namespace coordinate_problem_l2773_277327

theorem coordinate_problem (x₁ y₁ x₂ y₂ : ℕ) : 
  (x₁ > 0) → (y₁ > 0) → (x₂ > 0) → (y₂ > 0) →  -- Positive integer coordinates
  (y₁ > x₁) →  -- Angle OA > 45°
  (x₂ > y₂) →  -- Angle OB < 45°
  (x₂ * y₂ = x₁ * y₁ + 67) →  -- Area difference condition
  (x₁ = 1 ∧ y₁ = 5 ∧ x₂ = 9 ∧ y₂ = 8) := by
sorry

end coordinate_problem_l2773_277327


namespace factorial_80_mod_7_l2773_277304

def last_three_nonzero_digits (n : ℕ) : ℕ := sorry

theorem factorial_80_mod_7 : 
  last_three_nonzero_digits (Nat.factorial 80) % 7 = 6 := by sorry

end factorial_80_mod_7_l2773_277304


namespace unique_consecutive_sum_20_l2773_277392

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)
  (h_length : length ≥ 2)

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  (s.length * (2 * s.start + s.length - 1)) / 2

/-- Theorem: There exists exactly one set of consecutive positive integers that sum to 20 -/
theorem unique_consecutive_sum_20 : 
  ∃! s : ConsecutiveSet, sum_consecutive s = 20 :=
sorry

end unique_consecutive_sum_20_l2773_277392


namespace childrens_ticket_cost_l2773_277351

/-- Given information about ticket sales for a show, prove the cost of a children's ticket. -/
theorem childrens_ticket_cost
  (adult_ticket_cost : ℝ)
  (adult_count : ℕ)
  (total_receipts : ℝ)
  (h1 : adult_ticket_cost = 5.50)
  (h2 : adult_count = 152)
  (h3 : total_receipts = 1026)
  (h4 : adult_count = 2 * (adult_count / 2)) :
  ∃ (childrens_ticket_cost : ℝ),
    childrens_ticket_cost = 2.50 ∧
    total_receipts = adult_count * adult_ticket_cost + (adult_count / 2) * childrens_ticket_cost :=
by sorry

end childrens_ticket_cost_l2773_277351


namespace mary_seashells_count_l2773_277342

/-- The number of seashells found by Mary and Jessica together -/
def total_seashells : ℕ := 59

/-- The number of seashells found by Jessica -/
def jessica_seashells : ℕ := 41

/-- The number of seashells found by Mary -/
def mary_seashells : ℕ := total_seashells - jessica_seashells

theorem mary_seashells_count : mary_seashells = 18 := by
  sorry

end mary_seashells_count_l2773_277342


namespace train_length_calculation_l2773_277313

-- Define the given values
def train_speed : ℝ := 60  -- km/hr
def man_speed : ℝ := 6     -- km/hr
def time_to_pass : ℝ := 17.998560115190788  -- seconds

-- Define the theorem
theorem train_length_calculation :
  let relative_speed : ℝ := (train_speed + man_speed) * (5 / 18)  -- Convert to m/s
  let train_length : ℝ := relative_speed * time_to_pass
  train_length = 330 := by sorry

end train_length_calculation_l2773_277313


namespace line_circle_properties_l2773_277336

-- Define the line l and circle C
def line_l (m x y : ℝ) : Prop := (m + 2) * x + (1 - 2 * m) * y + 4 * m - 2 = 0
def circle_C (x y : ℝ) : Prop := x^2 - 2 * x + y^2 = 0

-- Define the intersection points M and N
def intersect_points (m : ℝ) : Prop := ∃ x_M y_M x_N y_N : ℝ,
  line_l m x_M y_M ∧ circle_C x_M y_M ∧
  line_l m x_N y_N ∧ circle_C x_N y_N ∧
  (x_M ≠ x_N ∨ y_M ≠ y_N)

-- Define the slopes of OM and ON
def slope_OM_ON (m : ℝ) : Prop := ∃ k₁ k₂ x_M y_M x_N y_N : ℝ,
  line_l m x_M y_M ∧ circle_C x_M y_M ∧
  line_l m x_N y_N ∧ circle_C x_N y_N ∧
  k₁ = y_M / x_M ∧ k₂ = y_N / x_N

-- Theorem statement
theorem line_circle_properties :
  (∀ m : ℝ, line_l m 0 2) ∧
  (∀ m : ℝ, intersect_points m → -(m + 2) / (1 - 2 * m) < -3/4) ∧
  (∀ m : ℝ, slope_OM_ON m → ∃ k₁ k₂ : ℝ, k₁ + k₂ = 1) :=
sorry

end line_circle_properties_l2773_277336


namespace onions_on_scale_l2773_277393

/-- The number of onions initially on the scale -/
def N : ℕ := sorry

/-- The total weight of onions in grams -/
def W : ℕ := 7680

/-- The average weight of remaining onions in grams -/
def avg_remaining : ℕ := 190

/-- The average weight of removed onions in grams -/
def avg_removed : ℕ := 206

/-- The number of removed onions -/
def removed : ℕ := 5

theorem onions_on_scale :
  W = (N - removed) * avg_remaining + removed * avg_removed ∧ N = 40 := by sorry

end onions_on_scale_l2773_277393


namespace m_plus_n_values_l2773_277376

theorem m_plus_n_values (m n : ℤ) 
  (h1 : |m - n| = n - m) 
  (h2 : |m| = 4) 
  (h3 : |n| = 3) : 
  m + n = -1 ∨ m + n = -7 := by
sorry

end m_plus_n_values_l2773_277376


namespace cymbal_triangle_sync_l2773_277330

theorem cymbal_triangle_sync (cymbal_beats triangle_beats : ℕ) 
  (h1 : cymbal_beats = 7) (h2 : triangle_beats = 2) : 
  Nat.lcm cymbal_beats triangle_beats = 14 := by
  sorry

end cymbal_triangle_sync_l2773_277330


namespace rectangular_sheet_area_l2773_277323

theorem rectangular_sheet_area (area1 area2 : ℝ) : 
  area1 = 4 * area2 →  -- First part is four times larger than the second
  area1 - area2 = 2208 →  -- First part is 2208 cm² larger than the second
  area1 + area2 = 3680 :=  -- Total area of the sheet
by sorry

end rectangular_sheet_area_l2773_277323


namespace solution_equation1_solution_equation2_l2773_277360

-- Define the equations
def equation1 (x : ℝ) : Prop := 2 * (2 * x + 1) - (3 * x - 4) = 2
def equation2 (y : ℝ) : Prop := (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6

-- Theorem statements
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = -4 := by sorry

theorem solution_equation2 : ∃ y : ℝ, equation2 y ∧ y = -1 := by sorry

end solution_equation1_solution_equation2_l2773_277360


namespace arithmetic_not_geometric_l2773_277372

/-- An arithmetic sequence containing 1 and √2 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (k l : ℕ), 
    (∀ n, a (n + 1) = a n + r) ∧ 
    a k = 1 ∧ 
    a l = Real.sqrt 2

/-- Three terms form a geometric sequence -/
def IsGeometric (x y z : ℝ) : Prop :=
  y * y = x * z

theorem arithmetic_not_geometric (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  ¬ ∃ (m n p : ℕ), m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ IsGeometric (a m) (a n) (a p) := by
  sorry

end arithmetic_not_geometric_l2773_277372


namespace constant_sum_perpendicular_distances_l2773_277352

/-- A regular pentagon with circumradius R -/
structure RegularPentagon where
  R : ℝ
  R_pos : R > 0

/-- A point inside a regular pentagon -/
structure InnerPoint (p : RegularPentagon) where
  x : ℝ
  y : ℝ
  inside : x^2 + y^2 < p.R^2

/-- The sum of perpendicular distances from a point to the sides of a regular pentagon -/
noncomputable def sum_perpendicular_distances (p : RegularPentagon) (k : InnerPoint p) : ℝ :=
  sorry

/-- Theorem stating that the sum of perpendicular distances is constant -/
theorem constant_sum_perpendicular_distances (p : RegularPentagon) :
  ∃ (c : ℝ), ∀ (k : InnerPoint p), sum_perpendicular_distances p k = c :=
sorry

end constant_sum_perpendicular_distances_l2773_277352


namespace intersecting_circles_angle_equality_l2773_277321

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the center of a circle
variable (center : Circle → Point)

-- Define the property of a point being on a circle
variable (on_circle : Point → Circle → Prop)

-- Define the property of two circles intersecting
variable (intersect : Circle → Circle → Prop)

-- Define the property of points being collinear
variable (collinear : Point → Point → Point → Prop)

-- Define the angle between three points
variable (angle : Point → Point → Point → ℝ)

-- State the theorem
theorem intersecting_circles_angle_equality
  (C1 C2 : Circle) (O1 O2 P Q U V : Point) :
  center C1 = O1 →
  center C2 = O2 →
  intersect C1 C2 →
  on_circle P C1 →
  on_circle P C2 →
  on_circle Q C1 →
  on_circle Q C2 →
  on_circle U C1 →
  on_circle V C2 →
  collinear U P V →
  angle U Q V = angle O1 Q O2 := by
  sorry

end intersecting_circles_angle_equality_l2773_277321


namespace specific_room_surface_area_l2773_277315

/-- Calculates the interior surface area of a cubic room with a central cubical hole -/
def interior_surface_area (room_edge : ℝ) (hole_edge : ℝ) : ℝ :=
  6 * room_edge^2 - 3 * hole_edge^2

/-- Theorem stating the interior surface area of a specific cubic room with a hole -/
theorem specific_room_surface_area :
  interior_surface_area 10 2 = 588 := by
  sorry

#check specific_room_surface_area

end specific_room_surface_area_l2773_277315
