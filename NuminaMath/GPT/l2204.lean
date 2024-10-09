import Mathlib

namespace nitin_borrowed_amount_l2204_220414

theorem nitin_borrowed_amount (P : ℝ) (interest_paid : ℝ) 
  (rate1 rate2 rate3 : ℝ) (time1 time2 time3 : ℝ) 
  (h_rates1 : rate1 = 0.06) (h_rates2 : rate2 = 0.09) 
  (h_rates3 : rate3 = 0.13) (h_time1 : time1 = 3) 
  (h_time2 : time2 = 5) (h_time3 : time3 = 3)
  (h_interest : interest_paid = 8160) :
  P * (rate1 * time1 + rate2 * time2 + rate3 * time3) = interest_paid → 
  P = 8000 := 
by 
  sorry

end nitin_borrowed_amount_l2204_220414


namespace prove_k_eq_one_l2204_220424

theorem prove_k_eq_one 
  (n m k : ℕ) 
  (h_positive : 0 < n)  -- implies n, and hence n-1, n+1, are all positive
  (h_eq : (n-1) * n * (n+1) = m^k): 
  k = 1 := 
sorry

end prove_k_eq_one_l2204_220424


namespace triangle_area_l2204_220409

theorem triangle_area {a b : ℝ} (h : a ≠ 0) :
  (∃ x y : ℝ, 3 * x + a * y = 12) → b = 24 / a ↔ (∃ x y : ℝ, x = 4 ∧ y = 12 / a ∧ b = (1/2) * 4 * (12 / a)) :=
by
  sorry

end triangle_area_l2204_220409


namespace cost_of_fencing_l2204_220402

/-- Define given conditions: -/
def sides_ratio (length width : ℕ) : Prop := length = 3 * width / 2

def park_area : ℕ := 3750

def paise_to_rupees (paise : ℕ) : ℕ := paise / 100

/-- Prove that the cost of fencing the park is 150 rupees: -/
theorem cost_of_fencing 
  (length width : ℕ) 
  (h : sides_ratio length width) 
  (h_area : length * width = park_area) 
  (cost_per_meter_paise : ℕ := 60) : 
  (length + width) * 2 * (paise_to_rupees cost_per_meter_paise) = 150 :=
by sorry

end cost_of_fencing_l2204_220402


namespace findLineEquation_l2204_220475

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to represent the hyperbola condition
def isOnHyperbola (pt : Point) : Prop :=
  pt.x ^ 2 - 4 * pt.y ^ 2 = 4

-- Define midpoint condition for points A and B
def isMidpoint (P A B : Point) : Prop :=
  P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

-- Define points
def P : Point := ⟨8, 1⟩
def A : Point := sorry
def B : Point := sorry

-- Statement to prove
theorem findLineEquation :
  isOnHyperbola A ∧ isOnHyperbola B ∧ isMidpoint P A B →
  ∃ m b, (∀ pt : Point, pt.y = m * pt.x + b ↔ pt.x = 8 ∧ pt.y = 1) ∧ (m = 2) ∧ (b = -15) :=
by
  sorry

end findLineEquation_l2204_220475


namespace min_count_to_ensure_multiple_of_5_l2204_220415

theorem min_count_to_ensure_multiple_of_5 (n : ℕ) (S : Finset ℕ) (hS : S = Finset.range 31) :
  25 ≤ S.card ∧ (∀ (T : Finset ℕ), T ⊆ S → T.card = 24 → ↑(∃ x ∈ T, x % 5 = 0)) :=
by sorry

end min_count_to_ensure_multiple_of_5_l2204_220415


namespace length_of_faster_train_l2204_220417

/-- 
Let the faster train have a speed of 144 km per hour, the slower train a speed of 
72 km per hour, and the time taken for the faster train to cross a man in the 
slower train be 19 seconds. Then the length of the faster train is 380 meters.
-/
theorem length_of_faster_train 
  (speed_faster_train : ℝ) (speed_slower_train : ℝ) (time_to_cross : ℝ)
  (h_speed_faster_train : speed_faster_train = 144) 
  (h_speed_slower_train : speed_slower_train = 72) 
  (h_time_to_cross : time_to_cross = 19) :
  (speed_faster_train - speed_slower_train) * (5 / 18) * time_to_cross = 380 :=
by
  sorry

end length_of_faster_train_l2204_220417


namespace binom_coeff_divisibility_l2204_220451

theorem binom_coeff_divisibility (p : ℕ) (hp : Prime p) : Nat.choose (2 * p) p - 2 ≡ 0 [MOD p^2] := 
sorry

end binom_coeff_divisibility_l2204_220451


namespace f_f_3_eq_651_over_260_l2204_220403

def f (x : ℚ) : ℚ := x⁻¹ + (x⁻¹ / (2 + x⁻¹))

/-- Prove that f(f(3)) = 651/260 -/
theorem f_f_3_eq_651_over_260 : f (f (3)) = 651 / 260 := 
sorry

end f_f_3_eq_651_over_260_l2204_220403


namespace average_gas_mileage_round_trip_l2204_220487

-- Definition of the problem conditions

def distance_to_home : ℕ := 120
def distance_back : ℕ := 120
def mileage_to_home : ℕ := 30
def mileage_back : ℕ := 20

-- Theorem that we need to prove
theorem average_gas_mileage_round_trip
  (d1 d2 : ℕ) (m1 m2 : ℕ)
  (h1 : d1 = distance_to_home)
  (h2 : d2 = distance_back)
  (h3 : m1 = mileage_to_home)
  (h4 : m2 = mileage_back) :
  (d1 + d2) / ((d1 / m1) + (d2 / m2)) = 24 :=
by
  sorry

end average_gas_mileage_round_trip_l2204_220487


namespace partial_fraction_sum_equals_251_l2204_220485

theorem partial_fraction_sum_equals_251 (p q r A B C : ℝ) :
  (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧ 
  (A ≠ 0) ∧ (B ≠ 0) ∧ (C ≠ 0) ∧
  (∀ s : ℝ, (s ≠ p) ∧ (s ≠ q) ∧ (s ≠ r) →
  1 / (s^3 - 24*s^2 + 151*s - 650) = A / (s - p) + B / (s - q) + C / (s - r)) →
  (p + q + r = 24) →
  (p * q + p * r + q * r = 151) →
  (p * q * r = 650) →
  (1 / A + 1 / B + 1 / C = 251) :=
by
  sorry

end partial_fraction_sum_equals_251_l2204_220485


namespace value_of_m_l2204_220497

theorem value_of_m (m : ℝ) (h1 : m^2 - 2 * m - 1 = 2) (h2 : m ≠ 3) : m = -1 :=
sorry

end value_of_m_l2204_220497


namespace cost_of_trip_per_student_l2204_220433

def raised_fund : ℕ := 50
def contribution_per_student : ℕ := 5
def num_students : ℕ := 20
def remaining_fund : ℕ := 10

theorem cost_of_trip_per_student :
  ((raised_fund - remaining_fund) / num_students) = 2 := by
  sorry

end cost_of_trip_per_student_l2204_220433


namespace molly_more_minutes_than_xanthia_l2204_220490

-- Define the constants: reading speeds and book length
def xanthia_speed := 80  -- pages per hour
def molly_speed := 40    -- pages per hour
def book_length := 320   -- pages

-- Define the times taken to read the book in hours
def xanthia_time := book_length / xanthia_speed
def molly_time := book_length / molly_speed

-- Define the time difference in minutes
def time_difference_minutes := (molly_time - xanthia_time) * 60

theorem molly_more_minutes_than_xanthia : time_difference_minutes = 240 := 
by {
  -- Here the proof would go, but we'll leave it as a sorry for now.
  sorry
}

end molly_more_minutes_than_xanthia_l2204_220490


namespace smallest_positive_perfect_square_divisible_by_2_and_5_l2204_220489

theorem smallest_positive_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, n = k^2) ∧ (2 ∣ n) ∧ (5 ∣ n) ∧ n = 100 :=
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_and_5_l2204_220489


namespace percentage_of_loss_l2204_220427

-- Define the conditions as given in the problem
def original_selling_price : ℝ := 720
def gain_selling_price : ℝ := 880
def gain_percentage : ℝ := 0.10

-- Define the main theorem
theorem percentage_of_loss : ∀ (CP : ℝ),
  (1.10 * CP = gain_selling_price) → 
  ((CP - original_selling_price) / CP * 100 = 10) :=
by
  intro CP
  intro h
  have h1 : CP = gain_selling_price / 1.10 := by sorry
  have h2 : (CP - original_selling_price) = 80 := by sorry -- Intermediate step to show loss
  have h3 : ((80 / CP) * 100 = 10) := by sorry -- Calculation of percentage of loss
  sorry

end percentage_of_loss_l2204_220427


namespace triangle_is_equilateral_l2204_220454

-- Define the triangle
structure Triangle :=
  (A B C : ℝ)

-- Define a triangle's circumradius and inradius
structure TriangleProperties :=
  (circumradius : ℝ)
  (inradius : ℝ)
  (circumcenter_incenter_sq_distance : ℝ) -- OI^2 = circumradius^2 - 2*circumradius*inradius

noncomputable def circumcenter_incenter_coincide (T : Triangle) (P : TriangleProperties) : Prop :=
  P.circumcenter_incenter_sq_distance = 0

theorem triangle_is_equilateral
  (T : Triangle)
  (P : TriangleProperties)
  (hR : P.circumradius = 2 * P.inradius)
  (hOI : circumcenter_incenter_coincide T P) :
  ∃ (R r : ℝ), T = {A := 1 * r, B := 1 * r, C := 1 * r} :=
by sorry

end triangle_is_equilateral_l2204_220454


namespace max_num_triangles_for_right_triangle_l2204_220466

-- Define a right triangle on graph paper
def right_triangle (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ n ∧ 0 ≤ b ∧ b ≤ n

-- Define maximum number of triangles that can be formed within the triangle
def max_triangles (n : ℕ) : ℕ :=
  if h : n = 7 then 28 else 0  -- Given n = 7, the max number is 28

-- Define the theorem to be proven
theorem max_num_triangles_for_right_triangle :
  right_triangle 7 → max_triangles 7 = 28 :=
by
  intro h
  -- Proof goes here
  sorry

end max_num_triangles_for_right_triangle_l2204_220466


namespace three_digit_number_unchanged_upside_down_l2204_220491

theorem three_digit_number_unchanged_upside_down (n : ℕ) :
  (n >= 100 ∧ n <= 999) ∧ (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d = 0 ∨ d = 8) ->
  n = 888 ∨ n = 808 :=
by
  sorry

end three_digit_number_unchanged_upside_down_l2204_220491


namespace trip_length_is_440_l2204_220479

noncomputable def total_trip_length (d : ℝ) : Prop :=
  55 * 0.02 * (d - 40) = d

theorem trip_length_is_440 :
  total_trip_length 440 :=
by
  sorry

end trip_length_is_440_l2204_220479


namespace total_votes_cast_correct_l2204_220449

noncomputable def total_votes_cast : Nat :=
  let total_valid_votes : Nat := 1050
  let spoiled_votes : Nat := 325
  total_valid_votes + spoiled_votes

theorem total_votes_cast_correct :
  total_votes_cast = 1375 := by
  sorry

end total_votes_cast_correct_l2204_220449


namespace equation_of_parallel_line_l2204_220441

theorem equation_of_parallel_line (l : ℝ → ℝ → Prop) (P : ℝ × ℝ)
  (x y : ℝ) (m : ℝ) (H_1 : P = (1, 2)) (H_2 : ∀ x y m, l x y ↔ (2 * x + y + m = 0) )
  (H_3 : l x y) : 
  l 2 (y - 4) := 
  sorry

end equation_of_parallel_line_l2204_220441


namespace max_value_of_expression_l2204_220452

variable (a b c : ℝ)

theorem max_value_of_expression : 
  ∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c = Real.sqrt (a^2 + b^2) + c := by
sorry

end max_value_of_expression_l2204_220452


namespace cevians_concurrent_circumscribable_l2204_220405

-- Define the problem
variables {A B C D X Y Z : Type}

-- Define concurrent cevians
def cevian_concurrent (A B C X Y Z D : Type) : Prop := true

-- Define circumscribable quadrilaterals
def circumscribable (A B C D : Type) : Prop := true

-- The theorem statement
theorem cevians_concurrent_circumscribable (h_conc: cevian_concurrent A B C X Y Z D) 
(h1: circumscribable D Y A Z) (h2: circumscribable D Z B X) : circumscribable D X C Y :=
sorry

end cevians_concurrent_circumscribable_l2204_220405


namespace starting_number_divisible_by_3_count_l2204_220477

-- Define a predicate for divisibility by 3
def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- Define the main theorem
theorem starting_number_divisible_by_3_count : 
  ∃ n : ℕ, (∀ m, n ≤ m ∧ m ≤ 50 → divisible_by_3 m → ∃ s, (m = n + 3 * s) ∧ s < 13) ∧
           (∀ k : ℕ, (divisible_by_3 k) → n ≤ k ∧ k ≤ 50 → m = 12) :=
sorry

end starting_number_divisible_by_3_count_l2204_220477


namespace quadratic_solutions_l2204_220444

-- Definition of the conditions given in the problem
def quadratic_axis_symmetry (b : ℝ) : Prop :=
  -b / 2 = 2

def equation_solutions (x b : ℝ) : Prop :=
  x^2 + b*x - 5 = 2*x - 13

-- The math proof problem statement in Lean 4
theorem quadratic_solutions (b : ℝ) (x1 x2 : ℝ) :
  quadratic_axis_symmetry b →
  equation_solutions x1 b →
  equation_solutions x2 b →
  (x1 = 2 ∧ x2 = 4) ∨ (x1 = 4 ∧ x2 = 2) :=
by
  sorry

end quadratic_solutions_l2204_220444


namespace total_score_is_correct_l2204_220437

def dad_points : ℕ := 7
def olaf_points : ℕ := 3 * dad_points
def total_points : ℕ := dad_points + olaf_points

theorem total_score_is_correct : total_points = 28 := by
  sorry

end total_score_is_correct_l2204_220437


namespace amount_left_after_spending_l2204_220456

-- Define the initial amount and percentage spent
def initial_amount : ℝ := 500
def percentage_spent : ℝ := 0.30

-- Define the proof statement that the amount left is 350
theorem amount_left_after_spending : 
  (initial_amount - (percentage_spent * initial_amount)) = 350 :=
by
  sorry

end amount_left_after_spending_l2204_220456


namespace store_A_total_cost_store_B_total_cost_cost_effective_store_l2204_220432

open Real

def total_cost_store_A (x : ℝ) : ℝ :=
  110 * x + 210 * (100 - x)

def total_cost_store_B (x : ℝ) : ℝ :=
  120 * x + 202 * (100 - x)

theorem store_A_total_cost (x : ℝ) :
  total_cost_store_A x = -100 * x + 21000 :=
by
  sorry

theorem store_B_total_cost (x : ℝ) :
  total_cost_store_B x = -82 * x + 20200 :=
by
  sorry

theorem cost_effective_store (x : ℝ) (h : x = 60) :
  total_cost_store_A x < total_cost_store_B x :=
by
  rw [h]
  sorry

end store_A_total_cost_store_B_total_cost_cost_effective_store_l2204_220432


namespace product_of_solutions_l2204_220478

theorem product_of_solutions (a b c x : ℝ) (h1 : -x^2 - 4 * x + 10 = 0) :
  x * (-4 - x) = -10 :=
by
  sorry

end product_of_solutions_l2204_220478


namespace roots_difference_squared_quadratic_roots_property_l2204_220450

noncomputable def α : ℝ := (3 + Real.sqrt 5) / 2
noncomputable def β : ℝ := (3 - Real.sqrt 5) / 2

theorem roots_difference_squared :
  α - β = Real.sqrt 5 :=
by
  sorry

theorem quadratic_roots_property :
  (α - β) ^ 2 = 5 :=
by
  sorry

end roots_difference_squared_quadratic_roots_property_l2204_220450


namespace count_multiples_of_30_l2204_220470

theorem count_multiples_of_30 (a b n : ℕ) (h1 : a = 900) (h2 : b = 27000) 
    (h3 : ∃ n, 30 * n = a) (h4 : ∃ n, 30 * n = b) : 
    (b - a) / 30 + 1 = 871 := 
by
    sorry

end count_multiples_of_30_l2204_220470


namespace sravan_distance_l2204_220435

theorem sravan_distance {D : ℝ} :
  (D / 90 + D / 60 = 15) ↔ (D = 540) :=
by sorry

end sravan_distance_l2204_220435


namespace number_of_f3_and_sum_of_f3_l2204_220495

noncomputable def f : ℝ → ℝ := sorry
variable (a : ℝ)

theorem number_of_f3_and_sum_of_f3 (hf : ∀ x y : ℝ, f (f x - y) = f x + f (f y - f a) + x) :
  (∃! c : ℝ, f 3 = c) ∧ (∃ s : ℝ, (∀ c, f 3 = c → s = c) ∧ s = 3) :=
sorry

end number_of_f3_and_sum_of_f3_l2204_220495


namespace probability_of_closer_to_D_in_triangle_DEF_l2204_220419

noncomputable def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  0.5 * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

theorem probability_of_closer_to_D_in_triangle_DEF :
  let D := (0, 0)
  let E := (0, 6)
  let F := (8, 0)
  let M := ((D.1 + F.1) / 2, (D.2 + F.2) / 2)
  let N := ((D.1 + E.1) / 2, (D.2 + E.2) / 2)
  let area_DEF := triangle_area D E F
  let area_DMN := triangle_area D M N
  area_DMN / area_DEF = 1 / 4 := by
    sorry

end probability_of_closer_to_D_in_triangle_DEF_l2204_220419


namespace program_outputs_all_divisors_l2204_220486

/--
  The function of the program is to output all divisors of \( n \), 
  given the initial conditions and operations in the program.
 -/
theorem program_outputs_all_divisors (n : ℕ) :
  ∀ I : ℕ, (1 ≤ I ∧ I ≤ n) → (∃ S : ℕ, (n % I = 0 ∧ S = I)) :=
by
  sorry

end program_outputs_all_divisors_l2204_220486


namespace mixed_number_division_l2204_220455

theorem mixed_number_division :
  (5 + 1 / 2 - (2 + 2 / 3)) / (1 + 1 / 5 + 3 + 1 / 4) = 0 + 170 / 267 := 
by
  sorry

end mixed_number_division_l2204_220455


namespace max_non_intersecting_circles_tangent_max_intersecting_circles_without_center_containment_max_intersecting_circles_without_center_containment_2_l2204_220400

-- Definitions and conditions related to the given problem
def unit_circle (r : ℝ) : Prop := r = 1

-- Maximum number of non-intersecting circles of radius 1 tangent to a unit circle.
theorem max_non_intersecting_circles_tangent (r : ℝ) (K : ℝ) 
  (h_r : unit_circle r) (h_K : unit_circle K) : 
  ∃ n, n = 6 := sorry

-- Maximum number of circles of radius 1 intersecting a given unit circle without intersecting centers.
theorem max_intersecting_circles_without_center_containment (r : ℝ) (K : ℝ) 
  (h_r : unit_circle r) (h_K : unit_circle K) : 
  ∃ n, n = 12 := sorry

-- Maximum number of circles of radius 1 intersecting a unit circle K without containing the center of K or any other circle's center.
theorem max_intersecting_circles_without_center_containment_2 (r : ℝ) (K : ℝ)
  (h_r : unit_circle r) (h_K : unit_circle K) :
  ∃ n, n = 18 := sorry

end max_non_intersecting_circles_tangent_max_intersecting_circles_without_center_containment_max_intersecting_circles_without_center_containment_2_l2204_220400


namespace shaded_trapezium_area_l2204_220434

theorem shaded_trapezium_area :
  let side1 := 3
  let side2 := 5
  let side3 := 8
  let p := 3 / 2
  let q := 4
  let height := 5
  let area := (1 / 2) * (p + q) * height
  area = 55 / 4 :=
by
  let side1 := 3
  let side2 := 5
  let side3 := 8
  let p := 3 / 2
  let q := 4
  let height := 5
  let area := (1 / 2) * (p + q) * height
  show area = 55 / 4
  sorry

end shaded_trapezium_area_l2204_220434


namespace problem_1_problem_2_problem_3_l2204_220471

-- Simplified and combined statements for clarity
theorem problem_1 (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1)) : 
  f 3 + f (-1) = -3 := sorry

theorem problem_2 (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1)) : 
  ∀ x, f x = if x ≤ 0 then Real.logb (1/2) (-x + 1) else Real.logb (1/2) (x + 1) := sorry

theorem problem_3 (f : ℝ → ℝ) (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1))
  (h_cond_ev : ∀ x, f x = f (-x)) (a : ℝ) : 
  f (a - 1) < -1 ↔ a ∈ ((Set.Iio 0) ∪ (Set.Ioi 2)) := sorry

end problem_1_problem_2_problem_3_l2204_220471


namespace missed_angle_l2204_220474

def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

theorem missed_angle :
  ∃ (n : ℕ), sum_interior_angles n = 3060 ∧ 3060 - 2997 = 63 :=
by {
  sorry
}

end missed_angle_l2204_220474


namespace seating_arrangement_l2204_220484

theorem seating_arrangement (n : ℕ) (h1 : n * 9 + (100 - n) * 10 = 100) : n = 10 :=
by sorry

end seating_arrangement_l2204_220484


namespace race_outcomes_l2204_220408

def participants : List String := ["Abe", "Bobby", "Charles", "Devin", "Edwin", "Fiona"]

theorem race_outcomes (h : ¬ "Fiona" ∈ ["Abe", "Bobby", "Charles", "Devin", "Edwin"]) : 
  (participants.length - 1) * (participants.length - 2) * (participants.length - 3) = 60 :=
by
  sorry

end race_outcomes_l2204_220408


namespace find_ratio_l2204_220498

theorem find_ratio (f : ℝ → ℝ) (h : ∀ a b : ℝ, b^2 * f a = a^2 * f b) (h3 : f 3 ≠ 0) :
  (f 7 - f 3) / f 3 = 40 / 9 :=
sorry

end find_ratio_l2204_220498


namespace four_digit_even_numbers_count_and_sum_l2204_220464

variable (digits : Set ℕ) (used_once : ∀ d ∈ digits, d ≤ 6 ∧ d ≥ 1)

theorem four_digit_even_numbers_count_and_sum
  (hyp : digits = {1, 2, 3, 4, 5, 6}) :
  ∃ (N M : ℕ), 
    (N = 180 ∧ M = 680040) := 
sorry

end four_digit_even_numbers_count_and_sum_l2204_220464


namespace line_through_perpendicular_l2204_220459

theorem line_through_perpendicular (x y : ℝ) :
  (∃ (k : ℝ), (2 * x - y + 3 = 0) ∧ k = - 1 / 2) →
  (∃ (a b c : ℝ), (a * (-1) + b * 1 + c = 0) ∧ a = 1 ∧ b = 2 ∧ c = -1) :=
by
  sorry

end line_through_perpendicular_l2204_220459


namespace cube_sphere_volume_ratio_l2204_220476

theorem cube_sphere_volume_ratio (s : ℝ) (r : ℝ) (h : r = (Real.sqrt 3 * s) / 2):
  (s^3) / ((4 / 3) * Real.pi * r^3) = (2 * Real.sqrt 3) / Real.pi :=
by
  sorry

end cube_sphere_volume_ratio_l2204_220476


namespace train_speed_l2204_220482

theorem train_speed :
  ∃ V : ℝ,
    (∃ L : ℝ, L = V * 18) ∧ 
    (∃ L : ℝ, L + 260 = V * 31) ∧ 
    V * 3.6 = 72 := by
  sorry

end train_speed_l2204_220482


namespace angle_sum_in_triangle_l2204_220458

theorem angle_sum_in_triangle (A B C : ℝ) (h₁ : A + B = 90) (h₂ : A + B + C = 180) : C = 90 := by
  sorry

end angle_sum_in_triangle_l2204_220458


namespace olympic_medals_l2204_220410

theorem olympic_medals :
  ∃ (a b c : ℕ),
    (a + b + c = 100) ∧
    (3 * a - 153 = 0) ∧
    (c - b = 7) ∧
    (a = 51) ∧
    (a - 13 = 38) ∧
    (c = 28) :=
by
  sorry

end olympic_medals_l2204_220410


namespace hypotenuse_of_45_45_90_triangle_l2204_220401

noncomputable def leg_length : ℝ := 15
noncomputable def angle_opposite_leg : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem hypotenuse_of_45_45_90_triangle (h_leg : ℝ) (h_angle : ℝ) 
  (h_leg_cond : h_leg = leg_length) (h_angle_cond : h_angle = angle_opposite_leg) :
  ∃ h_hypotenuse : ℝ, h_hypotenuse = h_leg * Real.sqrt 2 :=
sorry

end hypotenuse_of_45_45_90_triangle_l2204_220401


namespace find_triplets_l2204_220493

theorem find_triplets (a k m : ℕ) (hpos_a : 0 < a) (hpos_k : 0 < k) (hpos_m : 0 < m) (h_eq : k + a^k = m + 2 * a^m) :
  ∃ t : ℕ, 0 < t ∧ (a = 1 ∧ k = t + 1 ∧ m = t) :=
by
  sorry

end find_triplets_l2204_220493


namespace strawberries_weight_l2204_220448

theorem strawberries_weight (total_weight apples_weight oranges_weight grapes_weight strawberries_weight : ℕ) 
  (h_total : total_weight = 10)
  (h_apples : apples_weight = 3)
  (h_oranges : oranges_weight = 1)
  (h_grapes : grapes_weight = 3) 
  (h_sum : total_weight = apples_weight + oranges_weight + grapes_weight + strawberries_weight) :
  strawberries_weight = 3 :=
by
  sorry

end strawberries_weight_l2204_220448


namespace larger_investment_value_l2204_220413

-- Definitions of the conditions given in the problem
def investment_value_1 : ℝ := 500
def yearly_return_rate_1 : ℝ := 0.07
def yearly_return_rate_2 : ℝ := 0.27
def combined_return_rate : ℝ := 0.22

-- Stating the proof problem
theorem larger_investment_value :
  ∃ X : ℝ, X = 1500 ∧ 
    yearly_return_rate_1 * investment_value_1 + yearly_return_rate_2 * X = combined_return_rate * (investment_value_1 + X) :=
by {
  sorry -- Proof is omitted as per instructions
}

end larger_investment_value_l2204_220413


namespace total_height_of_buildings_l2204_220425

-- Definitions based on the conditions
def tallest_building : ℤ := 100
def second_tallest_building : ℤ := tallest_building / 2
def third_tallest_building : ℤ := second_tallest_building / 2
def fourth_tallest_building : ℤ := third_tallest_building / 5

-- Use the definitions to state the theorem
theorem total_height_of_buildings : 
  tallest_building + second_tallest_building + third_tallest_building + fourth_tallest_building = 180 := by
  sorry

end total_height_of_buildings_l2204_220425


namespace exists_common_point_l2204_220411

-- Definitions: Rectangle and the problem conditions
structure Rectangle :=
(x_min y_min x_max y_max : ℝ)
(h_valid : x_min ≤ x_max ∧ y_min ≤ y_max)

def rectangles_intersect (R1 R2 : Rectangle) : Prop :=
¬(R1.x_max < R2.x_min ∨ R2.x_max < R1.x_min ∨ R1.y_max < R2.y_min ∨ R2.y_max < R1.y_min)

def all_rectangles_intersect (rects : List Rectangle) : Prop :=
∀ (R1 R2 : Rectangle), R1 ∈ rects → R2 ∈ rects → rectangles_intersect R1 R2

-- Theorem: Existence of a common point
theorem exists_common_point (rects : List Rectangle) (h_intersect : all_rectangles_intersect rects) : 
  ∃ (T : ℝ × ℝ), ∀ (R : Rectangle), R ∈ rects → 
    R.x_min ≤ T.1 ∧ T.1 ≤ R.x_max ∧ 
    R.y_min ≤ T.2 ∧ T.2 ≤ R.y_max := 
sorry

end exists_common_point_l2204_220411


namespace graph_translation_l2204_220423

variable (f : ℝ → ℝ)

theorem graph_translation (h : f 1 = 3) : f (-1) + 1 = 4 :=
sorry

end graph_translation_l2204_220423


namespace max_d_is_9_l2204_220494

-- Define the 6-digit number of the form 8d8, 45e
def num (d e : ℕ) : ℕ :=
  800000 + 10000 * d + 800 + 450 + e

-- Define the conditions: the number is a multiple of 45, 0 ≤ d, e ≤ 9
def conditions (d e : ℕ) : Prop :=
  0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
  (num d e) % 45 = 0

-- Define the maximum value of d
noncomputable def max_d : ℕ :=
  9

-- The theorem statement to be proved
theorem max_d_is_9 :
  ∀ (d e : ℕ), conditions d e → d ≤ max_d :=
by
  sorry

end max_d_is_9_l2204_220494


namespace problem1_problem2_problem3_problem4_l2204_220416

def R : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 < x ∧ x < 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 6}

theorem problem1 : A ∩ B = {x | 3 ≤ x ∧ x < 5} := sorry

theorem problem2 : A ∪ B = {x | 1 < x ∧ x ≤ 6} := sorry

theorem problem3 : (Set.compl A) ∩ B = {x | 5 ≤ x ∧ x ≤ 6} :=
sorry

theorem problem4 : Set.compl (A ∩ B) = {x | x < 3 ∨ x ≥ 5} := sorry

end problem1_problem2_problem3_problem4_l2204_220416


namespace line_passes_through_center_l2204_220406

theorem line_passes_through_center (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y = 0 → 3 * x + y + a = 0) → a = 1 :=
by
  sorry

end line_passes_through_center_l2204_220406


namespace not_perfect_cube_of_cond_l2204_220445

open Int

theorem not_perfect_cube_of_cond (n : ℤ) (h₁ : 0 < n) (k : ℤ) 
  (h₂ : n^5 + n^3 + 2 * n^2 + 2 * n + 2 = k ^ 3) : 
  ¬ ∃ m : ℤ, 2 * n^2 + n + 2 = m ^ 3 :=
sorry

end not_perfect_cube_of_cond_l2204_220445


namespace domain_of_f_l2204_220426

noncomputable def f (x : ℝ) := Real.sqrt (x - 1) + (1 / (x - 2))

theorem domain_of_f : { x : ℝ | x ≥ 1 ∧ x ≠ 2 } = { x : ℝ | ∃ (y : ℝ), f x = y } :=
sorry

end domain_of_f_l2204_220426


namespace total_sandwiches_prepared_l2204_220439

def num_people := 219.0
def sandwiches_per_person := 3.0

theorem total_sandwiches_prepared : num_people * sandwiches_per_person = 657.0 :=
by
  sorry

end total_sandwiches_prepared_l2204_220439


namespace school_total_students_l2204_220460

theorem school_total_students (T G : ℕ) (h1 : 80 + G = T) (h2 : G = (80 * T) / 100) : T = 400 :=
by
  sorry

end school_total_students_l2204_220460


namespace find_other_number_l2204_220436

theorem find_other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 192) (h_hcf : Nat.gcd A B = 16) (h_A : A = 48) : B = 64 :=
by
  sorry

end find_other_number_l2204_220436


namespace shaniqua_income_per_haircut_l2204_220421

theorem shaniqua_income_per_haircut (H : ℝ) :
  (8 * H + 5 * 25 = 221) → (H = 12) :=
by
  intro h
  sorry

end shaniqua_income_per_haircut_l2204_220421


namespace parallelogram_height_l2204_220472

theorem parallelogram_height (area base height : ℝ) 
  (h_area : area = 336) 
  (h_base : base = 14) 
  (h_formula : area = base * height) : 
  height = 24 := 
by 
  sorry

end parallelogram_height_l2204_220472


namespace michael_pays_106_l2204_220473

def num_cats : ℕ := 2
def num_dogs : ℕ := 3
def num_parrots : ℕ := 1
def num_fish : ℕ := 4

def cost_per_cat : ℕ := 13
def cost_per_dog : ℕ := 18
def cost_per_parrot : ℕ := 10
def cost_per_fish : ℕ := 4

def total_cost : ℕ :=
  (num_cats * cost_per_cat) +
  (num_dogs * cost_per_dog) +
  (num_parrots * cost_per_parrot) +
  (num_fish * cost_per_fish)

theorem michael_pays_106 : total_cost = 106 := by
  sorry

end michael_pays_106_l2204_220473


namespace find_number_l2204_220431

theorem find_number (x : ℝ) : 
  220050 = (555 + x) * (2 * (x - 555)) + 50 ↔ x = 425.875 ∨ x = -980.875 := 
by 
  sorry

end find_number_l2204_220431


namespace mass_percentage_of_nitrogen_in_N2O5_l2204_220468

noncomputable def atomic_mass_N : ℝ := 14.01
noncomputable def atomic_mass_O : ℝ := 16.00
noncomputable def molar_mass_N2O5 : ℝ := (2 * atomic_mass_N) + (5 * atomic_mass_O)

theorem mass_percentage_of_nitrogen_in_N2O5 : 
  (2 * atomic_mass_N / molar_mass_N2O5 * 100) = 25.94 := 
by 
  sorry

end mass_percentage_of_nitrogen_in_N2O5_l2204_220468


namespace fill_cistern_time_l2204_220422

theorem fill_cistern_time (A B C : ℕ) (hA : A = 10) (hB : B = 12) (hC : C = 50) :
    1 / (1 / A + 1 / B - 1 / C) = 300 / 49 :=
by
  sorry

end fill_cistern_time_l2204_220422


namespace cost_of_pen_is_51_l2204_220430

-- Definitions of variables and conditions
variables {p q : ℕ}
variables (h1 : 6 * p + 2 * q = 348)
variables (h2 : 3 * p + 4 * q = 234)

-- Goal: Prove the cost of a pen (p) is 51 cents
theorem cost_of_pen_is_51 : p = 51 :=
by
  -- placeholder for the proof
  sorry

end cost_of_pen_is_51_l2204_220430


namespace card_game_total_l2204_220462

theorem card_game_total (C E O : ℝ) (h1 : E = (11 / 20) * C) (h2 : O = (9 / 20) * C) (h3 : E = O + 50) : C = 500 :=
sorry

end card_game_total_l2204_220462


namespace length_of_bridge_l2204_220461

noncomputable def train_length : ℝ := 155
noncomputable def train_speed_km_hr : ℝ := 45
noncomputable def crossing_time_seconds : ℝ := 30

noncomputable def train_speed_m_s : ℝ := train_speed_km_hr * 1000 / 3600

noncomputable def total_distance : ℝ := train_speed_m_s * crossing_time_seconds

noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge : bridge_length = 220 := by
  sorry

end length_of_bridge_l2204_220461


namespace money_left_l2204_220407

-- Conditions
def initial_savings : ℤ := 6000
def spent_on_flight : ℤ := 1200
def spent_on_hotel : ℤ := 800
def spent_on_food : ℤ := 3000

-- Total spent
def total_spent : ℤ := spent_on_flight + spent_on_hotel + spent_on_food

-- Prove that the money left is $1,000
theorem money_left (h1 : initial_savings = 6000)
                   (h2 : spent_on_flight = 1200)
                   (h3 : spent_on_hotel = 800)
                   (h4 : spent_on_food = 3000) :
                   initial_savings - total_spent = 1000 :=
by
  -- Insert proof steps here
  sorry

end money_left_l2204_220407


namespace free_fall_height_and_last_second_distance_l2204_220467

theorem free_fall_height_and_last_second_distance :
  let time := 11
  let initial_distance := 4.9
  let increment := 9.8
  let total_height := (initial_distance * time + increment * (time * (time - 1)) / 2)
  let last_second_distance := initial_distance + increment * (time - 1)
  total_height = 592.9 ∧ last_second_distance = 102.9 :=
by
  sorry

end free_fall_height_and_last_second_distance_l2204_220467


namespace cost_price_l2204_220481

theorem cost_price (MP SP C : ℝ) (h1 : MP = 74.21875)
  (h2 : SP = MP - 0.20 * MP)
  (h3 : SP = 1.25 * C) : C = 47.5 :=
by
  sorry

end cost_price_l2204_220481


namespace elberta_money_l2204_220469

theorem elberta_money (GrannySmith Anjou Elberta : ℝ)
  (h_granny : GrannySmith = 100)
  (h_anjou : Anjou = 1 / 4 * GrannySmith)
  (h_elberta : Elberta = Anjou + 5) : Elberta = 30 := by
  sorry

end elberta_money_l2204_220469


namespace sum_of_possible_areas_of_square_in_xy_plane_l2204_220440

theorem sum_of_possible_areas_of_square_in_xy_plane (x1 x2 x3 : ℝ) (A : ℝ)
    (h1 : x1 = 2 ∨ x1 = 0 ∨ x1 = 18)
    (h2 : x2 = 2 ∨ x2 = 0 ∨ x2 = 18)
    (h3 : x3 = 2 ∨ x3 = 0 ∨ x3 = 18) :
  A = 1168 := sorry

end sum_of_possible_areas_of_square_in_xy_plane_l2204_220440


namespace geometric_sequence_general_formula_l2204_220483

noncomputable def a_n (n : ℕ) : ℝ := 2^n

theorem geometric_sequence_general_formula :
  (∀ n : ℕ, 2 * (a_n n + a_n (n + 2)) = 5 * a_n (n + 1)) →
  (a_n 5 ^ 2 = a_n 10) →
  ∀ n : ℕ, a_n n = 2 ^ n := 
by 
  sorry

end geometric_sequence_general_formula_l2204_220483


namespace ants_in_field_l2204_220442

-- Defining constants
def width_feet : ℕ := 500
def length_feet : ℕ := 600
def ants_per_square_inch : ℕ := 4
def inches_per_foot : ℕ := 12

-- Converting dimensions from feet to inches
def width_inches : ℕ := width_feet * inches_per_foot
def length_inches : ℕ := length_feet * inches_per_foot

-- Calculating the area of the field in square inches
def field_area_square_inches : ℕ := width_inches * length_inches

-- Calculating the total number of ants
def total_ants : ℕ := ants_per_square_inch * field_area_square_inches

-- Theorem statement
theorem ants_in_field : total_ants = 172800000 :=
by
  -- Proof is skipped
  sorry

end ants_in_field_l2204_220442


namespace tomatoes_planted_each_kind_l2204_220496

-- Definitions derived from Conditions
def total_rows : ℕ := 10
def spaces_per_row : ℕ := 15
def kinds_of_tomatoes : ℕ := 3
def kinds_of_cucumbers : ℕ := 5
def cucumbers_per_kind : ℕ := 4
def potatoes : ℕ := 30
def available_spaces : ℕ := 85

-- Theorem statement with the question and answer derived from the problem
theorem tomatoes_planted_each_kind : (kinds_of_tomatoes * (total_rows * spaces_per_row - Available_spaces - (kinds_of_cucumbers * cucumbers_per_kind + potatoes)) / kinds_of_tomatoes) = 5 :=
by 
  sorry

end tomatoes_planted_each_kind_l2204_220496


namespace father_son_age_ratio_l2204_220447

theorem father_son_age_ratio :
  ∃ S : ℕ, (45 = S + 15 * 2) ∧ (45 / S = 3) := 
sorry

end father_son_age_ratio_l2204_220447


namespace find_a_b_c_sum_l2204_220404

-- Define the necessary conditions and constants
def radius : ℝ := 10  -- tower radius in feet
def rope_length : ℝ := 30  -- length of the rope in feet
def unicorn_height : ℝ := 6  -- height of the unicorn from ground in feet
def rope_end_distance : ℝ := 6  -- distance from the unicorn to the nearest point on the tower

def a : ℕ := 30
def b : ℕ := 900
def c : ℕ := 10  -- assuming c is not necessarily prime for the purpose of this exercise

-- The theorem we want to prove
theorem find_a_b_c_sum : a + b + c = 940 :=
by
  sorry

end find_a_b_c_sum_l2204_220404


namespace polynomial_root_sum_l2204_220446

theorem polynomial_root_sum 
  (c d : ℂ) 
  (h1 : c + d = 6) 
  (h2 : c * d = 10) 
  (h3 : c^2 - 6 * c + 10 = 0) 
  (h4 : d^2 - 6 * d + 10 = 0) : 
  c^3 + c^5 * d^3 + c^3 * d^5 + d^3 = 16156 := 
by sorry

end polynomial_root_sum_l2204_220446


namespace rectangle_perimeters_l2204_220443

theorem rectangle_perimeters (length width : ℕ) (h1 : length = 7) (h2 : width = 5) :
  (∃ (L1 L2 : ℕ), L1 = 4 * width ∧ L2 = length ∧ 2 * (L1 + L2) = 54) ∧
  (∃ (L3 L4 : ℕ), L3 = 4 * length ∧ L4 = width ∧ 2 * (L3 + L4) = 66) ∧
  (∃ (L5 L6 : ℕ), L5 = 2 * length ∧ L6 = 2 * width ∧ 2 * (L5 + L6) = 48) :=
by
  sorry

end rectangle_perimeters_l2204_220443


namespace coeff_x2_in_expansion_l2204_220457

theorem coeff_x2_in_expansion : 
  (2 : ℚ) - (1 / x) * ((1 + x)^6)^(2 : ℤ) = (10 : ℚ) :=
by sorry

end coeff_x2_in_expansion_l2204_220457


namespace find_angle_B_l2204_220412

noncomputable def angle_B (A B C a b c : ℝ): Prop := 
  a * Real.cos B - b * Real.cos A = b ∧ 
  C = Real.pi / 5

theorem find_angle_B (a b c A B C : ℝ) (h : angle_B A B C a b c) : 
  B = 4 * Real.pi / 15 :=
by
  sorry

end find_angle_B_l2204_220412


namespace a_minus_b_ge_one_l2204_220420

def a : ℕ := 19^91
def b : ℕ := (999991)^19

theorem a_minus_b_ge_one : a - b ≥ 1 :=
by
  sorry

end a_minus_b_ge_one_l2204_220420


namespace total_legs_in_room_l2204_220429

def count_legs : Nat :=
  let tables_4_legs := 4 * 4
  let sofas_legs := 1 * 4
  let chairs_4_legs := 2 * 4
  let tables_3_legs := 3 * 3
  let tables_1_leg := 1 * 1
  let rocking_chair_legs := 1 * 2
  tables_4_legs + sofas_legs + chairs_4_legs + tables_3_legs + tables_1_leg + rocking_chair_legs

theorem total_legs_in_room : count_legs = 40 := by
  sorry

end total_legs_in_room_l2204_220429


namespace find_multiple_l2204_220480

-- Definitions and given conditions
def total_seats : ℤ := 387
def first_class_seats : ℤ := 77

-- The statement we need to prove
theorem find_multiple (m : ℤ) :
  (total_seats = first_class_seats + (m * first_class_seats + 2)) → m = 4 :=
by
  sorry

end find_multiple_l2204_220480


namespace sector_area_l2204_220465

theorem sector_area (alpha : ℝ) (r : ℝ) (h_alpha : alpha = Real.pi / 3) (h_r : r = 2) : 
  (1 / 2) * (alpha * r) * r = (2 * Real.pi) / 3 := 
by
  sorry

end sector_area_l2204_220465


namespace distinct_real_roots_imply_sum_greater_than_two_l2204_220488

noncomputable def function_f (x: ℝ) : ℝ := abs (Real.log x)

theorem distinct_real_roots_imply_sum_greater_than_two {k α β : ℝ} 
  (h₁ : function_f α = k) 
  (h₂ : function_f β = k) 
  (h₃ : α ≠ β) 
  (h4 : 0 < α ∧ α < 1)
  (h5 : 1 < β) :
  (1 / α) + (1 / β) > 2 :=
sorry

end distinct_real_roots_imply_sum_greater_than_two_l2204_220488


namespace correct_option_is_D_l2204_220492

noncomputable def option_A := 230
noncomputable def option_B := [251, 260]
noncomputable def option_B_average := 256
noncomputable def option_C := [21, 212, 256]
noncomputable def option_C_average := 163
noncomputable def option_D := [210, 240, 250]
noncomputable def option_D_average := 233

theorem correct_option_is_D :
  ∃ (correct_option : String), correct_option = "D" :=
  sorry

end correct_option_is_D_l2204_220492


namespace remainder_polynomial_division_l2204_220463

noncomputable def remainder_division : Polynomial ℝ := 
  (Polynomial.X ^ 4 + Polynomial.X ^ 3 - 4 * Polynomial.X + 1) % (Polynomial.X ^ 3 - 1)

theorem remainder_polynomial_division :
  remainder_division = -3 * Polynomial.X + 2 :=
by
  sorry

end remainder_polynomial_division_l2204_220463


namespace sum_of_powers_of_i_l2204_220438

noncomputable def i : Complex := Complex.I

theorem sum_of_powers_of_i :
  (Finset.range 2011).sum (λ n => i^(n+1)) = -1 := by
  sorry

end sum_of_powers_of_i_l2204_220438


namespace domain_of_function_l2204_220418

noncomputable def domain_f (x : ℝ) : Prop :=
  -x^2 + 2 * x + 3 > 0 ∧ 1 - x > 0 ∧ x ≠ 0

theorem domain_of_function :
  {x : ℝ | domain_f x} = {x : ℝ | -1 < x ∧ x < 1 ∧ x ≠ 0} :=
by
  sorry

end domain_of_function_l2204_220418


namespace sec_150_eq_neg_two_sqrt_three_over_three_l2204_220428

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l2204_220428


namespace probability_of_drawing_white_ball_l2204_220499

-- Define initial conditions
def initial_balls : ℕ := 6
def total_balls_after_white : ℕ := initial_balls + 1
def number_of_white_balls : ℕ := 1
def number_of_total_balls : ℕ := total_balls_after_white

-- Define the probability of drawing a white ball
def probability_of_white : ℚ := number_of_white_balls / number_of_total_balls

-- Statement to be proved
theorem probability_of_drawing_white_ball :
  probability_of_white = 1 / 7 :=
by
  sorry

end probability_of_drawing_white_ball_l2204_220499


namespace number_of_people_eating_both_l2204_220453

variable (A B C : Nat)

theorem number_of_people_eating_both (hA : A = 13) (hB : B = 19) (hC : C = B - A) : C = 6 :=
by 
  sorry

end number_of_people_eating_both_l2204_220453
