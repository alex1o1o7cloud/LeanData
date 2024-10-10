import Mathlib

namespace inverse_of_periodic_function_l898_89869

def PeriodicFunction (f : ℝ → ℝ) :=
  ∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x

def SmallestPositivePeriod (f : ℝ → ℝ) (T : ℝ) :=
  PeriodicFunction f ∧ T > 0 ∧ ∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S

def InverseInInterval (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ g : ℝ → ℝ, ∀ x ∈ Set.Ioo a b, g (f x) = x ∧ f (g x) = x

theorem inverse_of_periodic_function
  (f : ℝ → ℝ) (T : ℝ)
  (h_periodic : SmallestPositivePeriod f T)
  (h_inverse : InverseInInterval f 0 T) :
  ∃ g : ℝ → ℝ, ∀ x ∈ Set.Ioo T (2 * T),
    g (f x) = x ∧ f (g x) = x ∧ g x = (Classical.choose h_inverse) (x - T) + T :=
by sorry

end inverse_of_periodic_function_l898_89869


namespace max_value_of_f_l898_89870

-- Define the function f
def f (x a : ℝ) : ℝ := -x^2 + 4*x + a

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≥ -2) ∧ 
  (∃ x ∈ Set.Icc 0 1, f x a = -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f x a ≤ 1) := by
sorry


end max_value_of_f_l898_89870


namespace complement_intersection_l898_89884

def U : Set ℕ := {x | 0 < x ∧ x < 7}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 4}

theorem complement_intersection :
  (U \ A) ∩ (U \ B) = {6} := by sorry

end complement_intersection_l898_89884


namespace f_properties_l898_89852

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + 1 / (a^x)

theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, f a (-x) = f a x) ∧
  (∀ x y : ℝ, 0 ≤ x → x < y → f a x < f a y) ∧
  (∀ x y : ℝ, x < y → y ≤ 0 → f a x > f a y) ∧
  (Set.Ioo (-2 : ℝ) 0 = {x : ℝ | f a (x - 1) > f a (2*x + 1)}) :=
by sorry

end f_properties_l898_89852


namespace largest_number_problem_l898_89881

theorem largest_number_problem (a b c d e : ℕ) 
  (sum1 : a + b + c + d = 350)
  (sum2 : a + b + c + e = 370)
  (sum3 : a + b + d + e = 390)
  (sum4 : a + c + d + e = 410)
  (sum5 : b + c + d + e = 430) :
  max a (max b (max c (max d e))) = 138 := by
sorry

end largest_number_problem_l898_89881


namespace arithmetic_sequence_properties_l898_89807

-- Define the arithmetic sequence
def arithmetic_sequence (n : ℕ+) : ℚ := 2 * n - 1

-- Define the sum of the first n terms
def S (n : ℕ+) : ℚ := n * (arithmetic_sequence 1 + arithmetic_sequence n) / 2

-- Define b_n
def b (n : ℕ+) : ℚ := 1 / (arithmetic_sequence (n + 1) * arithmetic_sequence (n + 2))

-- Define T_n
def T (n : ℕ+) : ℚ := (Finset.range n).sum (λ i => b ⟨i + 1, Nat.succ_pos i⟩)

-- Theorem statement
theorem arithmetic_sequence_properties :
  (arithmetic_sequence 1 + arithmetic_sequence 13 = 26) ∧
  (S 9 = 81) →
  (∀ n : ℕ+, arithmetic_sequence n = 2 * n - 1) ∧
  (∀ n : ℕ+, T n = n / (3 * (2 * n + 3))) :=
by sorry

end arithmetic_sequence_properties_l898_89807


namespace modular_arithmetic_problem_l898_89863

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), 
    (7 * a) % 72 = 1 ∧ 
    (13 * b) % 72 = 1 ∧ 
    ((3 * a + 9 * b) % 72) % 72 = 6 :=
by sorry

end modular_arithmetic_problem_l898_89863


namespace family_age_relations_l898_89879

structure Family where
  rachel_age : ℕ
  grandfather_age : ℕ
  mother_age : ℕ
  father_age : ℕ
  aunt_age : ℕ

def family_ages : Family where
  rachel_age := 12
  grandfather_age := 7 * 12
  mother_age := (7 * 12) / 2
  father_age := (7 * 12) / 2 + 5
  aunt_age := 7 * 12 - 8

theorem family_age_relations (f : Family) :
  f.rachel_age = 12 ∧
  f.grandfather_age = 7 * f.rachel_age ∧
  f.mother_age = f.grandfather_age / 2 ∧
  f.father_age = f.mother_age + 5 ∧
  f.aunt_age = f.grandfather_age - 8 →
  f = family_ages :=
by sorry

end family_age_relations_l898_89879


namespace rectangle_dimension_change_l898_89813

theorem rectangle_dimension_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let new_length := 1.15 * L
  let new_area := 1.035 * (L * B)
  let new_breadth := new_area / new_length
  (new_breadth / B) = 0.9 := by sorry

end rectangle_dimension_change_l898_89813


namespace compound_molecular_weight_l898_89876

/-- The atomic weight of Copper (Cu) in g/mol -/
def atomic_weight_Cu : ℝ := 63.546

/-- The atomic weight of Carbon (C) in g/mol -/
def atomic_weight_C : ℝ := 12.011

/-- The atomic weight of Oxygen (O) in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- The number of Cu atoms in the compound -/
def num_Cu : ℕ := 1

/-- The number of C atoms in the compound -/
def num_C : ℕ := 1

/-- The number of O atoms in the compound -/
def num_O : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ :=
  (num_Cu : ℝ) * atomic_weight_Cu +
  (num_C : ℝ) * atomic_weight_C +
  (num_O : ℝ) * atomic_weight_O

theorem compound_molecular_weight :
  molecular_weight = 123.554 := by sorry

end compound_molecular_weight_l898_89876


namespace divisors_of_60_and_84_l898_89801

theorem divisors_of_60_and_84 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ d : ℕ, d > 0 ∧ (60 % d = 0 ∧ 84 % d = 0) ↔ d ∈ Finset.range n) :=
by sorry

end divisors_of_60_and_84_l898_89801


namespace right_triangle_hypotenuse_l898_89880

theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : a = 60) (h2 : b = 80) 
  (h3 : c^2 = a^2 + b^2) : c = 100 := by
  sorry

end right_triangle_hypotenuse_l898_89880


namespace least_number_divisibility_l898_89842

theorem least_number_divisibility (x : ℕ) : x = 171011 ↔ 
  (∀ y : ℕ, y < x → ¬(41 ∣ (1076 + y) ∧ 59 ∣ (1076 + y) ∧ 67 ∣ (1076 + y))) ∧
  (41 ∣ (1076 + x) ∧ 59 ∣ (1076 + x) ∧ 67 ∣ (1076 + x)) :=
by sorry

end least_number_divisibility_l898_89842


namespace xyz_product_l898_89872

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (y + z) = 198)
  (eq2 : y * (z + x) = 216)
  (eq3 : z * (x + y) = 234) :
  x * y * z = 1080 := by
sorry

end xyz_product_l898_89872


namespace f_properties_l898_89848

def f (x b c : ℝ) : ℝ := x * abs x + b * x + c

theorem f_properties (b c : ℝ) :
  (∀ x, f x b c = -f (-x) b c → c = 0) ∧
  (∃! x, f x 0 c = 0) ∧
  (∀ x, f (-x) b c + f x b c = 2 * c) :=
by sorry

end f_properties_l898_89848


namespace modified_star_angle_sum_l898_89897

/-- A modified n-pointed star --/
structure ModifiedStar where
  n : ℕ
  is_valid : n ≥ 6

/-- The sum of interior angles of the modified star --/
def interior_angle_sum (star : ModifiedStar) : ℝ :=
  180 * (star.n - 2)

/-- Theorem: The sum of interior angles of a modified n-pointed star is 180(n-2) degrees --/
theorem modified_star_angle_sum (star : ModifiedStar) :
  interior_angle_sum star = 180 * (star.n - 2) := by
  sorry

end modified_star_angle_sum_l898_89897


namespace complex_fraction_simplification_l898_89893

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 2 * i) / (1 + 4 * i) = -6 / 17 - (10 / 17) * i :=
by
  sorry


end complex_fraction_simplification_l898_89893


namespace y_share_per_x_rupee_l898_89837

/-- Given a sum divided among x, y, and z, prove that y gets 9/20 rupees for each rupee x gets. -/
theorem y_share_per_x_rupee (x y z : ℝ) (total : ℝ) (y_share : ℝ) (y_per_x : ℝ) : 
  total = 234 →
  y_share = 54 →
  x + y + z = total →
  y = y_per_x * x →
  z = 0.5 * x →
  y_per_x = 9/20 := by
  sorry

end y_share_per_x_rupee_l898_89837


namespace sophomore_sample_count_l898_89800

/-- Given a school with 1000 students, including 320 sophomores,
    prove that a random sample of 200 students will contain 64 sophomores. -/
theorem sophomore_sample_count (total_students : ℕ) (sophomores : ℕ) (sample_size : ℕ) :
  total_students = 1000 →
  sophomores = 320 →
  sample_size = 200 →
  (sophomores : ℚ) / total_students * sample_size = 64 := by
  sorry

end sophomore_sample_count_l898_89800


namespace square_difference_plus_constant_problem_solution_l898_89840

theorem square_difference_plus_constant (a b c : ℤ) :
  a ^ 2 - b ^ 2 + c = (a + b) * (a - b) + c := by sorry

theorem problem_solution :
  632 ^ 2 - 568 ^ 2 + 100 = 76900 := by sorry

end square_difference_plus_constant_problem_solution_l898_89840


namespace complex_magnitude_3_minus_10i_l898_89846

theorem complex_magnitude_3_minus_10i :
  Complex.abs (3 - 10 * Complex.I) = Real.sqrt 109 := by
  sorry

end complex_magnitude_3_minus_10i_l898_89846


namespace geometric_sequence_properties_l898_89867

/-- Geometric sequence with first term 3 and second sum 9 -/
def geometric_sequence (n : ℕ) : ℝ :=
  3 * 2^(n - 1)

/-- Sum of the first n terms of the geometric sequence -/
def geometric_sum (n : ℕ) : ℝ :=
  3 * (2^n - 1)

theorem geometric_sequence_properties :
  (geometric_sequence 1 = 3) ∧
  (geometric_sum 2 = 9) ∧
  (∀ n : ℕ, n ≥ 1 → geometric_sequence n = 3 * 2^(n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → geometric_sum n = 3 * (2^n - 1)) :=
by sorry

end geometric_sequence_properties_l898_89867


namespace expression_evaluation_l898_89887

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end expression_evaluation_l898_89887


namespace multiples_of_15_between_25_and_205_l898_89866

theorem multiples_of_15_between_25_and_205 : 
  (Finset.filter (fun n => n % 15 = 0) (Finset.range 205 \ Finset.range 26)).card = 12 := by
  sorry

end multiples_of_15_between_25_and_205_l898_89866


namespace nested_expression_equals_one_l898_89825

def nested_expression : ℤ :=
  (3 * (3 * (3 * (3 * (3 - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1)

theorem nested_expression_equals_one : nested_expression = 1 := by
  sorry

end nested_expression_equals_one_l898_89825


namespace pet_store_birds_l898_89844

theorem pet_store_birds (total_animals : ℕ) (talking_birds : ℕ) (non_talking_birds : ℕ) (dogs : ℕ) :
  total_animals = 180 →
  talking_birds = 64 →
  non_talking_birds = 13 →
  dogs = 40 →
  talking_birds = 4 * ((total_animals - (talking_birds + non_talking_birds + dogs)) / 4) →
  talking_birds + non_talking_birds = 124 :=
by sorry

end pet_store_birds_l898_89844


namespace family_park_cost_l898_89832

/-- Calculates the total cost for a family to visit a park and one attraction -/
def total_cost (num_children num_adults entrance_fee child_attraction_fee adult_attraction_fee : ℕ) : ℕ :=
  (num_children + num_adults) * entrance_fee + 
  num_children * child_attraction_fee + 
  num_adults * adult_attraction_fee

/-- Proves that the total cost for the given family is $55 -/
theorem family_park_cost : 
  total_cost 4 3 5 2 4 = 55 := by
  sorry

end family_park_cost_l898_89832


namespace isabel_ds_games_l898_89862

theorem isabel_ds_games (initial_games : ℕ) (remaining_games : ℕ) (given_games : ℕ) : 
  initial_games = 90 → remaining_games = 3 → given_games = initial_games - remaining_games → given_games = 87 := by
  sorry

end isabel_ds_games_l898_89862


namespace triangle_angles_theorem_l898_89826

theorem triangle_angles_theorem (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- Angles are positive
  A + B + C = π ∧ -- Sum of angles in a triangle
  A + C = 2 * B ∧ -- Given condition
  Real.tan A * Real.tan C = 2 + Real.sqrt 3 -- Given condition
  →
  ((A = π / 4 ∧ B = π / 3 ∧ C = 5 * π / 12) ∨
   (A = 5 * π / 12 ∧ B = π / 3 ∧ C = π / 4)) :=
by sorry

end triangle_angles_theorem_l898_89826


namespace triangle_segment_ratio_l898_89833

-- Define the triangle and points
variable (A B C E D F : ℝ × ℝ)

-- Define the conditions
variable (h_E_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • B)
variable (h_D_on_BC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ D = (1 - s) • B + s • C)
variable (h_AE_EB : ∃ k : ℝ, dist A E = k * dist E B ∧ k = 1/3)
variable (h_CD_DB : ∃ m : ℝ, dist C D = m * dist D B ∧ m = 1/2)
variable (h_F_intersect : ∃ u v : ℝ, 0 < u ∧ u < 1 ∧ 0 < v ∧ v < 1 ∧
  F = (1 - u) • A + u • D ∧ F = (1 - v) • C + v • E)

-- Define the theorem
theorem triangle_segment_ratio :
  dist E F / dist F C + dist A F / dist F D = 3/2 :=
sorry

end triangle_segment_ratio_l898_89833


namespace largest_constant_divisor_inequality_l898_89803

/-- The number of divisors function -/
def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The statement of the theorem -/
theorem largest_constant_divisor_inequality :
  (∃ (c : ℝ), c > 0 ∧
    (∀ (n : ℕ), n ≥ 2 →
      (∃ (d : ℕ), d > 0 ∧ d ∣ n ∧ (d : ℝ) ≤ Real.sqrt n ∧
        (tau d : ℝ) ≥ c * Real.sqrt (tau n : ℝ)))) ∧
  (∀ (c : ℝ), c > Real.sqrt (1 / 2) →
    (∃ (n : ℕ), n ≥ 2 ∧
      (∀ (d : ℕ), d > 0 → d ∣ n → (d : ℝ) ≤ Real.sqrt n →
        (tau d : ℝ) < c * Real.sqrt (tau n : ℝ)))) :=
by sorry

end largest_constant_divisor_inequality_l898_89803


namespace digit_sum_l898_89894

/-- Given two digits x and y, if 3x * y4 = 156, then x + y = 13 -/
theorem digit_sum (x y : Nat) : 
  x ≤ 9 → y ≤ 9 → (30 + x) * (10 * y + 4) = 156 → x + y = 13 := by
  sorry

end digit_sum_l898_89894


namespace positive_real_pair_with_integer_product_and_floor_sum_l898_89821

theorem positive_real_pair_with_integer_product_and_floor_sum (x y : ℝ) : 
  x > 0 → y > 0 → (∃ n : ℤ, x * y = n) → x + y = ⌊x^2 - y^2⌋ → 
  ∃ d : ℕ, d ≥ 2 ∧ x = d ∧ y = d - 1 := by
sorry

end positive_real_pair_with_integer_product_and_floor_sum_l898_89821


namespace bruce_age_bruce_current_age_l898_89851

theorem bruce_age : ℕ → Prop :=
  fun b =>
    let son_age : ℕ := 8
    let future_years : ℕ := 6
    (b + future_years = 3 * (son_age + future_years)) →
    b = 36

-- Proof
theorem bruce_current_age : ∃ b : ℕ, bruce_age b :=
  sorry

end bruce_age_bruce_current_age_l898_89851


namespace smallest_prime_dividing_sum_l898_89836

theorem smallest_prime_dividing_sum : ∃ p : ℕ, 
  Nat.Prime p ∧ p > 5 ∧ p ∣ (2^14 + 3^15) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (2^14 + 3^15) → q ≥ p := by
  sorry

end smallest_prime_dividing_sum_l898_89836


namespace ratio_of_divisor_sums_l898_89895

def M : ℕ := 35 * 36 * 65 * 280

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry
def sum_of_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_of_odd_divisors M) * 62 = sum_of_even_divisors M := by sorry

end ratio_of_divisor_sums_l898_89895


namespace spinner_probability_l898_89830

theorem spinner_probability : ∀ (p_C p_D p_E : ℚ),
  (p_C = p_D) →
  (p_D = p_E) →
  (1/5 : ℚ) + (1/5 : ℚ) + p_C + p_D + p_E = 1 →
  p_C = (1/5 : ℚ) := by
sorry

end spinner_probability_l898_89830


namespace frame_price_increase_l898_89811

theorem frame_price_increase (budget : ℝ) (remaining : ℝ) (ratio : ℝ) : 
  budget = 60 → 
  remaining = 6 → 
  ratio = 3/4 → 
  let smaller_frame_price := budget - remaining
  let initial_frame_price := smaller_frame_price / ratio
  (initial_frame_price - budget) / budget * 100 = 20 := by
sorry

end frame_price_increase_l898_89811


namespace marks_lawyer_hourly_rate_l898_89865

/-- Calculates the lawyer's hourly rate for Mark's speeding ticket case -/
theorem marks_lawyer_hourly_rate 
  (base_fine : ℕ) 
  (speed_fine_rate : ℕ) 
  (marks_speed : ℕ) 
  (speed_limit : ℕ) 
  (court_costs : ℕ) 
  (lawyer_hours : ℕ) 
  (total_owed : ℕ) 
  (h1 : base_fine = 50)
  (h2 : speed_fine_rate = 2)
  (h3 : marks_speed = 75)
  (h4 : speed_limit = 30)
  (h5 : court_costs = 300)
  (h6 : lawyer_hours = 3)
  (h7 : total_owed = 820) :
  (total_owed - (2 * (base_fine + speed_fine_rate * (marks_speed - speed_limit)) + court_costs)) / lawyer_hours = 80 := by
  sorry

end marks_lawyer_hourly_rate_l898_89865


namespace vessel_volume_ratio_l898_89804

/-- Represents a vessel containing a mixture of milk and water -/
structure Vessel where
  milk : ℚ
  water : ℚ

/-- The ratio of milk to water in a vessel -/
def milkWaterRatio (v : Vessel) : ℚ := v.milk / v.water

/-- The total volume of a vessel -/
def volume (v : Vessel) : ℚ := v.milk + v.water

/-- Combines the contents of two vessels -/
def combineVessels (v1 v2 : Vessel) : Vessel :=
  { milk := v1.milk + v2.milk, water := v1.water + v2.water }

theorem vessel_volume_ratio (v1 v2 : Vessel) :
  milkWaterRatio v1 = 1/2 →
  milkWaterRatio v2 = 6/4 →
  milkWaterRatio (combineVessels v1 v2) = 1 →
  volume v1 / volume v2 = 9/5 := by
  sorry

end vessel_volume_ratio_l898_89804


namespace lcm_of_135_and_195_l898_89864

theorem lcm_of_135_and_195 : Nat.lcm 135 195 = 1755 := by
  sorry

end lcm_of_135_and_195_l898_89864


namespace club_truncator_probability_l898_89818

/-- Represents the outcome of a single match -/
inductive MatchResult
  | Win
  | Loss
  | Tie

/-- The total number of matches played by Club Truncator -/
def total_matches : ℕ := 5

/-- The probability of each match result -/
def match_probability : ℚ := 1 / 3

/-- Calculates the probability of having more wins than losses in the season -/
noncomputable def prob_more_wins_than_losses : ℚ := sorry

/-- The main theorem stating the probability of more wins than losses -/
theorem club_truncator_probability : prob_more_wins_than_losses = 32 / 81 := by sorry

end club_truncator_probability_l898_89818


namespace line_points_determine_k_l898_89858

/-- A line contains the points (6,10), (-2,k), and (-10,6). -/
def line_contains_points (k : ℝ) : Prop :=
  ∃ (m b : ℝ), 
    (10 = m * 6 + b) ∧
    (k = m * (-2) + b) ∧
    (6 = m * (-10) + b)

/-- If a line contains the points (6,10), (-2,k), and (-10,6), then k = 8. -/
theorem line_points_determine_k :
  ∀ k : ℝ, line_contains_points k → k = 8 :=
by
  sorry

end line_points_determine_k_l898_89858


namespace cricket_equipment_cost_l898_89861

theorem cricket_equipment_cost (bat_cost : ℕ) (ball_cost : ℕ) : 
  (7 * bat_cost + 6 * ball_cost = 3800) →
  (3 * bat_cost + 5 * ball_cost = 1750) →
  (bat_cost = 500) →
  ball_cost = 50 := by
sorry

end cricket_equipment_cost_l898_89861


namespace expression_factorization_l898_89823

/-- 
Given a, b, and c, prove that the expression 
a^4 (b^2 - c^2) + b^4 (c^2 - a^2) + c^4 (a^2 - b^2) 
can be factorized into the form (a - b)(b - c)(c - a) q(a, b, c),
where q(a, b, c) = a^3 b^2 + a^2 b^3 + b^3 c^2 + b^2 c^3 + c^3 a^2 + c^2 a^3
-/
theorem expression_factorization (a b c : ℝ) : 
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) = 
  (a - b) * (b - c) * (c - a) * (a^3 * b^2 + a^2 * b^3 + b^3 * c^2 + b^2 * c^3 + c^3 * a^2 + c^2 * a^3) := by
  sorry

end expression_factorization_l898_89823


namespace sequence_eventually_periodic_l898_89820

def is_eventually_periodic (a : ℕ → ℚ) : Prop :=
  ∃ k m : ℕ, k > 0 ∧ ∀ n ≥ m, a (n + k) = a n

theorem sequence_eventually_periodic
  (a : ℕ → ℚ)
  (h1 : ∀ n : ℕ, |a (n + 1) - 2 * a n| = 2)
  (h2 : ∀ n : ℕ, |a n| ≤ 2)
  : is_eventually_periodic a :=
sorry

end sequence_eventually_periodic_l898_89820


namespace bob_earnings_l898_89822

def regular_rate : ℕ := 5
def overtime_rate : ℕ := 6
def regular_hours : ℕ := 40
def first_week_hours : ℕ := 44
def second_week_hours : ℕ := 48

def calculate_earnings (hours_worked : ℕ) : ℕ :=
  regular_rate * regular_hours + 
  overtime_rate * (hours_worked - regular_hours)

theorem bob_earnings : 
  calculate_earnings first_week_hours + calculate_earnings second_week_hours = 472 := by
  sorry

end bob_earnings_l898_89822


namespace total_spent_is_36_98_l898_89885

/-- Calculates the total amount spent on video games --/
def total_spent (football_price : ℝ) (football_discount : ℝ) 
                (strategy_price : ℝ) (strategy_tax : ℝ)
                (batman_price_euro : ℝ) (exchange_rate : ℝ) : ℝ :=
  let football_discounted := football_price * (1 - football_discount)
  let strategy_with_tax := strategy_price * (1 + strategy_tax)
  let batman_price_usd := batman_price_euro * exchange_rate
  football_discounted + strategy_with_tax + batman_price_usd

/-- Theorem stating the total amount spent on video games --/
theorem total_spent_is_36_98 :
  total_spent 16 0.1 9.46 0.05 11 1.15 = 36.98 := by
  sorry

end total_spent_is_36_98_l898_89885


namespace arithmetic_sequence_property_l898_89845

/-- An arithmetic sequence {a_n} with a_2 = 2 and S_11 = 66 -/
def a (n : ℕ) : ℚ :=
  sorry

/-- The sum of the first n terms of the sequence a -/
def S (n : ℕ) : ℚ :=
  sorry

/-- The sequence b_n defined as 1 / (a_n * a_n+1) -/
def b (n : ℕ) : ℚ :=
  1 / (a n * a (n + 1))

/-- The sum of the first n terms of sequence b -/
def b_sum (n : ℕ) : ℚ :=
  sorry

theorem arithmetic_sequence_property :
  a 2 = 2 ∧ S 11 = 66 ∧ ∀ n : ℕ, n > 0 → b_sum n < 1 :=
  sorry

end arithmetic_sequence_property_l898_89845


namespace always_bal_answer_l898_89816

/-- Represents a guest in the castle -/
structure Guest where
  is_reliable : Bool

/-- Represents the possible questions that can be asked -/
inductive Question
  | q1  -- "Правильно ли ответить «бaл» на вопрос, надежны ли вы?"
  | q2  -- "Надежны ли вы в том и только в том случае, если «бaл» означает «да»?"

/-- The answer "бaл" -/
def bal : String := "бaл"

/-- Function representing a guest's response to a question -/
def guest_response (g : Guest) (q : Question) : String :=
  match q with
  | Question.q1 => bal
  | Question.q2 => bal

/-- Theorem stating that any guest will always answer "бaл" to either question -/
theorem always_bal_answer (g : Guest) (q : Question) :
  guest_response g q = bal := by sorry

end always_bal_answer_l898_89816


namespace power_inequality_l898_89834

theorem power_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ (a*b*c)^(a/5) := by
  sorry

end power_inequality_l898_89834


namespace factor_expression_l898_89873

theorem factor_expression (x : ℝ) : 3*x*(x+3) + 2*(x+3) = (x+3)*(3*x+2) := by
  sorry

end factor_expression_l898_89873


namespace math_team_combinations_l898_89828

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of girls in the math club --/
def num_girls : ℕ := 4

/-- The number of boys in the math club --/
def num_boys : ℕ := 6

/-- The number of girls to be chosen for the team --/
def girls_in_team : ℕ := 2

/-- The number of boys to be chosen for the team --/
def boys_in_team : ℕ := 2

theorem math_team_combinations :
  (choose num_girls girls_in_team) * (choose num_boys boys_in_team) = 90 := by
  sorry

end math_team_combinations_l898_89828


namespace difference_of_squares_l898_89853

theorem difference_of_squares : 601^2 - 597^2 = 4792 := by sorry

end difference_of_squares_l898_89853


namespace sum_of_cubes_of_roots_l898_89831

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - x^2 + x - 2

-- Define the roots
variable (p q r : ℝ)

-- State the theorem
theorem sum_of_cubes_of_roots :
  (f p = 0) → (f q = 0) → (f r = 0) → 
  p ≠ q → q ≠ r → r ≠ p →
  p^3 + q^3 + r^3 = 4 := by sorry

end sum_of_cubes_of_roots_l898_89831


namespace all_lines_pass_through_fixed_point_l898_89815

/-- A line in the xy-plane defined by the equation kx - y + 1 = k, where k is a real number. -/
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | k * p.1 - p.2 + 1 = k}

/-- The fixed point (1, 1) -/
def fixed_point : ℝ × ℝ := (1, 1)

/-- Theorem stating that all lines pass through the fixed point (1, 1) -/
theorem all_lines_pass_through_fixed_point :
  ∀ k : ℝ, fixed_point ∈ line k := by
  sorry


end all_lines_pass_through_fixed_point_l898_89815


namespace consecutive_integer_product_l898_89878

theorem consecutive_integer_product (n : ℤ) : 
  (6 ∣ n * (n + 1)) ∨ (n * (n + 1) % 18 = 2) := by
  sorry

end consecutive_integer_product_l898_89878


namespace function_domain_range_implies_b_equals_two_l898_89841

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 4

-- Define the theorem
theorem function_domain_range_implies_b_equals_two :
  ∀ b : ℝ,
  (∀ x ∈ Set.Icc 2 (2*b), f x ∈ Set.Icc 2 (2*b)) ∧
  (∀ y ∈ Set.Icc 2 (2*b), ∃ x ∈ Set.Icc 2 (2*b), f x = y) →
  b = 2 :=
by sorry

end function_domain_range_implies_b_equals_two_l898_89841


namespace georginas_parrot_days_l898_89856

/-- The number of days Georgina has had her parrot -/
def days_with_parrot (total_phrases current_phrases_per_week initial_phrases days_per_week : ℕ) : ℕ :=
  ((total_phrases - initial_phrases) / current_phrases_per_week) * days_per_week

/-- Proof that Georgina has had her parrot for 49 days -/
theorem georginas_parrot_days : 
  days_with_parrot 17 2 3 7 = 49 := by
  sorry

end georginas_parrot_days_l898_89856


namespace base7_addition_multiplication_l898_89860

/-- Converts a base 7 number represented as a list of digits to its decimal equivalent -/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a decimal number to its base 7 representation as a list of digits -/
def decimalToBase7 (n : Nat) : List Nat :=
  if n < 7 then [n]
  else (n % 7) :: decimalToBase7 (n / 7)

/-- Adds two base 7 numbers -/
def addBase7 (a b : List Nat) : List Nat :=
  decimalToBase7 (base7ToDecimal a + base7ToDecimal b)

/-- Multiplies a base 7 number by another base 7 number -/
def mulBase7 (a b : List Nat) : List Nat :=
  decimalToBase7 (base7ToDecimal a * base7ToDecimal b)

theorem base7_addition_multiplication :
  mulBase7 (addBase7 [5, 2] [4, 3, 3]) [2] = [4, 6, 6] := by sorry

end base7_addition_multiplication_l898_89860


namespace product_sum_fractions_l898_89835

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end product_sum_fractions_l898_89835


namespace sqrt_nested_expression_l898_89839

theorem sqrt_nested_expression : Real.sqrt (16 * Real.sqrt (8 * Real.sqrt 4)) = 8 := by sorry

end sqrt_nested_expression_l898_89839


namespace inheritance_problem_l898_89805

/-- The inheritance problem -/
theorem inheritance_problem (x : ℝ) 
  (h1 : 0.25 * x + 0.15 * x = 15000) : x = 37500 := by
  sorry

end inheritance_problem_l898_89805


namespace intersection_of_A_and_B_l898_89854

-- Define set A
def A : Set ℝ := {x : ℝ | |x| ≤ 1}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
by sorry

end intersection_of_A_and_B_l898_89854


namespace hoseok_candy_count_l898_89875

/-- The number of candies Hoseok has of type A -/
def candies_A : ℕ := 2

/-- The number of candies Hoseok has of type B -/
def candies_B : ℕ := 5

/-- The total number of candies Hoseok has -/
def total_candies : ℕ := candies_A + candies_B

theorem hoseok_candy_count : total_candies = 7 := by
  sorry

end hoseok_candy_count_l898_89875


namespace taxi_fare_calculation_l898_89819

/-- Proves that the charge for each additional 1/5 mile is $0.40 given the initial and total charges -/
theorem taxi_fare_calculation (initial_charge : ℚ) (total_charge : ℚ) (ride_length : ℚ) 
  (h1 : initial_charge = 2.1)
  (h2 : total_charge = 17.7)
  (h3 : ride_length = 8) :
  let additional_increments := (ride_length * 5) - 1
  (total_charge - initial_charge) / additional_increments = 0.4 := by
  sorry

end taxi_fare_calculation_l898_89819


namespace triangle_problem_l898_89824

theorem triangle_problem (DC CB : ℝ) (h1 : DC = 12) (h2 : CB = 9)
  (AD : ℝ) (h3 : AD > 0)
  (AB : ℝ) (h4 : AB = (1/3) * AD)
  (ED : ℝ) (h5 : ED = (3/4) * AD) :
  ∃ FC : ℝ, FC = 14.625 := by
sorry

end triangle_problem_l898_89824


namespace f_unique_zero_l898_89810

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - (a + 1) * x + a * Real.log x

theorem f_unique_zero (a : ℝ) (h : a > 0) : 
  ∃! x : ℝ, x > 0 ∧ f a x = 0 :=
by sorry

end f_unique_zero_l898_89810


namespace value_of_k_l898_89899

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define non-collinear vectors e₁ and e₂
variable (e₁ e₂ : V)
variable (h_non_collinear : ∀ (a b : ℝ), a • e₁ + b • e₂ = 0 → a = 0 ∧ b = 0)

-- Define points and vectors
variable (A B C D : V)
variable (k : ℝ)

-- Define the given vector relationships
variable (h_AB : B - A = 2 • e₁ + k • e₂)
variable (h_CB : B - C = e₁ + 3 • e₂)
variable (h_CD : D - C = 2 • e₁ - e₂)

-- Define collinearity of points A, B, and D
variable (h_collinear : ∃ (t : ℝ), B - A = t • (D - B))

-- Theorem statement
theorem value_of_k : k = -8 := by sorry

end value_of_k_l898_89899


namespace embroidery_project_time_l898_89871

/-- Represents the embroidery project details -/
structure EmbroideryProject where
  flower_stitches : ℕ
  flower_speed : ℕ
  unicorn_stitches : ℕ
  unicorn_speed : ℕ
  godzilla_stitches : ℕ
  godzilla_speed : ℕ
  num_flowers : ℕ
  num_unicorns : ℕ
  num_godzilla : ℕ
  break_duration : ℕ
  work_duration : ℕ

/-- Calculates the total time needed for the embroidery project -/
def total_time (project : EmbroideryProject) : ℕ :=
  let total_stitches := project.flower_stitches * project.num_flowers +
                        project.unicorn_stitches * project.num_unicorns +
                        project.godzilla_stitches * project.num_godzilla
  let total_work_time := (total_stitches / project.flower_speed * project.num_flowers +
                          total_stitches / project.unicorn_speed * project.num_unicorns +
                          total_stitches / project.godzilla_speed * project.num_godzilla)
  let num_breaks := total_work_time / project.work_duration
  let total_break_time := num_breaks * project.break_duration
  total_work_time + total_break_time

/-- The main theorem stating the total time for the given embroidery project -/
theorem embroidery_project_time :
  let project : EmbroideryProject := {
    flower_stitches := 60,
    flower_speed := 4,
    unicorn_stitches := 180,
    unicorn_speed := 5,
    godzilla_stitches := 800,
    godzilla_speed := 3,
    num_flowers := 50,
    num_unicorns := 3,
    num_godzilla := 1,
    break_duration := 5,
    work_duration := 30
  }
  total_time project = 1310 := by
  sorry


end embroidery_project_time_l898_89871


namespace digit_difference_in_base_d_l898_89882

/-- A digit in base d is a natural number less than d. -/
def Digit (d : ℕ) := { n : ℕ // n < d }

/-- The value of a two-digit number AB in base d. -/
def TwoDigitValue (d : ℕ) (A B : Digit d) : ℕ := A.val * d + B.val

theorem digit_difference_in_base_d (d : ℕ) (A B : Digit d) 
  (h_d : d > 7)
  (h_sum : TwoDigitValue d A B + TwoDigitValue d A A = 1 * d * d + 7 * d + 2) :
  A.val - B.val = 5 := by
  sorry

end digit_difference_in_base_d_l898_89882


namespace new_person_weight_l898_89896

/-- The weight of the new person given the conditions of the problem -/
theorem new_person_weight (n : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  n = 15 ∧ weight_increase = 3.8 ∧ replaced_weight = 75 →
  n * weight_increase + replaced_weight = 132 := by
sorry

end new_person_weight_l898_89896


namespace problem_solution_l898_89892

theorem problem_solution (a b c : ℝ) 
  (h1 : a * b * c = 1) 
  (h2 : a + b + c = 2) 
  (h3 : a^2 + b^2 + c^2 = 3) : 
  1 / (a * b + c - 1) + 1 / (b * c + a - 1) + 1 / (c * a + b - 1) = 2 := by
sorry

end problem_solution_l898_89892


namespace ram_distances_l898_89886

/-- Represents a mountain on the map -/
structure Mountain where
  name : String
  scale : ℝ  -- km per inch

/-- Represents a location on the map -/
structure Location where
  name : String
  distanceA : ℝ  -- distance from mountain A in inches
  distanceB : ℝ  -- distance from mountain B in inches

def map_distance : ℝ := 312  -- inches
def actual_distance : ℝ := 136  -- km

def mountainA : Mountain := { name := "A", scale := 1 }
def mountainB : Mountain := { name := "B", scale := 2 }

def ram_location : Location := { name := "Ram", distanceA := 25, distanceB := 40 }

/-- Calculates the actual distance from a location to a mountain -/
def actual_distance_to_mountain (loc : Location) (m : Mountain) : ℝ :=
  if m.name = "A" then loc.distanceA * m.scale else loc.distanceB * m.scale

theorem ram_distances :
  actual_distance_to_mountain ram_location mountainA = 25 ∧
  actual_distance_to_mountain ram_location mountainB = 80 := by
  sorry

end ram_distances_l898_89886


namespace seans_total_spend_is_21_l898_89868

/-- The total amount Sean spent on his Sunday purchases -/
def seans_total_spend : ℝ :=
  let almond_croissant : ℝ := 4.50
  let salami_cheese_croissant : ℝ := 4.50
  let plain_croissant : ℝ := 3.00
  let focaccia : ℝ := 4.00
  let latte : ℝ := 2.50
  let num_lattes : ℕ := 2

  almond_croissant + salami_cheese_croissant + plain_croissant + focaccia + (num_lattes : ℝ) * latte

/-- Theorem stating that Sean's total spend is $21.00 -/
theorem seans_total_spend_is_21 : seans_total_spend = 21 := by
  sorry

end seans_total_spend_is_21_l898_89868


namespace sum_of_sqrt_sequence_l898_89817

theorem sum_of_sqrt_sequence :
  Real.sqrt 6 + Real.sqrt (6 + 8) + Real.sqrt (6 + 8 + 10) + 
  Real.sqrt (6 + 8 + 10 + 12) + Real.sqrt (6 + 8 + 10 + 12 + 14) = 
  Real.sqrt 6 + Real.sqrt 14 + Real.sqrt 24 + 6 + 5 * Real.sqrt 2 := by
  sorry

end sum_of_sqrt_sequence_l898_89817


namespace negative_sum_l898_89891

theorem negative_sum : (-2) + (-5) = -7 := by
  sorry

end negative_sum_l898_89891


namespace lakers_win_in_seven_games_l898_89889

/-- The probability of the Knicks winning a single game -/
def p_knicks_win : ℚ := 3/4

/-- The probability of the Lakers winning a single game -/
def p_lakers_win : ℚ := 1 - p_knicks_win

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The maximum number of games in the series -/
def max_games : ℕ := 7

/-- The number of ways to choose 3 wins from 6 games -/
def ways_to_choose_3_from_6 : ℕ := 20

theorem lakers_win_in_seven_games :
  let p_lakers_win_series := (ways_to_choose_3_from_6 : ℚ) * p_lakers_win^3 * p_knicks_win^3 * p_lakers_win
  p_lakers_win_series = 540/16384 := by sorry

end lakers_win_in_seven_games_l898_89889


namespace ratio_a5_a7_l898_89877

/-- A positive geometric sequence with specific properties -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  decreasing : ∀ n, a (n + 1) < a n
  geometric : ∀ n k, a (n + k) = a n * (a 2 / a 1) ^ k
  prop1 : a 2 * a 8 = 6
  prop2 : a 4 + a 6 = 5

/-- The main theorem about the ratio of a_5 to a_7 -/
theorem ratio_a5_a7 (seq : SpecialGeometricSequence) : seq.a 5 / seq.a 7 = 3 / 2 := by
  sorry

end ratio_a5_a7_l898_89877


namespace polynomial_identity_l898_89888

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) : 
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end polynomial_identity_l898_89888


namespace profit_percentage_l898_89838

theorem profit_percentage (cost selling : ℝ) (h : cost > 0) :
  60 * cost = 40 * selling →
  (selling - cost) / cost * 100 = 50 :=
by
  sorry

end profit_percentage_l898_89838


namespace q_is_false_l898_89898

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q :=
sorry

end q_is_false_l898_89898


namespace quadratic_function_properties_l898_89812

def y (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

theorem quadratic_function_properties :
  ∀ (a b : ℝ),
  (∀ x : ℝ, y a b x < 0 ↔ 1 < x ∧ x < 3) →
  a > 0 →
  b = -2 * a →
  (a = 1 ∧ b = -2) ∧
  (∀ x : ℝ,
    y a b x ≤ -1 ↔
      ((0 < a ∧ a < 1 → 2 ≤ x ∧ x ≤ 2/a) ∧
       (a = 1 → x = 2) ∧
       (a > 1 → 2/a ≤ x ∧ x ≤ 2))) :=
by sorry

end quadratic_function_properties_l898_89812


namespace complex_modulus_problem_l898_89847

theorem complex_modulus_problem (z : ℂ) (h : z - 2*Complex.I = z*Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l898_89847


namespace range_of_a_plus_3b_l898_89814

theorem range_of_a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b) (h2 : a + b ≤ 1) 
  (h3 : 1 ≤ a - 2*b) (h4 : a - 2*b ≤ 3) : 
  ∃ (x : ℝ), x = a + 3*b ∧ -11/3 ≤ x ∧ x ≤ 1 :=
sorry

end range_of_a_plus_3b_l898_89814


namespace necessary_not_sufficient_l898_89849

theorem necessary_not_sufficient (a b : ℝ) : 
  ((a > b) → (a > b - 1)) ∧ ¬((a > b - 1) → (a > b)) := by sorry

end necessary_not_sufficient_l898_89849


namespace clock_cost_price_l898_89843

theorem clock_cost_price (total_clocks : ℕ) (sold_at_10_percent : ℕ) (sold_at_20_percent : ℕ) 
  (uniform_profit_difference : ℝ) :
  total_clocks = 90 →
  sold_at_10_percent = 40 →
  sold_at_20_percent = 50 →
  uniform_profit_difference = 40 →
  ∃ (cost_price : ℝ),
    cost_price = 80 ∧
    (sold_at_10_percent : ℝ) * cost_price * 1.1 + 
    (sold_at_20_percent : ℝ) * cost_price * 1.2 - 
    (total_clocks : ℝ) * cost_price * 1.15 = uniform_profit_difference :=
by sorry

end clock_cost_price_l898_89843


namespace coin_flip_difference_l898_89874

/-- Given 211 total coin flips with 65 heads, the difference between the number of tails and heads is 81. -/
theorem coin_flip_difference (total_flips : ℕ) (heads : ℕ) 
    (h1 : total_flips = 211)
    (h2 : heads = 65) : 
  total_flips - heads - heads = 81 := by
  sorry

end coin_flip_difference_l898_89874


namespace trigonometric_identity_l898_89890

theorem trigonometric_identity : 
  (Real.sin (10 * π / 180) * Real.sin (80 * π / 180)) / 
  (Real.cos (35 * π / 180) ^ 2 - Real.sin (35 * π / 180) ^ 2) = 1 / 2 := by
  sorry

end trigonometric_identity_l898_89890


namespace xy_value_when_sum_of_abs_is_zero_l898_89883

theorem xy_value_when_sum_of_abs_is_zero (x y : ℝ) :
  |x - 1| + |y + 2| = 0 → x * y = -2 := by
sorry

end xy_value_when_sum_of_abs_is_zero_l898_89883


namespace largest_integer_satisfying_inequality_l898_89806

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, 8 - 5*x > 25 → x ≤ -4 ∧ 8 - 5*(-4) > 25 :=
by sorry

end largest_integer_satisfying_inequality_l898_89806


namespace smallest_among_four_l898_89857

theorem smallest_among_four (a b c d : ℝ) :
  a = |-2| ∧ b = -1 ∧ c = 0 ∧ d = -1/2 →
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by sorry

end smallest_among_four_l898_89857


namespace consecutive_odd_power_sum_divisibility_l898_89829

theorem consecutive_odd_power_sum_divisibility (p q m n : ℕ) : 
  Odd p → Odd q → p = q + 2 → Odd m → Odd n → m > 0 → n > 0 → 
  ∃ k : ℤ, p^m + q^n = k * (p + q) :=
sorry

end consecutive_odd_power_sum_divisibility_l898_89829


namespace set_inclusion_implies_a_range_l898_89855

/-- Given sets A, B, and C defined as follows:
    A = {x | 2 ≤ x < 7}
    B = {x | 3 < x ≤ 10}
    C = {x | a-5 < x < a}
    Prove that if C is a non-empty subset of A ∪ B, then 7 ≤ a ≤ 10. -/
theorem set_inclusion_implies_a_range (a : ℝ) :
  let A : Set ℝ := {x | 2 ≤ x ∧ x < 7}
  let B : Set ℝ := {x | 3 < x ∧ x ≤ 10}
  let C : Set ℝ := {x | a - 5 < x ∧ x < a}
  C.Nonempty → C ⊆ A ∪ B → 7 ≤ a ∧ a ≤ 10 := by
  sorry


end set_inclusion_implies_a_range_l898_89855


namespace intersection_implies_a_gt_three_l898_89802

/-- A function f(x) = x³ - ax² + 4 that intersects the positive x-axis at two different points -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

/-- The property that f intersects the positive x-axis at two different points -/
def intersects_positive_x_axis_twice (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0

/-- If f(x) = x³ - ax² + 4 intersects the positive x-axis at two different points, then a > 3 -/
theorem intersection_implies_a_gt_three :
  ∀ a : ℝ, intersects_positive_x_axis_twice a → a > 3 :=
sorry

end intersection_implies_a_gt_three_l898_89802


namespace cricket_players_l898_89827

theorem cricket_players (B C Both Total : ℕ) : 
  B = 7 → 
  Both = 3 → 
  Total = 9 → 
  Total = B + C - Both → 
  C = 5 :=
by sorry

end cricket_players_l898_89827


namespace work_left_after_collaboration_l898_89859

/-- Represents the fraction of work completed in one day -/
def work_rate (days : ℕ) : ℚ := 1 / days

/-- Represents the total work completed by two people in a given number of days -/
def total_work (rate_a rate_b : ℚ) (days : ℕ) : ℚ := (rate_a + rate_b) * days

theorem work_left_after_collaboration (days_a days_b collab_days : ℕ) 
  (h1 : days_a = 15) (h2 : days_b = 20) (h3 : collab_days = 4) : 
  1 - total_work (work_rate days_a) (work_rate days_b) collab_days = 8 / 15 := by
  sorry

#check work_left_after_collaboration

end work_left_after_collaboration_l898_89859


namespace original_number_is_ten_l898_89809

theorem original_number_is_ten : ∃ x : ℝ, (2 * x + 5 = x / 2 + 20) ∧ (x = 10) := by
  sorry

end original_number_is_ten_l898_89809


namespace garden_furniture_cost_l898_89850

/-- The combined cost of a garden table and a bench -/
def combined_cost (bench_cost : ℝ) (table_cost : ℝ) : ℝ :=
  bench_cost + table_cost

theorem garden_furniture_cost :
  ∀ (bench_cost : ℝ) (table_cost : ℝ),
  bench_cost = 250.0 →
  table_cost = 2 * bench_cost →
  combined_cost bench_cost table_cost = 750.0 := by
sorry

end garden_furniture_cost_l898_89850


namespace min_value_sum_l898_89808

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 / x) + (8 / y) = 1) : x + y ≥ 18 := by
  sorry

end min_value_sum_l898_89808
