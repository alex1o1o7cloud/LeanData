import Mathlib

namespace crayons_remaining_l72_7222

theorem crayons_remaining (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) : 
  initial_crayons = 48 → 
  kiley_fraction = 1/4 →
  joe_fraction = 1/2 →
  (initial_crayons - (kiley_fraction * initial_crayons).floor - 
   (joe_fraction * (initial_crayons - (kiley_fraction * initial_crayons).floor)).floor) = 18 :=
by sorry

end crayons_remaining_l72_7222


namespace max_intersection_points_l72_7223

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of intersection points between a circle and a line --/
def intersection_count (circle : Circle) (line : Line) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of intersection points between a circle and a line is 2 --/
theorem max_intersection_points (circle : Circle) (line : Line) :
  intersection_count circle line ≤ 2 :=
sorry

end max_intersection_points_l72_7223


namespace train_speed_train_speed_is_24_l72_7271

theorem train_speed (person_speed : ℝ) (overtake_time : ℝ) (train_length : ℝ) : ℝ :=
  let relative_speed := train_length / overtake_time * 3600 / 1000
  relative_speed + person_speed

#check train_speed 4 9 49.999999999999986 = 24

theorem train_speed_is_24 :
  train_speed 4 9 49.999999999999986 = 24 := by sorry

end train_speed_train_speed_is_24_l72_7271


namespace kostya_bulbs_count_l72_7276

/-- Function to calculate the number of bulbs after one round of planting -/
def plant_between (n : ℕ) : ℕ := 2 * n - 1

/-- Function to calculate the number of bulbs after three rounds of planting -/
def plant_three_rounds (n : ℕ) : ℕ := plant_between (plant_between (plant_between n))

/-- Theorem stating that if Kostya planted n bulbs and the final count after three rounds is 113, then n must be 15 -/
theorem kostya_bulbs_count : 
  ∀ n : ℕ, plant_three_rounds n = 113 → n = 15 := by
sorry

#eval plant_three_rounds 15  -- Should output 113

end kostya_bulbs_count_l72_7276


namespace decimal_333_to_octal_l72_7203

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The octal representation of decimal 333 is 515 -/
theorem decimal_333_to_octal :
  decimal_to_octal 333 = 515 := by
  sorry

end decimal_333_to_octal_l72_7203


namespace twenty_matches_exist_l72_7287

/-- Represents the number of matches played up to each day -/
def MatchSequence : Type := Fin 71 → ℕ

/-- The sequence is non-decreasing -/
def IsNonDecreasing (s : MatchSequence) : Prop :=
  ∀ i j : Fin 71, i < j → s i ≤ s j

/-- The difference between consecutive days is at least 1 and at most 12 -/
def HasValidDifference (s : MatchSequence) : Prop :=
  ∀ i : Fin 70, 1 ≤ s (i + 1) - s i ∧ s (i + 1) - s i ≤ 12

/-- The total number of matches played in 70 days does not exceed 120 -/
def HasValidTotal (s : MatchSequence) : Prop :=
  s ⟨70, by norm_num⟩ ≤ 120

theorem twenty_matches_exist (s : MatchSequence)
  (h1 : IsNonDecreasing s)
  (h2 : HasValidDifference s)
  (h3 : HasValidTotal s)
  (h4 : s 0 = 0) :
  ∃ i j : Fin 71, i < j ∧ s j - s i = 20 := by
  sorry


end twenty_matches_exist_l72_7287


namespace sqrt_x_div_sqrt_y_l72_7229

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  (1/3)^2 + (1/4)^2 = (13*x / 53*y) * ((1/5)^2 + (1/6)^2) →
  Real.sqrt x / Real.sqrt y = 1092 / 338 := by
  sorry

end sqrt_x_div_sqrt_y_l72_7229


namespace largest_coeff_x3_sum_64_l72_7244

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The condition that the coefficient of x^3 is the largest in (1+x)^n -/
def coeff_x3_largest (n : ℕ) : Prop :=
  ∀ k, k ≠ 3 → binomial n 3 ≥ binomial n k

/-- The sum of all coefficients in the expansion of (1+x)^n -/
def sum_coefficients (n : ℕ) : ℕ := 2^n

theorem largest_coeff_x3_sum_64 :
  ∀ n : ℕ, coeff_x3_largest n → sum_coefficients n = 64 := by sorry

end largest_coeff_x3_sum_64_l72_7244


namespace consecutive_even_sum_l72_7274

theorem consecutive_even_sum (n : ℤ) : 
  (∃ (a b c d : ℤ), 
    a = n ∧ 
    b = n + 2 ∧ 
    c = n + 4 ∧ 
    d = n + 6 ∧ 
    c = 14) → 
  (n + (n + 2) + (n + 4) + (n + 6) = 52) := by
sorry

end consecutive_even_sum_l72_7274


namespace total_books_l72_7282

/-- The total number of books Tim, Sam, and Emma have together is 133. -/
theorem total_books (tim_books sam_books emma_books : ℕ) 
  (h1 : tim_books = 44)
  (h2 : sam_books = 52)
  (h3 : emma_books = 37) : 
  tim_books + sam_books + emma_books = 133 := by
  sorry

end total_books_l72_7282


namespace impossible_three_quadratics_with_two_roots_l72_7280

theorem impossible_three_quadratics_with_two_roots :
  ¬ ∃ (a b c : ℝ),
    (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) ∧
    (∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ c * y₁^2 + a * y₁ + b = 0 ∧ c * y₂^2 + a * y₂ + b = 0) ∧
    (∃ (z₁ z₂ : ℝ), z₁ ≠ z₂ ∧ b * z₁^2 + c * z₁ + a = 0 ∧ b * z₂^2 + c * z₂ + a = 0) :=
by sorry

end impossible_three_quadratics_with_two_roots_l72_7280


namespace alyssa_fruit_expenditure_l72_7218

/-- The amount Alyssa paid for grapes in dollars -/
def grapes_cost : ℚ := 12.08

/-- The amount Alyssa paid for cherries in dollars -/
def cherries_cost : ℚ := 9.85

/-- The total amount Alyssa spent on fruits -/
def total_cost : ℚ := grapes_cost + cherries_cost

theorem alyssa_fruit_expenditure : total_cost = 21.93 := by
  sorry

end alyssa_fruit_expenditure_l72_7218


namespace curve_properties_l72_7279

-- Define the curve
def curve (x y : ℝ) : Prop := x * y = 6

-- Define the property of the tangent being bisected
def tangent_bisected (x y : ℝ) : Prop :=
  ∀ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 →
    (curve x y) →
    (a * y = x * b) →
    ((x - 0) ^ 2 + (y - 0) ^ 2 = (a - x) ^ 2 + (0 - y) ^ 2) ∧
    ((x - 0) ^ 2 + (y - 0) ^ 2 = (0 - x) ^ 2 + (b - y) ^ 2)

theorem curve_properties :
  (curve 2 3) ∧
  (∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 → curve x y → tangent_bisected x y) :=
by sorry

end curve_properties_l72_7279


namespace max_area_triangle_OAB_l72_7296

/-- The maximum area of triangle OAB in the complex plane -/
theorem max_area_triangle_OAB :
  ∀ (α β : ℂ),
  β = (1 + Complex.I) * α →
  Complex.abs (α - 2) = 1 →
  (∀ (S : ℝ),
    S = (Complex.abs α * Complex.abs β * Real.sin (Real.pi / 4)) / 2 →
    S ≤ 9 / 2) ∧
  ∃ (α₀ β₀ : ℂ),
    β₀ = (1 + Complex.I) * α₀ ∧
    Complex.abs (α₀ - 2) = 1 ∧
    (Complex.abs α₀ * Complex.abs β₀ * Real.sin (Real.pi / 4)) / 2 = 9 / 2 :=
by sorry

end max_area_triangle_OAB_l72_7296


namespace max_negative_integers_l72_7214

theorem max_negative_integers
  (a b c d e f : ℤ)
  (h : a * b + c * d * e * f < 0) :
  ∃ (neg_count : ℕ),
    neg_count ≤ 4 ∧
    (∃ (na nb nc nd ne nf : ℕ),
      (na + nb + nc + nd + ne + nf = neg_count) ∧
      (a < 0 ↔ na = 1) ∧
      (b < 0 ↔ nb = 1) ∧
      (c < 0 ↔ nc = 1) ∧
      (d < 0 ↔ nd = 1) ∧
      (e < 0 ↔ ne = 1) ∧
      (f < 0 ↔ nf = 1)) ∧
    ∀ (m : ℕ), m > neg_count →
      ¬∃ (ma mb mc md me mf : ℕ),
        (ma + mb + mc + md + me + mf = m) ∧
        (a < 0 ↔ ma = 1) ∧
        (b < 0 ↔ mb = 1) ∧
        (c < 0 ↔ mc = 1) ∧
        (d < 0 ↔ md = 1) ∧
        (e < 0 ↔ me = 1) ∧
        (f < 0 ↔ mf = 1) := by
  sorry

end max_negative_integers_l72_7214


namespace correct_answers_count_l72_7253

/-- Represents a test with a specific scoring system. -/
structure Test where
  total_questions : ℕ
  score : ℕ → ℕ → ℤ
  all_answered : ℕ → ℕ → Prop

/-- Theorem stating the number of correct answers given the test conditions. -/
theorem correct_answers_count (test : Test)
    (h_total : test.total_questions = 100)
    (h_score : ∀ c i, test.score c i = c - 2 * i)
    (h_all_answered : ∀ c i, test.all_answered c i ↔ c + i = test.total_questions)
    (h_student_score : ∃ c i, test.all_answered c i ∧ test.score c i = 73) :
    ∃ c i, test.all_answered c i ∧ test.score c i = 73 ∧ c = 91 := by
  sorry

#check correct_answers_count

end correct_answers_count_l72_7253


namespace complement_of_M_l72_7257

def U : Set Nat := {1, 2, 3, 4}

def M : Set Nat := {x ∈ U | x^2 - 5*x + 6 = 0}

theorem complement_of_M :
  (U \ M) = {1, 4} := by sorry

end complement_of_M_l72_7257


namespace work_completion_time_l72_7293

theorem work_completion_time 
  (total_men : ℕ) 
  (initial_days : ℕ) 
  (absent_men : ℕ) 
  (h1 : total_men = 15)
  (h2 : initial_days = 8)
  (h3 : absent_men = 3) : 
  (total_men * initial_days) / (total_men - absent_men) = 10 := by
sorry

end work_completion_time_l72_7293


namespace night_crew_ratio_l72_7239

theorem night_crew_ratio (D N : ℕ) (B : ℝ) (h1 : D > 0) (h2 : N > 0) (h3 : B > 0) :
  (D * B) / ((D * B) + (N * (B / 2))) = 5 / 7 →
  (N : ℝ) / D = 4 / 5 := by
sorry

end night_crew_ratio_l72_7239


namespace range_of_a_l72_7281

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a*x > 0) → 
  a < 1 := by
sorry

end range_of_a_l72_7281


namespace max_projection_area_is_one_l72_7243

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  /-- Two adjacent faces are isosceles right triangles -/
  adjacent_faces_isosceles_right : Bool
  /-- Hypotenuse of the isosceles right triangles is 2 -/
  hypotenuse : ℝ
  /-- Dihedral angle between the two adjacent faces is 60 degrees -/
  dihedral_angle : ℝ

/-- Calculates the maximum projection area of the rotating tetrahedron -/
def max_projection_area (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the maximum projection area is 1 -/
theorem max_projection_area_is_one (t : Tetrahedron) 
  (h1 : t.adjacent_faces_isosceles_right = true)
  (h2 : t.hypotenuse = 2)
  (h3 : t.dihedral_angle = π / 3) : 
  max_projection_area t = 1 := by
  sorry

end max_projection_area_is_one_l72_7243


namespace partial_fraction_decomposition_l72_7265

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ),
    ∀ (x : ℚ), x ≠ 2 → x ≠ 4 →
      (3 * x + 7) / ((x - 4) * (x - 2)^2) =
      A / (x - 4) + B / (x - 2) + C / (x - 2)^2 ∧
      A = 19 / 4 ∧ B = -19 / 4 ∧ C = -13 / 2 :=
by sorry

end partial_fraction_decomposition_l72_7265


namespace only_finance_opposite_meanings_l72_7216

-- Define a type for quantity pairs
inductive QuantityPair
  | Distance (d1 d2 : ℕ)
  | Finance (f1 f2 : ℤ)
  | HeightWeight (h w : ℚ)
  | Scores (s1 s2 : ℕ)

-- Define a function to check if a pair has opposite meanings
def hasOppositeMeanings (pair : QuantityPair) : Prop :=
  match pair with
  | QuantityPair.Finance f1 f2 => f1 * f2 < 0
  | _ => False

-- Theorem statement
theorem only_finance_opposite_meanings 
  (a : QuantityPair) 
  (b : QuantityPair) 
  (c : QuantityPair) 
  (d : QuantityPair) 
  (ha : a = QuantityPair.Distance 500 200)
  (hb : b = QuantityPair.Finance (-3000) 12000)
  (hc : c = QuantityPair.HeightWeight 1.5 (-2.4))
  (hd : d = QuantityPair.Scores 50 70) :
  hasOppositeMeanings b ∧ 
  ¬hasOppositeMeanings a ∧ 
  ¬hasOppositeMeanings c ∧ 
  ¬hasOppositeMeanings d := by
  sorry

end only_finance_opposite_meanings_l72_7216


namespace fraction_equals_decimal_l72_7231

theorem fraction_equals_decimal : (1 : ℚ) / 4 = 0.25 := by
  sorry

end fraction_equals_decimal_l72_7231


namespace circle_area_radius_increase_l72_7242

theorem circle_area_radius_increase : 
  ∀ (r : ℝ) (r' : ℝ), r > 0 → r' > 0 →
  (π * r' ^ 2 = 4 * π * r ^ 2) → 
  (r' - r) / r * 100 = 100 := by
sorry

end circle_area_radius_increase_l72_7242


namespace water_volume_for_spheres_in_cylinder_l72_7235

/-- The volume of water required to cover two spheres in a cylinder -/
theorem water_volume_for_spheres_in_cylinder (cylinder_diameter cylinder_height : ℝ)
  (small_sphere_radius large_sphere_radius : ℝ) :
  cylinder_diameter = 27 →
  cylinder_height = 30 →
  small_sphere_radius = 6 →
  large_sphere_radius = 9 →
  (π * (cylinder_diameter / 2)^2 * (large_sphere_radius + small_sphere_radius + large_sphere_radius)) -
  (4/3 * π * small_sphere_radius^3 + 4/3 * π * large_sphere_radius^3) = 3114 * π :=
by sorry

end water_volume_for_spheres_in_cylinder_l72_7235


namespace trajectory_and_point_existence_l72_7227

-- Define the plane and points
variable (x y : ℝ)
def F : ℝ × ℝ := (1, 0)
def S : ℝ × ℝ := (x, y)

-- Define the distance ratio condition
def distance_ratio (S : ℝ × ℝ) : Prop :=
  Real.sqrt ((S.1 - F.1)^2 + S.2^2) / |S.1 - 2| = Real.sqrt 2 / 2

-- Define the trajectory equation
def trajectory_equation (S : ℝ × ℝ) : Prop :=
  S.1^2 / 2 + S.2^2 = 1

-- Define the line l (not perpendicular to x-axis)
variable (k : ℝ)
def line_l (x : ℝ) : ℝ := k * (x - 1)

-- Define points P and Q on the intersection of line_l and trajectory
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define point M
variable (m : ℝ)
def M : ℝ × ℝ := (m, 0)

-- Define the dot product condition
def dot_product_condition (M P Q : ℝ × ℝ) : Prop :=
  let MP := (P.1 - M.1, P.2 - M.2)
  let MQ := (Q.1 - M.1, Q.2 - M.2)
  let PQ := (Q.1 - P.1, Q.2 - P.2)
  (MP.1 + MQ.1) * PQ.1 + (MP.2 + MQ.2) * PQ.2 = 0

-- Main theorem
theorem trajectory_and_point_existence :
  ∀ S, distance_ratio S →
    (trajectory_equation S ∧
     ∃ m, 0 ≤ m ∧ m < 1/2 ∧
       ∀ k ≠ 0, dot_product_condition (M m) P Q) := by sorry

end trajectory_and_point_existence_l72_7227


namespace committee_selection_l72_7297

theorem committee_selection (n : ℕ) : 
  (n.choose 3 = 20) → (n.choose 4 = 15) :=
by
  sorry

end committee_selection_l72_7297


namespace magnitude_of_z_l72_7210

open Complex

theorem magnitude_of_z : ∃ z : ℂ, z = 1 + 2*I + I^3 ∧ abs z = Real.sqrt 2 := by
  sorry

end magnitude_of_z_l72_7210


namespace common_difference_from_sum_condition_l72_7225

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n * (a 1 + a n)) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_from_sum_condition (seq : ArithmeticSequence) 
    (h : seq.S 4 / 12 - seq.S 3 / 9 = 1) : 
    common_difference seq = 6 := by
  sorry


end common_difference_from_sum_condition_l72_7225


namespace simple_interest_problem_l72_7263

/-- 
Given a principal amount P and an interest rate R, 
if increasing the rate by 3% for 4 years results in Rs. 120 more interest, 
then P = 1000.
-/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 3) * 4) / 100 - (P * R * 4) / 100 = 120 → P = 1000 := by
  sorry

end simple_interest_problem_l72_7263


namespace william_farm_tax_l72_7241

/-- Calculates an individual's farm tax payment given the total tax collected and their land percentage -/
def individual_farm_tax (total_tax : ℝ) (land_percentage : ℝ) : ℝ :=
  land_percentage * total_tax

/-- Proves that given the conditions, Mr. William's farm tax payment is $960 -/
theorem william_farm_tax :
  let total_tax : ℝ := 3840
  let william_land_percentage : ℝ := 0.25
  individual_farm_tax total_tax william_land_percentage = 960 := by
sorry

end william_farm_tax_l72_7241


namespace X_is_greatest_l72_7275

def X : ℚ := 2010/2009 + 2010/2011
def Y : ℚ := 2010/2011 + 2012/2011
def Z : ℚ := 2011/2010 + 2011/2012

theorem X_is_greatest : X > Y ∧ X > Z := by
  sorry

end X_is_greatest_l72_7275


namespace min_value_fraction_l72_7212

theorem min_value_fraction (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^2 + b^2) / (a*b - b^2) ≥ 2 + 2*Real.sqrt 2 ∧
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (a^2 + b^2) / (a*b - b^2) = 2 + 2*Real.sqrt 2 :=
by sorry

end min_value_fraction_l72_7212


namespace power_of_three_mod_seven_l72_7251

theorem power_of_three_mod_seven : 3^2023 % 7 = 3 := by sorry

end power_of_three_mod_seven_l72_7251


namespace camping_hike_distance_l72_7290

/-- Hiking distances on a camping trip -/
theorem camping_hike_distance 
  (total_distance : ℝ)
  (car_to_stream : ℝ)
  (stream_to_meadow : ℝ)
  (h_total : total_distance = 0.7)
  (h_car_stream : car_to_stream = 0.2)
  (h_stream_meadow : stream_to_meadow = 0.4) :
  total_distance - (car_to_stream + stream_to_meadow) = 0.1 := by
  sorry

end camping_hike_distance_l72_7290


namespace money_distribution_inconsistency_l72_7250

/-- Prove that the given conditions about money distribution are inconsistent -/
theorem money_distribution_inconsistency :
  ¬∃ (a b c : ℤ),
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧  -- Money amounts are non-negative
    a + c = 200 ∧            -- A and C together have 200
    b + c = 350 ∧            -- B and C together have 350
    c = 250                  -- C has 250
    := by sorry

end money_distribution_inconsistency_l72_7250


namespace sin_alpha_minus_nine_pi_halves_l72_7260

theorem sin_alpha_minus_nine_pi_halves (α : Real)
  (h1 : π / 2 < α)
  (h2 : α < π)
  (h3 : 3 * Real.sin (2 * α) = 2 * Real.cos α) :
  Real.sin (α - 9 * π / 2) = 2 * Real.sqrt 2 / 3 := by
  sorry

end sin_alpha_minus_nine_pi_halves_l72_7260


namespace helens_oranges_l72_7299

/-- Helen's orange counting problem -/
theorem helens_oranges (initial : ℕ) (from_ann : ℕ) (to_sarah : ℕ) : 
  initial = 9 → from_ann = 29 → to_sarah = 14 → 
  initial + from_ann - to_sarah = 24 :=
by
  sorry

end helens_oranges_l72_7299


namespace geometric_progression_constant_l72_7284

theorem geometric_progression_constant (x : ℝ) : 
  (((30 + x) ^ 2 = (10 + x) * (90 + x)) ↔ x = 0) ∧
  (∀ y : ℝ, ((30 + y) ^ 2 = (10 + y) * (90 + y)) → y = 0) :=
by sorry

end geometric_progression_constant_l72_7284


namespace triangle_angle_adjustment_l72_7234

/-- 
Given a triangle with interior angles in a 3:4:9 ratio, prove that if the largest angle is 
decreased by x degrees such that the smallest angle doubles its initial value while 
maintaining the sum of angles as 180 degrees, then x = 33.75 degrees.
-/
theorem triangle_angle_adjustment (k : ℝ) (x : ℝ) 
  (h1 : 3*k + 4*k + 9*k = 180)  -- Sum of initial angles is 180 degrees
  (h2 : 3*k + 4*k + (9*k - x) = 180)  -- Sum of angles after adjustment is 180 degrees
  (h3 : 2*(3*k) = 3*k + 4*k)  -- Smallest angle doubles its initial value
  : x = 33.75 := by sorry

end triangle_angle_adjustment_l72_7234


namespace local_minimum_condition_l72_7228

/-- The function f(x) = x(x - m)^2 attains a local minimum at x = 1 -/
theorem local_minimum_condition (m : ℝ) :
  let f : ℝ → ℝ := λ x => x * (x - m)^2
  (∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1) →
  m = 1 := by
  sorry

end local_minimum_condition_l72_7228


namespace arithmetic_sequence_sum_l72_7233

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 6 = 30 → a 3 + a 9 = 60 := by
  sorry

end arithmetic_sequence_sum_l72_7233


namespace skyscraper_anniversary_l72_7298

/-- Calculates the number of years in the future when it will be 5 years before the 200th anniversary of a skyscraper built 100 years ago. -/
theorem skyscraper_anniversary (years_since_built : ℕ) (years_to_anniversary : ℕ) (years_before_anniversary : ℕ) : 
  years_since_built = 100 →
  years_to_anniversary = 200 →
  years_before_anniversary = 5 →
  years_to_anniversary - years_before_anniversary - years_since_built = 95 :=
by sorry

end skyscraper_anniversary_l72_7298


namespace jason_total_games_l72_7292

/-- The total number of games Jason will attend over three months -/
def total_games (this_month last_month next_month : ℕ) : ℕ :=
  this_month + last_month + next_month

/-- Theorem stating the total number of games Jason will attend -/
theorem jason_total_games : 
  total_games 11 17 16 = 44 := by
  sorry

end jason_total_games_l72_7292


namespace no_integer_solutions_for_3a2_eq_b2_plus_1_l72_7246

theorem no_integer_solutions_for_3a2_eq_b2_plus_1 :
  ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end no_integer_solutions_for_3a2_eq_b2_plus_1_l72_7246


namespace numbers_statistics_l72_7204

def numbers : List ℝ := [158, 149, 155, 157, 156, 162, 155, 168]

def median (xs : List ℝ) : ℝ := sorry

def mean (xs : List ℝ) : ℝ := sorry

def mode (xs : List ℝ) : ℝ := sorry

theorem numbers_statistics :
  median numbers = 155.5 ∧
  mean numbers = 157.5 ∧
  mode numbers = 155 := by sorry

end numbers_statistics_l72_7204


namespace sum_of_fourth_powers_l72_7294

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  a^4 + b^4 + c^4 = 1/2 := by
  sorry

end sum_of_fourth_powers_l72_7294


namespace money_lending_problem_l72_7221

/-- Given a sum of money divided into two parts where:
    1. The interest on the first part for 8 years at 3% per annum is equal to
       the interest on the second part for 3 years at 5% per annum.
    2. The second part is Rs. 1656.
    Prove that the total sum lent is Rs. 2691. -/
theorem money_lending_problem (first_part second_part total_sum : ℚ) : 
  second_part = 1656 →
  (first_part * 3 / 100 * 8 = second_part * 5 / 100 * 3) →
  total_sum = first_part + second_part →
  total_sum = 2691 := by
  sorry

#check money_lending_problem

end money_lending_problem_l72_7221


namespace tom_teaching_years_l72_7256

theorem tom_teaching_years :
  ∀ (tom_years devin_years : ℕ),
    tom_years + devin_years = 70 →
    devin_years = tom_years / 2 - 5 →
    tom_years = 50 :=
by
  sorry

end tom_teaching_years_l72_7256


namespace batsman_second_set_matches_l72_7254

/-- Given information about a batsman's performance, prove the number of matches in the second set -/
theorem batsman_second_set_matches 
  (first_set_matches : ℕ) 
  (total_matches : ℕ) 
  (first_set_average : ℝ) 
  (second_set_average : ℝ) 
  (total_average : ℝ) 
  (h1 : first_set_matches = 35)
  (h2 : total_matches = 49)
  (h3 : first_set_average = 36)
  (h4 : second_set_average = 15)
  (h5 : total_average = 30) :
  total_matches - first_set_matches = 14 := by
  sorry

#check batsman_second_set_matches

end batsman_second_set_matches_l72_7254


namespace cycling_route_length_l72_7217

theorem cycling_route_length (upper_segments : List ℝ) (left_segments : List ℝ) :
  upper_segments = [4, 7, 2] →
  left_segments = [6, 7] →
  2 * (upper_segments.sum + left_segments.sum) = 52 := by
  sorry

end cycling_route_length_l72_7217


namespace ratio_equivalence_l72_7288

theorem ratio_equivalence (x y m n : ℚ) 
  (h : (5 * x + 7 * y) / (3 * x + 2 * y) = m / n) :
  (13 * x + 16 * y) / (2 * x + 5 * y) = (2 * m + n) / (m - n) := by
  sorry

end ratio_equivalence_l72_7288


namespace thomas_monthly_pay_l72_7269

/-- The amount paid to a worker after one month, given their weekly rate and the number of weeks in a month -/
def monthly_pay (weekly_rate : ℕ) (weeks_per_month : ℕ) : ℕ :=
  weekly_rate * weeks_per_month

theorem thomas_monthly_pay :
  monthly_pay 4550 4 = 18200 := by
  sorry

end thomas_monthly_pay_l72_7269


namespace parenthesization_pigeonhole_l72_7220

theorem parenthesization_pigeonhole : ∃ (n : ℕ) (k : ℕ), 
  n > 0 ∧ 
  k > 0 ∧ 
  (2 ^ n > (k * (k + 1))) ∧ 
  (∀ (f : Fin (2^n) → ℤ), ∃ (i j : Fin (2^n)), i ≠ j ∧ f i = f j) := by
  sorry

end parenthesization_pigeonhole_l72_7220


namespace space_for_another_circle_l72_7230

/-- The side length of the large square N -/
def N : ℝ := 6

/-- The side length of the small squares -/
def small_square_side : ℝ := 1

/-- The diameter of the circles -/
def circle_diameter : ℝ := 1

/-- The number of small squares -/
def num_squares : ℕ := 4

/-- The number of circles -/
def num_circles : ℕ := 3

/-- The theorem stating that there is space for another circle -/
theorem space_for_another_circle :
  (N - 1)^2 - (num_squares * (small_square_side^2 + small_square_side * circle_diameter + Real.pi * (circle_diameter / 2)^2) +
   num_circles * Real.pi * (circle_diameter / 2)^2) > 0 := by
  sorry

end space_for_another_circle_l72_7230


namespace arithmetic_operations_l72_7267

theorem arithmetic_operations : 
  (12 - (-18) + (-7) + (-15) = 8) ∧ 
  ((-1)^7 * 2 + (-3)^2 / 9 = -1) := by sorry

end arithmetic_operations_l72_7267


namespace number_of_teams_l72_7264

theorem number_of_teams (n : ℕ) (k : ℕ) : n = 10 → k = 5 → Nat.choose n k = 252 := by
  sorry

end number_of_teams_l72_7264


namespace basketball_tournament_matches_l72_7255

/-- The number of matches in a round-robin tournament with n teams -/
def roundRobinMatches (n : ℕ) : ℕ := n.choose 2

/-- The total number of matches played in the basketball tournament -/
def totalMatches (groups numTeams : ℕ) : ℕ :=
  groups * roundRobinMatches numTeams + roundRobinMatches groups

theorem basketball_tournament_matches :
  totalMatches 5 6 = 85 := by
  sorry

end basketball_tournament_matches_l72_7255


namespace isosceles_triangle_side_ratio_l72_7262

theorem isosceles_triangle_side_ratio (a b : ℝ) (h_isosceles : b > 0) (h_vertex_angle : Real.cos (20 * π / 180) = a / (2 * b)) : 2 < b / a ∧ b / a < 3 := by
  sorry

end isosceles_triangle_side_ratio_l72_7262


namespace tournament_has_cycle_of_length_3_l72_7215

/-- A tournament is a complete directed graph where each edge represents a match outcome. -/
def Tournament (n : ℕ) := Fin n → Fin n → Prop

/-- In a valid tournament, every pair of distinct players has exactly one match outcome. -/
def is_valid_tournament (t : Tournament n) : Prop :=
  ∀ i j : Fin n, i ≠ j → (t i j ∧ ¬t j i) ∨ (t j i ∧ ¬t i j)

/-- A player wins at least one match if there exists another player they defeated. -/
def player_wins_at_least_one (t : Tournament n) (i : Fin n) : Prop :=
  ∃ j : Fin n, t i j

/-- A cycle of length 3 in a tournament. -/
def has_cycle_of_length_3 (t : Tournament n) : Prop :=
  ∃ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ t a b ∧ t b c ∧ t c a

theorem tournament_has_cycle_of_length_3 :
  ∀ (t : Tournament 12),
    is_valid_tournament t →
    (∀ i : Fin 12, player_wins_at_least_one t i) →
    has_cycle_of_length_3 t :=
by sorry


end tournament_has_cycle_of_length_3_l72_7215


namespace square_sum_eq_two_l72_7259

theorem square_sum_eq_two (a b : ℝ) : (a^2 + b^2)^4 - 8*(a^2 + b^2)^2 + 16 = 0 → a^2 + b^2 = 2 := by
  sorry

end square_sum_eq_two_l72_7259


namespace expand_expression_l72_7291

theorem expand_expression (x y z : ℝ) : 
  (x + 12) * (3 * y + 2 * z + 15) = 3 * x * y + 2 * x * z + 15 * x + 36 * y + 24 * z + 180 := by
  sorry

end expand_expression_l72_7291


namespace birthday_250_years_ago_l72_7266

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the day of the week that is n days before the given day -/
def daysBefore (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
  sorry

/-- Calculates the number of leap years in a 250-year period, excluding certain century years -/
def leapYearsIn250Years : ℕ :=
  sorry

/-- Represents the number of days to go backwards for 250 years -/
def daysBackFor250Years : ℕ :=
  sorry

theorem birthday_250_years_ago (anniversary_day : DayOfWeek) : 
  anniversary_day = DayOfWeek.Tuesday → 
  daysBefore anniversary_day daysBackFor250Years = DayOfWeek.Saturday :=
sorry

end birthday_250_years_ago_l72_7266


namespace negation_of_existence_squared_nonpositive_l72_7273

theorem negation_of_existence_squared_nonpositive :
  (¬ ∃ x : ℝ, x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 > 0) := by
  sorry

end negation_of_existence_squared_nonpositive_l72_7273


namespace square_difference_equality_l72_7240

theorem square_difference_equality : 1005^2 - 995^2 - 1003^2 + 997^2 = 8000 := by
  sorry

end square_difference_equality_l72_7240


namespace floor_ceiling_sum_l72_7224

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by sorry

end floor_ceiling_sum_l72_7224


namespace complex_equation_solution_l72_7277

theorem complex_equation_solution (i : ℂ) (z : ℂ) 
  (h1 : i * i = -1)
  (h2 : z * (1 - i) = 3 + 2 * i) :
  z = 1/2 + 5/2 * i := by
  sorry

end complex_equation_solution_l72_7277


namespace sum_of_possible_e_values_l72_7272

theorem sum_of_possible_e_values : 
  ∃ (e₁ e₂ : ℝ), (2 * |2 - e₁| = 5) ∧ (2 * |2 - e₂| = 5) ∧ (e₁ + e₂ = 4) := by
  sorry

end sum_of_possible_e_values_l72_7272


namespace largest_s_value_l72_7219

theorem largest_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) : 
  (r - 2) * s * 61 = (s - 2) * r * 60 → s ≤ 121 ∧ ∃ (r' : ℕ), r' ≥ 121 ∧ (r' - 2) * 121 * 61 = 119 * r' * 60 :=
sorry

end largest_s_value_l72_7219


namespace set_intersection_problem_l72_7252

theorem set_intersection_problem (M N P : Set Nat) 
  (hM : M = {1})
  (hN : N = {1, 2})
  (hP : P = {1, 2, 3}) :
  (M ∪ N) ∩ P = {1, 2, 3} := by
  sorry

end set_intersection_problem_l72_7252


namespace consecutive_odd_integers_sum_l72_7268

theorem consecutive_odd_integers_sum (n : ℤ) : 
  (∃ (a b c : ℤ), 
    (a = n - 2 ∧ b = n ∧ c = n + 2) ∧  -- Three consecutive odd integers
    (Odd a ∧ Odd b ∧ Odd c) ∧           -- All are odd
    (a + c = 152)) →                    -- Sum of first and third is 152
  n = 76 :=                             -- Second integer is 76
by sorry

end consecutive_odd_integers_sum_l72_7268


namespace seventh_rack_dvd_count_l72_7258

/-- Calculates the number of DVDs on a given rack based on the previous two racks -/
def dvd_count (n : ℕ) : ℕ :=
  match n with
  | 0 => 3  -- First rack
  | 1 => 4  -- Second rack
  | n + 2 => ((dvd_count (n + 1) - dvd_count n) * 2) + dvd_count (n + 1)

/-- The number of DVDs on the seventh rack is 66 -/
theorem seventh_rack_dvd_count :
  dvd_count 6 = 66 := by sorry

end seventh_rack_dvd_count_l72_7258


namespace unique_four_digit_square_divisible_by_11_ending_in_1_l72_7201

theorem unique_four_digit_square_divisible_by_11_ending_in_1 :
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ ∃ k : ℕ, n = k^2 ∧ n % 11 = 0 ∧ n % 10 = 1 :=
by
  -- The proof would go here
  sorry

end unique_four_digit_square_divisible_by_11_ending_in_1_l72_7201


namespace x_plus_y_value_l72_7207

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 2010)
  (eq2 : x + 2010 * Real.sin y = 2011)
  (y_range : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2011 + Real.pi := by
  sorry

end x_plus_y_value_l72_7207


namespace pond_diameter_l72_7237

/-- The diameter of a circular pond given specific conditions -/
theorem pond_diameter : ∃ (h k r : ℝ),
  (4 - h)^2 + (11 - k)^2 = r^2 ∧
  (12 - h)^2 + (9 - k)^2 = r^2 ∧
  (2 - h)^2 + (7 - k)^2 = (r - 1)^2 ∧
  2 * r = 9.2 := by
  sorry

end pond_diameter_l72_7237


namespace thirtieth_digit_of_sum_l72_7236

-- Define the fractions
def f1 : ℚ := 1 / 13
def f2 : ℚ := 1 / 11

-- Define the sum of the fractions
def sum : ℚ := f1 + f2

-- Define a function to get the nth digit after the decimal point
noncomputable def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem thirtieth_digit_of_sum : nthDigitAfterDecimal sum 30 = 9 := by sorry

end thirtieth_digit_of_sum_l72_7236


namespace birds_joining_fence_l72_7200

/-- Proves that 2 additional birds joined the fence given the initial and final conditions -/
theorem birds_joining_fence :
  let initial_birds : ℕ := 3
  let initial_storks : ℕ := 4
  let additional_birds : ℕ := 2
  let final_birds : ℕ := initial_birds + additional_birds
  let final_storks : ℕ := initial_storks
  final_birds = final_storks + 1 :=
by sorry

end birds_joining_fence_l72_7200


namespace head_start_value_l72_7283

/-- A race between two runners A and B -/
structure Race where
  length : ℝ
  speed_ratio : ℝ
  head_start : ℝ

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  r.length = 100 ∧ r.speed_ratio = 2 ∧ r.head_start > 0

/-- Both runners finish at the same time -/
def equal_finish_time (r : Race) : Prop :=
  r.length / r.speed_ratio = (r.length - r.head_start) / 1

theorem head_start_value (r : Race) 
  (h1 : race_conditions r) 
  (h2 : equal_finish_time r) : 
  r.head_start = 50 := by
  sorry

#check head_start_value

end head_start_value_l72_7283


namespace complex_addition_result_l72_7286

theorem complex_addition_result : ∃ z : ℂ, (5 - 3*I + z = -4 + 9*I) ∧ (z = -9 + 12*I) := by
  sorry

end complex_addition_result_l72_7286


namespace break_even_point_l72_7245

def parts_cost : ℕ := 3600
def patent_cost : ℕ := 4500
def variable_cost : ℕ := 25
def marketing_cost : ℕ := 2000
def selling_price : ℕ := 180

def total_fixed_cost : ℕ := parts_cost + patent_cost + marketing_cost
def contribution_margin : ℕ := selling_price - variable_cost

def break_even (n : ℕ) : Prop :=
  n * selling_price ≥ total_fixed_cost + n * variable_cost

theorem break_even_point : 
  ∀ m : ℕ, break_even m → m ≥ 66 :=
by sorry

end break_even_point_l72_7245


namespace fifth_term_of_geometric_sequence_l72_7285

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : is_positive_geometric_sequence a)
  (h_2 : a 2 = 3)
  (h_8 : a 8 = 27) :
  a 5 = 9 :=
sorry

end fifth_term_of_geometric_sequence_l72_7285


namespace johns_quilt_cost_l72_7226

/-- The cost of a rectangular quilt -/
def quilt_cost (length width price_per_sqft : ℝ) : ℝ :=
  length * width * price_per_sqft

/-- Theorem: The cost of John's quilt is $2240 -/
theorem johns_quilt_cost :
  quilt_cost 7 8 40 = 2240 := by
  sorry

end johns_quilt_cost_l72_7226


namespace chinese_sturgeon_probability_l72_7208

theorem chinese_sturgeon_probability (p_maturity p_spawn_reproduce : ℝ) 
  (h_maturity : p_maturity = 0.15)
  (h_spawn_reproduce : p_spawn_reproduce = 0.05) :
  p_spawn_reproduce / p_maturity = 1/3 := by
  sorry

end chinese_sturgeon_probability_l72_7208


namespace square_divisibility_l72_7209

theorem square_divisibility (n d : ℕ+) : 
  (n.val % d.val = 0) → 
  ((n.val^2 + d.val^2) % (d.val^2 * n.val + 1) = 0) → 
  n = d^2 := by
sorry

end square_divisibility_l72_7209


namespace range_of_function_l72_7289

theorem range_of_function (y : ℝ) : 
  (∃ x : ℝ, y = x / (1 + x^2)) ↔ -1/2 ≤ y ∧ y ≤ 1/2 := by
sorry

end range_of_function_l72_7289


namespace division_of_monomials_l72_7202

-- Define variables
variable (x y : ℝ)

-- Define the theorem
theorem division_of_monomials (x y : ℝ) :
  x ≠ 0 → y ≠ 0 → (-4 * x^5 * y^3) / (2 * x^3 * y) = -2 * x^2 * y^2 := by
  sorry

end division_of_monomials_l72_7202


namespace stationery_store_bundles_l72_7270

/-- Given the number of red and blue sheets of paper and the number of sheets per bundle,
    calculates the maximum number of complete bundles that can be made. -/
def max_bundles (red_sheets blue_sheets sheets_per_bundle : ℕ) : ℕ :=
  (red_sheets + blue_sheets) / sheets_per_bundle

/-- Proves that with 210 red sheets, 473 blue sheets, and 100 sheets per bundle,
    the maximum number of complete bundles is 6. -/
theorem stationery_store_bundles :
  max_bundles 210 473 100 = 6 := by
  sorry

#eval max_bundles 210 473 100

end stationery_store_bundles_l72_7270


namespace intersection_A_B_l72_7261

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B : Set ℝ := {x | |x - 2| < 2}

theorem intersection_A_B : ∀ x : ℝ, x ∈ (A ∩ B) ↔ 0 < x ∧ x ≤ 3 := by
  sorry

end intersection_A_B_l72_7261


namespace x_values_l72_7278

theorem x_values (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 6 * k)
  (h2 : x^2 > 144)
  (h3 : x < 30) :
  x = 18 ∨ x = 24 :=
by sorry

end x_values_l72_7278


namespace suji_age_is_16_l72_7232

/-- Represents the ages of Abi, Suji, and Ravi -/
structure Ages where
  x : ℕ
  deriving Repr

def Ages.abi (a : Ages) : ℕ := 5 * a.x
def Ages.suji (a : Ages) : ℕ := 4 * a.x
def Ages.ravi (a : Ages) : ℕ := 3 * a.x

def Ages.future_abi (a : Ages) : ℕ := a.abi + 6
def Ages.future_suji (a : Ages) : ℕ := a.suji + 6
def Ages.future_ravi (a : Ages) : ℕ := a.ravi + 6

/-- The theorem stating that Suji's present age is 16 years -/
theorem suji_age_is_16 (a : Ages) : 
  (a.future_abi / a.future_suji = 13 / 11) ∧ 
  (a.future_suji / a.future_ravi = 11 / 9) → 
  a.suji = 16 := by
  sorry

#eval Ages.suji { x := 4 }

end suji_age_is_16_l72_7232


namespace last_digit_base4_last_digit_390_base4_l72_7206

/-- Convert a natural number to its base-4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- The last digit of a number in base-4 is the same as the remainder when divided by 4 -/
theorem last_digit_base4 (n : ℕ) : 
  (toBase4 n).getLast? = some (n % 4) :=
sorry

/-- The last digit of 390 in base-4 is 2 -/
theorem last_digit_390_base4 : 
  (toBase4 390).getLast? = some 2 :=
sorry

end last_digit_base4_last_digit_390_base4_l72_7206


namespace max_inequality_constant_l72_7238

theorem max_inequality_constant : ∃ (M : ℝ), (∀ (x y : ℝ), x + y ≥ 0 → 
  (x^2 + y^2)^3 ≥ M * (x^3 + y^3) * (x*y - x - y)) ∧ 
  (∀ (M' : ℝ), (∀ (x y : ℝ), x + y ≥ 0 → 
    (x^2 + y^2)^3 ≥ M' * (x^3 + y^3) * (x*y - x - y)) → M' ≤ M) ∧
  M = 32 :=
by sorry

end max_inequality_constant_l72_7238


namespace largest_integral_x_l72_7249

theorem largest_integral_x : ∃ x : ℤ, x = 4 ∧ 
  (∀ y : ℤ, (1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 7/9 → y ≤ x) :=
by sorry

end largest_integral_x_l72_7249


namespace arithmetic_sequence_sum_l72_7205

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4 + a 7 = 19) →
  (a 3 + 5 * a 6 = 57) := by
  sorry

end arithmetic_sequence_sum_l72_7205


namespace line_perp_plane_parallel_plane_implies_planes_perp_planes_perp_parallel_implies_perp_l72_7211

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (linePerpendicular : Line → Plane → Prop)
variable (lineParallel : Line → Plane → Prop)

-- Theorem 1
theorem line_perp_plane_parallel_plane_implies_planes_perp
  (α β : Plane) (l : Line)
  (h1 : linePerpendicular l α)
  (h2 : lineParallel l β) :
  perpendicular α β :=
sorry

-- Theorem 2
theorem planes_perp_parallel_implies_perp
  (α β γ : Plane)
  (h1 : perpendicular α β)
  (h2 : parallel α γ) :
  perpendicular γ β :=
sorry

end line_perp_plane_parallel_plane_implies_planes_perp_planes_perp_parallel_implies_perp_l72_7211


namespace equation_one_solutions_equation_two_solutions_l72_7295

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  (x - 1)^2 = 2 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := by
sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  x^2 - 6*x - 7 = 0 ↔ x = -1 ∨ x = 7 := by
sorry

end equation_one_solutions_equation_two_solutions_l72_7295


namespace am_length_l72_7248

/-- Given points M, A, and B on a straight line, with AM twice as long as BM and AB = 6,
    the length of AM is either 4 or 12. -/
theorem am_length (M A B : ℝ) : 
  (∃ t : ℝ, M = t * A + (1 - t) * B) →  -- M, A, B are collinear
  abs (A - M) = 2 * abs (B - M) →       -- AM is twice as long as BM
  abs (A - B) = 6 →                     -- AB = 6
  abs (A - M) = 4 ∨ abs (A - M) = 12 := by
sorry


end am_length_l72_7248


namespace laptop_down_payment_percentage_l72_7213

theorem laptop_down_payment_percentage
  (laptop_cost : ℝ)
  (monthly_installment : ℝ)
  (additional_down_payment : ℝ)
  (balance_after_four_months : ℝ)
  (h1 : laptop_cost = 1000)
  (h2 : monthly_installment = 65)
  (h3 : additional_down_payment = 20)
  (h4 : balance_after_four_months = 520) :
  let down_payment_percentage := 100 * (laptop_cost - balance_after_four_months - 4 * monthly_installment - additional_down_payment) / laptop_cost
  down_payment_percentage = 20 := by
sorry

end laptop_down_payment_percentage_l72_7213


namespace presidency_meeting_arrangements_count_l72_7247

/- Define the number of schools -/
def num_schools : ℕ := 3

/- Define the number of members per school -/
def members_per_school : ℕ := 6

/- Define the number of representatives from the host school -/
def host_representatives : ℕ := 3

/- Define the number of representatives from each non-host school -/
def non_host_representatives : ℕ := 1

/- Function to calculate the number of ways to arrange the meeting -/
def presidency_meeting_arrangements : ℕ :=
  num_schools * (members_per_school.choose host_representatives) * 
  (members_per_school.choose non_host_representatives) * 
  (members_per_school.choose non_host_representatives)

/- Theorem stating the number of arrangements -/
theorem presidency_meeting_arrangements_count :
  presidency_meeting_arrangements = 2160 := by
  sorry

end presidency_meeting_arrangements_count_l72_7247
