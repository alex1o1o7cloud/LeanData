import Mathlib

namespace positive_integer_sum_greater_than_product_l710_71073

theorem positive_integer_sum_greater_than_product (a b : ℕ+) :
  a + b > a * b ↔ a = 1 ∨ b = 1 := by sorry

end positive_integer_sum_greater_than_product_l710_71073


namespace sum_of_three_consecutive_even_numbers_l710_71057

theorem sum_of_three_consecutive_even_numbers (n : ℕ) (h : n = 52) :
  n + (n + 2) + (n + 4) = 162 := by
sorry

end sum_of_three_consecutive_even_numbers_l710_71057


namespace set_equals_naturals_l710_71006

def is_closed_under_multiplication_by_four (X : Set ℕ) : Prop :=
  ∀ x ∈ X, (4 * x) ∈ X

def is_closed_under_floor_sqrt (X : Set ℕ) : Prop :=
  ∀ x ∈ X, Nat.sqrt x ∈ X

theorem set_equals_naturals (X : Set ℕ) 
  (h_nonempty : X.Nonempty)
  (h_mul_four : is_closed_under_multiplication_by_four X)
  (h_floor_sqrt : is_closed_under_floor_sqrt X) : 
  X = Set.univ :=
sorry

end set_equals_naturals_l710_71006


namespace derivative_f_at_negative_two_l710_71032

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem derivative_f_at_negative_two :
  deriv f (-2) = 0 := by sorry

end derivative_f_at_negative_two_l710_71032


namespace tangent_line_point_on_circle_l710_71093

/-- Given a line ax + by - 1 = 0 tangent to the circle x² + y² = 1,
    prove that the point P(a, b) lies on the circle. -/
theorem tangent_line_point_on_circle (a b : ℝ) :
  (∀ x y, x^2 + y^2 = 1 → (a*x + b*y = 1)) →  -- Line is tangent to circle
  a^2 + b^2 = 1  -- Point P(a, b) is on the circle
  := by sorry

end tangent_line_point_on_circle_l710_71093


namespace intersection_equals_N_l710_71036

def M : Set ℝ := {x | x < 2011}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_equals_N : M ∩ N = N := by sorry

end intersection_equals_N_l710_71036


namespace pushup_difference_l710_71098

-- Define the number of push-ups for each person
def zachary_pushups : ℕ := 51
def john_pushups : ℕ := 69

-- Define David's push-ups in terms of Zachary's
def david_pushups : ℕ := zachary_pushups + 22

-- Theorem to prove
theorem pushup_difference : david_pushups - john_pushups = 4 := by
  sorry

end pushup_difference_l710_71098


namespace cats_remaining_after_sale_l710_71017

theorem cats_remaining_after_sale (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 13 → house = 5 → sold = 10 → siamese + house - sold = 8 := by
sorry

end cats_remaining_after_sale_l710_71017


namespace calculation_proof_l710_71003

theorem calculation_proof :
  (1 / 6 + 2 / 3) * (-24) = -20 ∧
  (-3)^2 * (2 - (-6)) + 30 / (-5) = 66 := by
sorry

end calculation_proof_l710_71003


namespace sum_of_digits_up_to_2023_l710_71054

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := 
  (List.range n).map (λ i => sumOfDigits (i + 1)) |>.sum

/-- The sum of digits of all numbers from 1 to 2023 is 27314 -/
theorem sum_of_digits_up_to_2023 : sumOfDigitsUpTo 2023 = 27314 := by sorry

end sum_of_digits_up_to_2023_l710_71054


namespace franks_trivia_score_l710_71081

/-- Frank's trivia game score calculation -/
theorem franks_trivia_score :
  ∀ (first_half second_half points_per_question : ℕ),
    first_half = 3 →
    second_half = 2 →
    points_per_question = 3 →
    (first_half + second_half) * points_per_question = 15 :=
by
  sorry

end franks_trivia_score_l710_71081


namespace problem_statement_l710_71050

theorem problem_statement : 
  (∀ x : ℝ, x^2 + x + 1 > 0) ∧ (∃ x : ℝ, x^3 = 1 - x^2) := by
  sorry

end problem_statement_l710_71050


namespace arcsin_neg_sqrt2_over_2_l710_71046

theorem arcsin_neg_sqrt2_over_2 : Real.arcsin (-Real.sqrt 2 / 2) = -π / 4 := by sorry

end arcsin_neg_sqrt2_over_2_l710_71046


namespace louisa_first_day_travel_l710_71097

/-- Represents Louisa's travel details -/
structure LouisaTravel where
  first_day_miles : ℝ
  second_day_miles : ℝ
  average_speed : ℝ
  time_difference : ℝ

/-- Theorem stating that Louisa traveled 200 miles on the first day -/
theorem louisa_first_day_travel (t : LouisaTravel) 
  (h1 : t.second_day_miles = 350)
  (h2 : t.average_speed = 50)
  (h3 : t.time_difference = 3)
  (h4 : t.second_day_miles / t.average_speed = t.first_day_miles / t.average_speed + t.time_difference) :
  t.first_day_miles = 200 := by
  sorry

#check louisa_first_day_travel

end louisa_first_day_travel_l710_71097


namespace circle_area_ratio_after_radius_increase_l710_71096

theorem circle_area_ratio_after_radius_increase (r : ℝ) (h : r > 0) : 
  (π * r^2) / (π * (1.5 * r)^2) = 4/9 := by sorry

end circle_area_ratio_after_radius_increase_l710_71096


namespace fifth_term_equals_fourth_l710_71024

/-- A geometric sequence of positive integers -/
structure GeometricSequence where
  a : ℕ+  -- first term
  r : ℕ+  -- common ratio

/-- The nth term of a geometric sequence -/
def nthTerm (seq : GeometricSequence) (n : ℕ) : ℕ+ :=
  seq.a * (seq.r ^ (n - 1))

theorem fifth_term_equals_fourth (seq : GeometricSequence) 
  (h1 : seq.a = 4)
  (h2 : nthTerm seq 4 = 324) :
  nthTerm seq 5 = 324 := by
  sorry

end fifth_term_equals_fourth_l710_71024


namespace pseudo_symmetry_point_l710_71009

noncomputable def f (x : ℝ) : ℝ := x^2 - 6*x + 4*Real.log x

noncomputable def g (x₀ x : ℝ) : ℝ := 
  (2*x₀ + 4/x₀ - 6)*(x - x₀) + x₀^2 - 6*x₀ + 4*Real.log x₀

theorem pseudo_symmetry_point :
  ∃! x₀ : ℝ, x₀ > 0 ∧ 
  ∀ x, x > 0 → x ≠ x₀ → (f x - g x₀ x) / (x - x₀) > 0 :=
sorry

end pseudo_symmetry_point_l710_71009


namespace intersection_volume_is_half_l710_71011

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ

/-- The intersection of two regular tetrahedra -/
def tetrahedra_intersection (t1 t2 : RegularTetrahedron) : ℝ := sorry

/-- Reflection of a regular tetrahedron through its center -/
def reflect_tetrahedron (t : RegularTetrahedron) : RegularTetrahedron := sorry

theorem intersection_volume_is_half (t : RegularTetrahedron) 
  (h : t.volume = 1) : 
  tetrahedra_intersection t (reflect_tetrahedron t) = 1/2 := by sorry

end intersection_volume_is_half_l710_71011


namespace intersection_complement_A_and_B_range_of_a_for_C_subset_A_l710_71085

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 2*x < 0}
def B : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1)}

-- Define the set C
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 1}

-- Statement 1
theorem intersection_complement_A_and_B :
  (Set.compl A ∩ B) = {x : ℝ | x ≥ 0} := by sorry

-- Statement 2
theorem range_of_a_for_C_subset_A :
  {a : ℝ | C a ⊆ A} = {a : ℝ | a ≤ -1/2} := by sorry

end intersection_complement_A_and_B_range_of_a_for_C_subset_A_l710_71085


namespace initial_children_meals_l710_71045

/-- Calculates the number of meals initially available for children given the total adult meals and remaining meals after some adults eat. -/
def children_meals (total_adult_meals : ℕ) (adults_eaten : ℕ) (remaining_child_meals : ℕ) : ℕ :=
  (total_adult_meals * remaining_child_meals) / (total_adult_meals - adults_eaten)

/-- Proves that the number of meals initially available for children is 90. -/
theorem initial_children_meals :
  children_meals 70 14 72 = 90 := by
  sorry

end initial_children_meals_l710_71045


namespace inequality_condition_max_area_ellipse_l710_71022

-- Define the line l: y = k(x+1)
def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

-- Define the ellipse: x^2 + 4y^2 = a^2
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 + 4*y^2 = a^2

-- Define the intersection points A and B
def intersection_points (k a : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
    ellipse a x1 y1 ∧ y1 = line_l k x1 ∧
    ellipse a x2 y2 ∧ y2 = line_l k x2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2)

-- Define point C as the intersection of line l with x-axis
def point_c (k : ℝ) : ℝ := -1

-- Define the condition AC = 2CB
def ac_twice_cb (k a : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
    ellipse a x1 y1 ∧ y1 = line_l k x1 ∧
    ellipse a x2 y2 ∧ y2 = line_l k x2 ∧
    (x1 - point_c k) = 2 * (point_c k - x2)

-- Theorem 1: a^2 > 4k^2 / (1+k^2)
theorem inequality_condition (k a : ℝ) (h1 : a > 0) (h2 : intersection_points k a) :
  a^2 > 4*k^2 / (1 + k^2) := by sorry

-- Theorem 2: When the area of triangle OAB is maximized, the equation of the ellipse is x^2 + 4y^2 = 5
theorem max_area_ellipse (k a : ℝ) (h1 : a > 0) (h2 : intersection_points k a) (h3 : ac_twice_cb k a) :
  (∀ x y : ℝ, ellipse a x y ↔ x^2 + 4*y^2 = 5) := by sorry

end inequality_condition_max_area_ellipse_l710_71022


namespace division_value_problem_l710_71059

theorem division_value_problem (x : ℝ) : 
  ((7.5 / x) * 12 = 15) → x = 6 := by
  sorry

end division_value_problem_l710_71059


namespace swimmers_pass_178_times_l710_71040

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the swimming pool scenario --/
structure PoolScenario where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  totalTime : ℝ

/-- Calculates the number of times swimmers pass each other --/
def calculatePassings (scenario : PoolScenario) : ℕ :=
  sorry

/-- The specific pool scenario from the problem --/
def problemScenario : PoolScenario :=
  { poolLength := 100
    swimmer1 := { speed := 4, startPosition := 0 }
    swimmer2 := { speed := 3, startPosition := 100 }
    totalTime := 20 * 60 }  -- 20 minutes in seconds

theorem swimmers_pass_178_times :
  calculatePassings problemScenario = 178 :=
sorry

end swimmers_pass_178_times_l710_71040


namespace linear_function_expression_y_value_at_negative_four_l710_71018

/-- A linear function passing through two given points -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  point1 : k * 1 + b = 5
  point2 : k * (-1) + b = 1

/-- The unique linear function passing through (1, 5) and (-1, 1) -/
def uniqueLinearFunction : LinearFunction where
  k := 2
  b := 3
  point1 := by sorry
  point2 := by sorry

theorem linear_function_expression (f : LinearFunction) :
  f.k = 2 ∧ f.b = 3 := by sorry

theorem y_value_at_negative_four (f : LinearFunction) :
  f.k * (-4) + f.b = -5 := by sorry

end linear_function_expression_y_value_at_negative_four_l710_71018


namespace line_segment_parameterization_sum_of_squares_l710_71048

/-- Given a line segment connecting (-4, 10) and (2, -3), parameterized by x = at + b and y = ct + d
    where -1 ≤ t ≤ 1 and t = -1 corresponds to (-4, 10), prove that a^2 + b^2 + c^2 + d^2 = 321 -/
theorem line_segment_parameterization_sum_of_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, -1 ≤ t → t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (a * (-1) + b = -4 ∧ c * (-1) + d = 10) →
  (a * 1 + b = 2 ∧ c * 1 + d = -3) →
  a^2 + b^2 + c^2 + d^2 = 321 :=
by sorry

end line_segment_parameterization_sum_of_squares_l710_71048


namespace cube_sum_of_roots_l710_71035

theorem cube_sum_of_roots (p q r : ℂ) : 
  (p^3 - 2*p^2 + 3*p - 4 = 0) →
  (q^3 - 2*q^2 + 3*q - 4 = 0) →
  (r^3 - 2*r^2 + 3*r - 4 = 0) →
  p^3 + q^3 + r^3 = 2 := by
sorry

end cube_sum_of_roots_l710_71035


namespace dog_age_is_12_l710_71005

def cat_age : ℕ := 8

def rabbit_age (cat_age : ℕ) : ℕ := cat_age / 2

def dog_age (rabbit_age : ℕ) : ℕ := 3 * rabbit_age

theorem dog_age_is_12 : dog_age (rabbit_age cat_age) = 12 := by
  sorry

end dog_age_is_12_l710_71005


namespace race_problem_l710_71055

/-- Race problem statement -/
theorem race_problem (race_length : ℕ) (distance_between : ℕ) (jack_distance : ℕ) :
  race_length = 1000 →
  distance_between = 848 →
  jack_distance = race_length - distance_between →
  jack_distance = 152 :=
by sorry

end race_problem_l710_71055


namespace f_2018_equals_1_l710_71007

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem f_2018_equals_1
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_f0 : f 0 = -1)
  (h_fx : ∀ x, f x = -f (2 - x)) :
  f 2018 = 1 := by
  sorry

end f_2018_equals_1_l710_71007


namespace second_year_sampled_is_thirteen_l710_71012

/-- Calculates the number of second-year students sampled in a stratified survey. -/
def second_year_sampled (total_population : ℕ) (second_year_population : ℕ) (total_sampled : ℕ) : ℕ :=
  (second_year_population * total_sampled) / total_population

/-- Proves that the number of second-year students sampled is 13 given the problem conditions. -/
theorem second_year_sampled_is_thirteen :
  second_year_sampled 2100 780 35 = 13 := by
  sorry

end second_year_sampled_is_thirteen_l710_71012


namespace complement_P_inter_Q_l710_71015

open Set

def P : Set ℝ := { x | x - 1 ≤ 0 }
def Q : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

theorem complement_P_inter_Q : (P.compl ∩ Q) = Ioo 1 2 := by sorry

end complement_P_inter_Q_l710_71015


namespace cone_cylinder_volume_ratio_l710_71066

theorem cone_cylinder_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * π * r^2 * (h / 3)) / (π * r^2 * h) = 1 / 9 := by
  sorry

end cone_cylinder_volume_ratio_l710_71066


namespace total_lobster_amount_l710_71025

/-- The amount of lobster in pounds for each harbor -/
structure HarborLobster where
  hooperBay : ℝ
  harborA : ℝ
  harborB : ℝ
  harborC : ℝ
  harborD : ℝ

/-- The conditions for the lobster distribution problem -/
def LobsterDistribution (h : HarborLobster) : Prop :=
  h.harborA = 50 ∧
  h.harborB = 70.5 ∧
  h.harborC = (2/3) * h.harborB ∧
  h.harborD = h.harborA - 0.15 * h.harborA ∧
  h.hooperBay = 3 * (h.harborA + h.harborB + h.harborC + h.harborD)

/-- The theorem stating that the total amount of lobster is 840 pounds -/
theorem total_lobster_amount (h : HarborLobster) 
  (hDist : LobsterDistribution h) : 
  h.hooperBay + h.harborA + h.harborB + h.harborC + h.harborD = 840 := by
  sorry

end total_lobster_amount_l710_71025


namespace max_students_above_median_l710_71072

theorem max_students_above_median (n : ℕ) (h : n = 101) :
  ∃ (scores : Fin n → ℝ),
    (∃ (median : ℝ), ∀ i : Fin n, scores i ≥ median → 
      (Fintype.card {i : Fin n | scores i > median} ≤ 50)) ∧
    (∃ (median : ℝ), Fintype.card {i : Fin n | scores i > median} = 50) :=
by sorry

end max_students_above_median_l710_71072


namespace fourth_tea_price_theorem_l710_71078

/-- Calculates the price of the fourth tea variety given the prices of three varieties,
    their mixing ratios, and the final mixture price. -/
def fourth_tea_price (p1 p2 p3 mix_price : ℚ) : ℚ :=
  let r1 : ℚ := 2
  let r2 : ℚ := 3
  let r3 : ℚ := 4
  let r4 : ℚ := 5
  let total_ratio : ℚ := r1 + r2 + r3 + r4
  (mix_price * total_ratio - (p1 * r1 + p2 * r2 + p3 * r3)) / r4

/-- Theorem stating that given the prices of three tea varieties, their mixing ratios,
    and the final mixture price, the price of the fourth variety is 205.8. -/
theorem fourth_tea_price_theorem (p1 p2 p3 mix_price : ℚ) 
  (h1 : p1 = 126) (h2 : p2 = 135) (h3 : p3 = 156) (h4 : mix_price = 165) :
  fourth_tea_price p1 p2 p3 mix_price = 205.8 := by
  sorry

#eval fourth_tea_price 126 135 156 165

end fourth_tea_price_theorem_l710_71078


namespace plane_determining_pairs_count_l710_71070

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  /-- The number of edges in a regular tetrahedron -/
  num_edges : ℕ
  /-- The number of edges that intersect with each edge -/
  intersecting_edges : ℕ
  /-- Property that the number of edges is 6 -/
  edge_count : num_edges = 6
  /-- Property that each edge intersects with 2 other edges -/
  intersect_count : intersecting_edges = 2
  /-- Property that there are no skew edges -/
  no_skew_edges : True

/-- The number of unordered pairs of edges that determine a plane in a regular tetrahedron -/
def plane_determining_pairs (t : RegularTetrahedron) : ℕ :=
  t.num_edges * t.intersecting_edges / 2

/-- Theorem stating that the number of unordered pairs of edges that determine a plane in a regular tetrahedron is 6 -/
theorem plane_determining_pairs_count (t : RegularTetrahedron) :
  plane_determining_pairs t = 6 := by
  sorry

end plane_determining_pairs_count_l710_71070


namespace journey_distance_l710_71041

theorem journey_distance (total_journey : ℕ) (remaining : ℕ) (driven : ℕ) : 
  total_journey = 1200 → remaining = 277 → driven = total_journey - remaining → driven = 923 := by
sorry

end journey_distance_l710_71041


namespace sum_of_coordinates_D_l710_71061

/-- Given that M(5,3) is the midpoint of segment CD and C(2,6), prove that the sum of D's coordinates is 8 -/
theorem sum_of_coordinates_D (C D M : ℝ × ℝ) : 
  C = (2, 6) →
  M = (5, 3) →
  M.1 = (C.1 + D.1) / 2 →
  M.2 = (C.2 + D.2) / 2 →
  D.1 + D.2 = 8 := by
  sorry

end sum_of_coordinates_D_l710_71061


namespace arithmetic_expression_equals_one_l710_71001

theorem arithmetic_expression_equals_one :
  2016 * 2014 - 2013 * 2015 + 2012 * 2015 - 2013 * 2016 = 1 := by
  sorry

end arithmetic_expression_equals_one_l710_71001


namespace total_eyes_count_l710_71010

theorem total_eyes_count (num_boys : ℕ) (eyes_per_boy : ℕ) (h1 : num_boys = 23) (h2 : eyes_per_boy = 2) :
  num_boys * eyes_per_boy = 46 := by
  sorry

end total_eyes_count_l710_71010


namespace solution_values_l710_71026

def has_55_solutions (n : ℕ+) : Prop :=
  (Finset.filter (fun (x, y, z) => 3 * x + 3 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0)
    (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card = 55

theorem solution_values (n : ℕ+) (h : has_55_solutions n) : n = 34 ∨ n = 37 := by
  sorry

end solution_values_l710_71026


namespace propositions_truth_l710_71019

theorem propositions_truth :
  (¬ ∀ x : ℝ, x^4 > x^2) ∧
  (∃ α : ℝ, Real.sin (3 * α) = 3 * Real.sin α) ∧
  (¬ ∃ a : ℝ, ∀ x : ℝ, x^2 + 2*x + a < 0) :=
by sorry

end propositions_truth_l710_71019


namespace unique_solution_condition_l710_71027

theorem unique_solution_condition (b : ℝ) : 
  (∃! x : ℝ, x^3 - 2*b*x^2 + b*x + b^2 - 2 = 0) ↔ (b = 0 ∨ b = 2) :=
by sorry

end unique_solution_condition_l710_71027


namespace sixth_degree_polynomial_identity_l710_71090

theorem sixth_degree_polynomial_identity (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 
     (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃)) : 
  b₁^2 + b₂^2 + b₃^2 = 1 := by
  sorry

end sixth_degree_polynomial_identity_l710_71090


namespace expand_product_l710_71052

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end expand_product_l710_71052


namespace mans_age_to_sons_age_ratio_l710_71063

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1 -/
theorem mans_age_to_sons_age_ratio :
  ∀ (man_age son_age : ℕ),
    son_age = 18 →
    man_age = son_age + 20 →
    ∃ (k : ℕ), (man_age + 2) = k * (son_age + 2) →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end mans_age_to_sons_age_ratio_l710_71063


namespace equal_numbers_product_l710_71033

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 ∧ 
  a = 12 ∧ 
  b = 22 ∧ 
  c = d → 
  c * d = 529 := by
sorry

end equal_numbers_product_l710_71033


namespace replaced_person_weight_l710_71099

/-- Given a group of 8 people, if replacing one person with a new person weighing 89 kg
    increases the average weight by 3 kg, then the replaced person's weight was 65 kg. -/
theorem replaced_person_weight (initial_count : Nat) (new_person_weight : ℝ) (average_increase : ℝ) :
  initial_count = 8 →
  new_person_weight = 89 →
  average_increase = 3 →
  new_person_weight - (initial_count : ℝ) * average_increase = 65 :=
by sorry

end replaced_person_weight_l710_71099


namespace fixed_charge_is_six_l710_71029

/-- Represents Elvin's telephone bill components and totals -/
structure PhoneBill where
  fixed_charge : ℝ  -- Fixed monthly charge for internet service
  january_call_charge : ℝ  -- Charge for calls made in January
  january_total : ℝ  -- Total bill for January
  february_total : ℝ  -- Total bill for February

/-- Theorem stating that given the conditions, the fixed monthly charge is $6 -/
theorem fixed_charge_is_six (bill : PhoneBill) 
  (h1 : bill.fixed_charge + bill.january_call_charge = bill.january_total)
  (h2 : bill.fixed_charge + 2 * bill.january_call_charge = bill.february_total)
  (h3 : bill.january_total = 48)
  (h4 : bill.february_total = 90) :
  bill.fixed_charge = 6 := by
  sorry

end fixed_charge_is_six_l710_71029


namespace integral_reciprocal_plus_x_l710_71084

theorem integral_reciprocal_plus_x : ∫ x in (2 : ℝ)..4, (1 / x + x) = Real.log 2 + 6 := by
  sorry

end integral_reciprocal_plus_x_l710_71084


namespace complex_equation_solution_l710_71076

theorem complex_equation_solution (a : ℝ) : (a + Complex.I) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end complex_equation_solution_l710_71076


namespace compound_interest_rate_l710_71092

/-- Compound interest calculation --/
theorem compound_interest_rate (P : ℝ) (t : ℝ) (n : ℝ) (CI : ℝ) (r : ℝ) : 
  P = 20000 →
  t = 2 →
  n = 2 →
  CI = 1648.64 →
  (P + CI) = P * (1 + r / n) ^ (n * t) →
  r = 0.04 := by
sorry

end compound_interest_rate_l710_71092


namespace one_student_per_class_l710_71080

/-- Represents a school with a reading program -/
structure School where
  classes : ℕ
  books_per_student_per_month : ℕ
  total_books_per_year : ℕ

/-- Calculates the number of students in each class -/
def students_per_class (school : School) : ℕ :=
  school.total_books_per_year / (school.books_per_student_per_month * 12)

/-- Theorem stating that the number of students in each class is 1 -/
theorem one_student_per_class (school : School) 
  (h1 : school.classes > 0)
  (h2 : school.books_per_student_per_month = 3)
  (h3 : school.total_books_per_year = 36) : 
  students_per_class school = 1 := by
  sorry

#check one_student_per_class

end one_student_per_class_l710_71080


namespace factorial_500_trailing_zeroes_l710_71079

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end factorial_500_trailing_zeroes_l710_71079


namespace intersection_complement_theorem_l710_71071

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 1}

theorem intersection_complement_theorem :
  A ∩ (Set.univ \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_complement_theorem_l710_71071


namespace max_area_is_eight_l710_71058

/-- A line in the form kx - y + 2 = 0 -/
structure Line where
  k : ℝ

/-- A circle in the form x^2 + y^2 - 4x - 12 = 0 -/
def Circle : Type := Unit

/-- Points of intersection between the line and the circle -/
structure Intersection where
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The maximum area of triangle QRC given a line and a circle -/
def max_area (l : Line) (C : Circle) (i : Intersection) : ℝ := 8

/-- Theorem stating that the maximum area of triangle QRC is 8 -/
theorem max_area_is_eight (l : Line) (C : Circle) (i : Intersection) :
  max_area l C i = 8 := by sorry

end max_area_is_eight_l710_71058


namespace sum_of_reciprocal_squares_l710_71034

-- Define the cubic equation
def cubic_equation (x : ℝ) : Prop := x^3 - 6*x^2 + 11*x - 6 = 0

-- Define the roots of the equation
def roots (a b c : ℝ) : Prop := cubic_equation a ∧ cubic_equation b ∧ cubic_equation c

-- Theorem statement
theorem sum_of_reciprocal_squares (a b c : ℝ) :
  roots a b c → 1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by sorry

end sum_of_reciprocal_squares_l710_71034


namespace min_value_fraction_sum_l710_71000

theorem min_value_fraction_sum (x y a b : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ a > 0 ∧ b > 0) 
  (h_sum : x + y = 1) : 
  (a / x + b / y) ≥ (Real.sqrt a + Real.sqrt b)^2 := by
  sorry

end min_value_fraction_sum_l710_71000


namespace collinear_vectors_x_value_l710_71091

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (2, -3)
  let b : ℝ × ℝ := (x, 6)
  collinear a b → x = -4 :=
by
  sorry

end collinear_vectors_x_value_l710_71091


namespace some_students_not_club_members_l710_71068

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (StudyScience : U → Prop)
variable (ClubMember : U → Prop)
variable (Honest : U → Prop)

-- Define the conditions
variable (h1 : ∃ x, Student x ∧ ¬StudyScience x)
variable (h2 : ∀ x, ClubMember x → (StudyScience x ∧ Honest x))

-- State the theorem
theorem some_students_not_club_members :
  ∃ x, Student x ∧ ¬ClubMember x :=
sorry

end some_students_not_club_members_l710_71068


namespace complex_power_sum_l710_71074

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^1800 + 1/(z^1800) = -2 := by
  sorry

end complex_power_sum_l710_71074


namespace maxim_birth_probability_maxim_birth_probability_proof_l710_71020

/-- The year Maxim starts first grade -/
def start_year : ℕ := 2014

/-- The month Maxim starts first grade (September = 9) -/
def start_month : ℕ := 9

/-- The day Maxim starts first grade -/
def start_day : ℕ := 1

/-- Maxim's age when he starts first grade -/
def start_age : ℕ := 6

/-- The year we're interested in for Maxim's birth -/
def birth_year_of_interest : ℕ := 2008

/-- Function to determine if a year is a leap year -/
def is_leap_year (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Function to get the number of days in a month -/
def days_in_month (year : ℕ) (month : ℕ) : ℕ :=
  if month == 2 then
    if is_leap_year year then 29 else 28
  else if month ∈ [4, 6, 9, 11] then 30
  else 31

/-- The probability that Maxim was born in 2008 -/
theorem maxim_birth_probability : ℚ :=
  244 / 365

/-- Proof of the probability calculation -/
theorem maxim_birth_probability_proof :
  maxim_birth_probability = 244 / 365 := by
  sorry

end maxim_birth_probability_maxim_birth_probability_proof_l710_71020


namespace converse_statement_l710_71047

/-- Given that m is a real number, prove that the converse of the statement 
    "If m > 0, then the equation x^2 + x - m = 0 has real roots" 
    is "If the equation x^2 + x - m = 0 has real roots, then m > 0" -/
theorem converse_statement (m : ℝ) : 
  (∃ x : ℝ, x^2 + x - m = 0) → m > 0 :=
sorry

end converse_statement_l710_71047


namespace workshop_workers_count_workshop_workers_count_is_49_l710_71075

/-- Proves that the total number of workers in a workshop is 49, given the following conditions:
  * The average salary of all workers is 8000
  * There are 7 technicians with an average salary of 20000
  * The average salary of the non-technicians is 6000
-/
theorem workshop_workers_count : ℕ → Prop :=
  fun (total_workers : ℕ) =>
    let avg_salary : ℚ := 8000
    let technician_count : ℕ := 7
    let technician_avg_salary : ℚ := 20000
    let non_technician_avg_salary : ℚ := 6000
    let non_technician_count : ℕ := total_workers - technician_count
    (↑total_workers * avg_salary = 
      ↑technician_count * technician_avg_salary + 
      ↑non_technician_count * non_technician_avg_salary) →
    total_workers = 49

theorem workshop_workers_count_is_49 : workshop_workers_count 49 := by
  sorry

end workshop_workers_count_workshop_workers_count_is_49_l710_71075


namespace triangle_line_equations_l710_71086

/-- Triangle ABC with vertices A(-4,0), B(0,-3), and C(-2,1) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Definition of the specific triangle in the problem -/
def triangle_ABC : Triangle :=
  { A := (-4, 0)
  , B := (0, -3)
  , C := (-2, 1) }

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the equations of line BC and altitude from A to BC -/
theorem triangle_line_equations (t : Triangle) (h : t = triangle_ABC) :
  ∃ (line_BC altitude : LineEquation),
    line_BC = { a := 2, b := 1, c := 3 } ∧
    altitude = { a := 1, b := -2, c := 4 } := by
  sorry

end triangle_line_equations_l710_71086


namespace sector_arc_length_l710_71004

theorem sector_arc_length (s r p : ℝ) : 
  s = 4 → r = 2 → s = (1/2) * r * p → p = 4 := by sorry

end sector_arc_length_l710_71004


namespace first_player_winning_strategy_l710_71088

/-- Represents the number of points on the circle -/
def n : Nat := 98

/-- Represents the number of moves to connect n-2 points -/
def N (n : Nat) : Nat := (n - 3) * (n - 4) / 2

/-- Represents whether a number is odd -/
def isOdd (m : Nat) : Prop := ∃ k, m = 2 * k + 1

/-- Represents the winning condition for the first player -/
def firstPlayerWins (n : Nat) : Prop := isOdd (N n)

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_winning_strategy : firstPlayerWins n := by
  sorry

end first_player_winning_strategy_l710_71088


namespace chopped_cube_height_l710_71069

/-- The height of a cube with a chopped corner -/
theorem chopped_cube_height (s : ℝ) (h_s : s = 2) : 
  let diagonal := s * Real.sqrt 3
  let triangle_side := Real.sqrt (2 * s^2)
  let triangle_area := (Real.sqrt 3 / 4) * triangle_side^2
  let pyramid_volume := (1 / 6) * s^3
  let pyramid_height := 3 * pyramid_volume / triangle_area
  s - pyramid_height = (2 * Real.sqrt 3 - 1) / Real.sqrt 3 := by
  sorry

end chopped_cube_height_l710_71069


namespace inclination_angle_range_l710_71038

/-- The range of inclination angles for a line with equation x*cos(θ) + √3*y - 1 = 0 -/
theorem inclination_angle_range (θ : ℝ) :
  let l : Set (ℝ × ℝ) := {(x, y) | x * Real.cos θ + Real.sqrt 3 * y - 1 = 0}
  let α := Real.arctan (-Real.sqrt 3 / 3 * Real.cos θ)
  α ∈ Set.union (Set.Icc 0 (Real.pi / 6)) (Set.Icc (5 * Real.pi / 6) Real.pi) :=
by sorry


end inclination_angle_range_l710_71038


namespace largest_valid_partition_l710_71014

/-- Represents a partition of the set {1, 2, ..., m} into n subsets -/
def Partition (m : ℕ) (n : ℕ) := Fin n → Finset (Fin m)

/-- Checks if a partition satisfies the condition that the product of two different
    elements in the same subset is never a perfect square -/
def ValidPartition (p : Partition m n) : Prop :=
  ∀ i : Fin n, ∀ x y : Fin m, x ∈ p i → y ∈ p i → x ≠ y →
    ¬ ∃ z : ℕ, (x.val + 1) * (y.val + 1) = z * z

/-- The main theorem stating that n^2 + 2n is the largest m for which
    a valid partition exists -/
theorem largest_valid_partition (n : ℕ) (h : 0 < n) :
  (∃ p : Partition (n^2 + 2*n) n, ValidPartition p) ∧
  (∀ m : ℕ, m > n^2 + 2*n → ¬ ∃ p : Partition m n, ValidPartition p) :=
sorry

end largest_valid_partition_l710_71014


namespace distribute_problems_l710_71043

theorem distribute_problems (num_problems : ℕ) (num_friends : ℕ) :
  num_problems = 6 → num_friends = 15 →
  (num_friends : ℕ) ^ (num_problems : ℕ) = 11390625 := by
  sorry

end distribute_problems_l710_71043


namespace six_digit_square_number_puzzle_l710_71023

theorem six_digit_square_number_puzzle :
  ∃ (n x y : ℕ), 
    100000 ≤ n^2 ∧ n^2 < 1000000 ∧
    10 ≤ x ∧ x ≤ 99 ∧
    0 ≤ y ∧ y ≤ 9 ∧
    n^2 = 10101 * x + y^2 ∧
    (n^2 = 232324 ∨ n^2 = 595984 ∨ n^2 = 929296) :=
by sorry

end six_digit_square_number_puzzle_l710_71023


namespace tan_x0_equals_3_l710_71064

/-- Given a function f(x) = sin x - cos x, prove that if f''(x₀) = 2f(x₀), then tan x₀ = 3 -/
theorem tan_x0_equals_3 (x₀ : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin x - Real.cos x
  (deriv (deriv f)) x₀ = 2 * f x₀ → Real.tan x₀ = 3 := by
  sorry

end tan_x0_equals_3_l710_71064


namespace trash_outside_classrooms_l710_71087

theorem trash_outside_classrooms 
  (total_trash : ℕ) 
  (classroom_trash : ℕ) 
  (h1 : total_trash = 1576) 
  (h2 : classroom_trash = 344) : 
  total_trash - classroom_trash = 1232 := by
sorry

end trash_outside_classrooms_l710_71087


namespace exists_winning_strategy_l710_71049

/-- Represents the state of the switch -/
inductive SwitchState
| Left
| Right

/-- Represents a trainee's action in the room -/
inductive TraineeAction
| FlipSwitch
| DoNothing
| Declare

/-- Represents the result of the challenge -/
inductive ChallengeResult
| Success
| Failure

/-- The strategy function type for a trainee -/
def TraineeStrategy := Nat → SwitchState → TraineeAction

/-- The type representing the challenge setup -/
structure Challenge where
  numTrainees : Nat
  initialState : SwitchState

/-- The function to simulate the challenge -/
noncomputable def simulateChallenge (c : Challenge) (strategies : List TraineeStrategy) : ChallengeResult :=
  sorry

/-- The main theorem to prove -/
theorem exists_winning_strategy :
  ∃ (strategies : List TraineeStrategy),
    strategies.length = 42 ∧
    ∀ (c : Challenge),
      c.numTrainees = 42 →
      simulateChallenge c strategies = ChallengeResult.Success :=
sorry

end exists_winning_strategy_l710_71049


namespace abs_sum_complex_roots_l710_71065

/-- Given complex numbers a, b, and c satisfying certain conditions,
    prove that |a + b + c| is either 0 or 1. -/
theorem abs_sum_complex_roots (a b c : ℂ) 
    (h1 : Complex.abs a = 1)
    (h2 : Complex.abs b = 1)
    (h3 : Complex.abs c = 1)
    (h4 : a^2 * b + b^2 * c + c^2 * a = 0) :
    Complex.abs (a + b + c) = 0 ∨ Complex.abs (a + b + c) = 1 := by
  sorry

end abs_sum_complex_roots_l710_71065


namespace pots_needed_for_path_l710_71067

/-- Calculate the number of pots needed for a path with given specifications. -/
def calculate_pots (path_length : ℕ) (pot_distance : ℕ) : ℕ :=
  let pots_per_side := path_length / pot_distance + 1
  2 * pots_per_side

/-- Theorem stating that 152 pots are needed for the given path specifications. -/
theorem pots_needed_for_path : calculate_pots 150 2 = 152 := by
  sorry

#eval calculate_pots 150 2

end pots_needed_for_path_l710_71067


namespace repeating_decimal_proof_l710_71089

/-- The repeating decimal 0.817817817... as a real number -/
def F : ℚ := 817 / 999

/-- The difference between the denominator and numerator of F when expressed as a fraction in lowest terms -/
def denominator_numerator_difference : ℕ := 999 - 817

theorem repeating_decimal_proof :
  F = 817 / 999 ∧ denominator_numerator_difference = 182 :=
sorry

end repeating_decimal_proof_l710_71089


namespace average_score_is_94_l710_71013

def june_score : ℝ := 97
def patty_score : ℝ := 85
def josh_score : ℝ := 100
def henry_score : ℝ := 94

def num_children : ℕ := 4

theorem average_score_is_94 :
  (june_score + patty_score + josh_score + henry_score) / num_children = 94 := by
  sorry

end average_score_is_94_l710_71013


namespace range_of_a_l710_71028

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ [0, 1] ∧ 
    x₀ + (Real.exp 2 - 1) * Real.log a ≥ (2 * a / Real.exp x₀) + Real.exp 2 * x₀ - 2) →
  a ∈ Set.Icc 1 (Real.exp 3) :=
by sorry

end range_of_a_l710_71028


namespace abs_inequality_abs_inequality_with_constraints_l710_71044

-- Part I
theorem abs_inequality (x : ℝ) : 
  |x - 1| + |2*x + 1| > 3 ↔ x < -1 ∨ x > 1 := by sorry

-- Part II
theorem abs_inequality_with_constraints (a b : ℝ) 
  (ha : a ∈ Set.Icc (-1 : ℝ) 1) (hb : b ∈ Set.Icc (-1 : ℝ) 1) : 
  |1 + a*b/4| > |(a + b)/2| := by sorry

end abs_inequality_abs_inequality_with_constraints_l710_71044


namespace remainder_of_m_l710_71021

theorem remainder_of_m (m : ℕ) (h1 : m^2 % 7 = 1) (h2 : m^3 % 7 = 6) : m % 7 = 6 := by
  sorry

end remainder_of_m_l710_71021


namespace max_value_of_f_l710_71008

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 2

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- Statement to prove
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ interval ∧ ∀ (x : ℝ), x ∈ interval → f x ≤ f c ∧ f c = 7 :=
sorry

end max_value_of_f_l710_71008


namespace gcd_lcm_product_180_l710_71056

theorem gcd_lcm_product_180 (a b : ℕ+) :
  (Nat.gcd a b) * (Nat.lcm a b) = 180 →
  (∃ s : Finset ℕ+, s.card = 9 ∧ ∀ x, x ∈ s ↔ ∃ c d : ℕ+, (Nat.gcd c d) * (Nat.lcm c d) = 180 ∧ Nat.gcd c d = x) :=
by sorry

end gcd_lcm_product_180_l710_71056


namespace students_not_eating_lunch_l710_71039

theorem students_not_eating_lunch (total : ℕ) (cafeteria : ℕ) (bring_lunch_multiplier : ℕ) :
  total = 90 →
  bring_lunch_multiplier = 4 →
  cafeteria = 12 →
  total - (cafeteria + bring_lunch_multiplier * cafeteria) = 30 :=
by sorry

end students_not_eating_lunch_l710_71039


namespace prob_sum_less_than_7_is_5_12_l710_71060

/-- The number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum less than 7) -/
def favorable_outcomes : ℕ := 15

/-- The probability of getting a sum less than 7 when throwing two dice -/
def prob_sum_less_than_7 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_less_than_7_is_5_12 : prob_sum_less_than_7 = 5 / 12 := by
  sorry

end prob_sum_less_than_7_is_5_12_l710_71060


namespace complex_cube_root_l710_71037

theorem complex_cube_root : ∃ (z : ℂ), z^2 + 2 = 0 → z^3 = 2 * Real.sqrt 2 * I ∨ z^3 = -2 * Real.sqrt 2 * I := by
  sorry

end complex_cube_root_l710_71037


namespace rand_code_is_1236_l710_71042

/-- Represents a coding system for words -/
structure CodeSystem where
  range_code : Nat
  random_code : Nat

/-- Extracts the code for a given letter based on its position in a word -/
def extract_code (n : Nat) (code : Nat) : Nat :=
  (code / (10 ^ (5 - n))) % 10

/-- Determines the code for "rand" based on the given coding system -/
def rand_code (cs : CodeSystem) : Nat :=
  let r := extract_code 1 cs.range_code
  let a := extract_code 2 cs.range_code
  let n := extract_code 3 cs.range_code
  let d := extract_code 4 cs.random_code
  r * 1000 + a * 100 + n * 10 + d

/-- Theorem stating that the code for "rand" is 1236 given the specified coding system -/
theorem rand_code_is_1236 (cs : CodeSystem) 
    (h1 : cs.range_code = 12345) 
    (h2 : cs.random_code = 123678) : 
  rand_code cs = 1236 := by
  sorry

end rand_code_is_1236_l710_71042


namespace walking_distance_l710_71062

/-- Proves that walking at 3 miles per hour for 1.5 hours results in a distance of 4.5 miles -/
theorem walking_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 3 → time = 1.5 → distance = speed * time → distance = 4.5 := by
  sorry

end walking_distance_l710_71062


namespace tax_calculation_correct_l710_71016

/-- Represents the tax rate for a given income range -/
structure TaxBracket where
  lower : ℝ
  upper : Option ℝ
  rate : ℝ

/-- Calculates the tax for a given taxable income -/
def calculateTax (brackets : List TaxBracket) (taxableIncome : ℝ) : ℝ :=
  sorry

/-- Represents the tax system with its parameters -/
structure TaxSystem where
  threshold : ℝ
  brackets : List TaxBracket
  elderlyDeduction : ℝ

/-- Calculates the after-tax income given a pre-tax income and tax system -/
def afterTaxIncome (preTaxIncome : ℝ) (system : TaxSystem) : ℝ :=
  sorry

theorem tax_calculation_correct (preTaxIncome : ℝ) (system : TaxSystem) :
  let taxPaid := 180
  let afterTax := 9720
  system.threshold = 5000 ∧
  system.elderlyDeduction = 1000 ∧
  system.brackets = [
    ⟨0, some 3000, 0.03⟩,
    ⟨3000, some 12000, 0.10⟩,
    ⟨12000, some 25000, 0.20⟩,
    ⟨25000, none, 0.25⟩
  ] →
  calculateTax system.brackets (preTaxIncome - system.threshold - system.elderlyDeduction) = taxPaid ∧
  afterTaxIncome preTaxIncome system = afterTax :=
by sorry

end tax_calculation_correct_l710_71016


namespace system_solution_l710_71095

theorem system_solution :
  ∃ (x y : ℚ), 
    (7 * x = -9 - 3 * y) ∧ 
    (4 * x = 5 * y - 34) ∧ 
    (x = -413 / 235) ∧ 
    (y = -202 / 47) := by
  sorry

end system_solution_l710_71095


namespace sum_of_page_numbers_constant_l710_71030

/-- Represents a magazine with nested double sheets. -/
structure Magazine where
  num_double_sheets : ℕ
  pages_per_double_sheet : ℕ

/-- Calculates the sum of page numbers on a double sheet. -/
def sum_of_page_numbers (m : Magazine) (sheet_number : ℕ) : ℕ :=
  sorry

/-- Theorem: The sum of page numbers on each double sheet is always 130. -/
theorem sum_of_page_numbers_constant (m : Magazine) (sheet_number : ℕ) :
  m.num_double_sheets = 16 →
  m.pages_per_double_sheet = 4 →
  sheet_number ≤ m.num_double_sheets →
  sum_of_page_numbers m sheet_number = 130 :=
sorry

end sum_of_page_numbers_constant_l710_71030


namespace subtraction_of_negatives_l710_71082

theorem subtraction_of_negatives : (-2) - (-4) = 2 := by
  sorry

end subtraction_of_negatives_l710_71082


namespace absolute_value_equation_unique_solution_l710_71083

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 3| = |x + 5| := by
sorry

end absolute_value_equation_unique_solution_l710_71083


namespace school_greening_area_equation_l710_71053

/-- Represents the growth of a greening area over time -/
def greeningAreaGrowth (initialArea finalArea : ℝ) (years : ℕ) (growthRate : ℝ) : Prop :=
  initialArea * (1 + growthRate) ^ years = finalArea

/-- The equation for the school's greening area growth -/
theorem school_greening_area_equation :
  greeningAreaGrowth 1000 1440 2 x ↔ 1000 * (1 + x)^2 = 1440 := by
  sorry

end school_greening_area_equation_l710_71053


namespace only_one_statement_correct_l710_71031

-- Define the concept of opposite numbers
def are_opposites (a b : ℝ) : Prop := a + b = 0

-- Define the four statements
def statement1 : Prop := ∀ a b : ℝ, (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) → are_opposites a b
def statement2 : Prop := ∀ a : ℝ, ∃ b : ℝ, are_opposites a b ∧ b < 0
def statement3 : Prop := ∀ a b : ℝ, are_opposites a b → a + b = 0
def statement4 : Prop := ∀ a b : ℝ, are_opposites a b → (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)

-- Theorem stating that only one of the statements is correct
theorem only_one_statement_correct :
  (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ∧
  ¬(statement1 ∨ statement2 ∨ statement4) :=
sorry

end only_one_statement_correct_l710_71031


namespace cos_product_sevenths_pi_l710_71077

theorem cos_product_sevenths_pi : 
  Real.cos (π / 7) * Real.cos (2 * π / 7) * Real.cos (4 * π / 7) = -1 / 8 := by
  sorry

end cos_product_sevenths_pi_l710_71077


namespace federal_guideline_requirement_l710_71002

/-- The daily minimum requirement of vegetables in cups according to federal guidelines. -/
def daily_requirement : ℕ := 3

/-- The number of days Sarah has been eating vegetables. -/
def days_counted : ℕ := 5

/-- The total amount of vegetables Sarah has eaten in cups. -/
def vegetables_eaten : ℕ := 8

/-- Sarah's daily consumption needed to meet the minimum requirement. -/
def sarah_daily_need : ℕ := 3

theorem federal_guideline_requirement :
  daily_requirement = sarah_daily_need :=
by sorry

end federal_guideline_requirement_l710_71002


namespace equation_solution_l710_71051

theorem equation_solution (x y : ℝ) :
  x * y^3 - y^2 = y * x^3 - x^2 → y = -x ∨ y = x ∨ y = 1 / x :=
by sorry

end equation_solution_l710_71051


namespace swim_club_additional_capacity_l710_71094

/-- Represents the swimming club's transportation setup -/
structure SwimClubTransport where
  num_cars : ℕ
  num_vans : ℕ
  people_per_car : ℕ
  people_per_van : ℕ
  max_car_capacity : ℕ
  max_van_capacity : ℕ

/-- Calculates the additional capacity of the swim club's transportation -/
def additional_capacity (t : SwimClubTransport) : ℕ :=
  (t.num_cars * t.max_car_capacity + t.num_vans * t.max_van_capacity) -
  (t.num_cars * t.people_per_car + t.num_vans * t.people_per_van)

/-- Theorem stating that the additional capacity for the given scenario is 17 -/
theorem swim_club_additional_capacity :
  let t : SwimClubTransport :=
    { num_cars := 2
    , num_vans := 3
    , people_per_car := 5
    , people_per_van := 3
    , max_car_capacity := 6
    , max_van_capacity := 8
    }
  additional_capacity t = 17 := by
  sorry


end swim_club_additional_capacity_l710_71094
