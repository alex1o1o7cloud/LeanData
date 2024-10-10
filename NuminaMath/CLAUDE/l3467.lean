import Mathlib

namespace number_problem_l3467_346754

theorem number_problem (x : ℝ) : 3 * (2 * x + 9) = 75 → x = 8 := by
  sorry

end number_problem_l3467_346754


namespace markup_percentage_l3467_346702

/-- Proves that given a cost price of 540, a selling price of 459, and a discount percentage
    of 26.08695652173913%, the percentage marked above the cost price is 15%. -/
theorem markup_percentage
  (cost_price : ℝ)
  (selling_price : ℝ)
  (discount_percentage : ℝ)
  (h_cost_price : cost_price = 540)
  (h_selling_price : selling_price = 459)
  (h_discount_percentage : discount_percentage = 26.08695652173913) :
  let marked_price := selling_price / (1 - discount_percentage / 100)
  (marked_price - cost_price) / cost_price * 100 = 15 := by
  sorry

end markup_percentage_l3467_346702


namespace miss_molly_class_size_l3467_346796

/-- The number of students in Miss Molly's class -/
def total_students : ℕ := 30

/-- The number of girls in the class -/
def num_girls : ℕ := 18

/-- The number of students who like yellow -/
def yellow_fans : ℕ := 9

/-- Theorem: The total number of students in Miss Molly's class is 30 -/
theorem miss_molly_class_size :
  (total_students / 2 = total_students - (num_girls / 3 + yellow_fans)) ∧
  (num_girls = 18) ∧
  (yellow_fans = 9) →
  total_students = 30 := by
sorry

end miss_molly_class_size_l3467_346796


namespace two_segment_trip_average_speed_l3467_346749

/-- Calculates the average speed of a two-segment trip -/
def average_speed (d1 d2 v1 v2 : ℚ) : ℚ :=
  (d1 + d2) / (d1 / v1 + d2 / v2)

theorem two_segment_trip_average_speed :
  let d1 : ℚ := 50
  let d2 : ℚ := 25
  let v1 : ℚ := 15
  let v2 : ℚ := 45
  average_speed d1 d2 v1 v2 = 675 / 35 := by
  sorry

end two_segment_trip_average_speed_l3467_346749


namespace abc_inequality_l3467_346701

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c = a^(1/7) + b^(1/7) + c^(1/7)) :
  a^a * b^b * c^c ≥ 1 := by
  sorry

end abc_inequality_l3467_346701


namespace min_swaps_for_initial_number_l3467_346747

def initial_number : ℕ := 9072543681

def is_divisible_by_99 (n : ℕ) : Prop :=
  n % 99 = 0

def adjacent_swap (n : ℕ) (i : ℕ) : ℕ :=
  sorry

def min_swaps_to_divisible_by_99 (n : ℕ) : ℕ :=
  sorry

theorem min_swaps_for_initial_number :
  min_swaps_to_divisible_by_99 initial_number = 2 :=
sorry

end min_swaps_for_initial_number_l3467_346747


namespace cylinder_volume_tripled_radius_cylinder_volume_increase_l3467_346776

/-- Proves that tripling the radius of a cylinder while keeping the height constant
    results in a volume that is 9 times the original volume. -/
theorem cylinder_volume_tripled_radius 
  (r h : ℝ) 
  (original_volume : ℝ) 
  (h_original_volume : original_volume = π * r^2 * h) 
  (h_positive : r > 0 ∧ h > 0) :
  let new_volume := π * (3*r)^2 * h
  new_volume = 9 * original_volume :=
by sorry

/-- Proves that if a cylinder with volume 10 cubic feet has its radius tripled
    while its height remains constant, its new volume is 90 cubic feet. -/
theorem cylinder_volume_increase
  (r h : ℝ)
  (h_original_volume : π * r^2 * h = 10)
  (h_positive : r > 0 ∧ h > 0) :
  let new_volume := π * (3*r)^2 * h
  new_volume = 90 :=
by sorry

end cylinder_volume_tripled_radius_cylinder_volume_increase_l3467_346776


namespace total_households_l3467_346733

/-- Represents the number of households in each category -/
structure HouseholdCounts where
  both : ℕ
  gasOnly : ℕ
  elecOnly : ℕ
  neither : ℕ

/-- The conditions of the survey -/
def surveyCounts : HouseholdCounts where
  both := 120
  gasOnly := 60
  elecOnly := 4 * 24
  neither := 24

/-- The theorem stating the total number of households surveyed -/
theorem total_households : 
  surveyCounts.both + surveyCounts.gasOnly + surveyCounts.elecOnly + surveyCounts.neither = 300 := by
  sorry


end total_households_l3467_346733


namespace prime_octuple_sum_product_relation_l3467_346703

theorem prime_octuple_sum_product_relation :
  ∀ (p₁ p₂ p₃ p₄ p₅ p₆ p₇ p₈ : ℕ),
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧
    Prime p₅ ∧ Prime p₆ ∧ Prime p₇ ∧ Prime p₈ →
    (p₁^2 + p₂^2 + p₃^2 + p₄^2 + p₅^2 + p₆^2 + p₇^2 + p₈^2 = 4 * (p₁ * p₂ * p₃ * p₄ * p₅ * p₆ * p₇ * p₈) - 992) →
    p₁ = 2 ∧ p₂ = 2 ∧ p₃ = 2 ∧ p₄ = 2 ∧ p₅ = 2 ∧ p₆ = 2 ∧ p₇ = 2 ∧ p₈ = 2 :=
by
  sorry

end prime_octuple_sum_product_relation_l3467_346703


namespace x_value_l3467_346719

theorem x_value :
  ∀ (x y z w : ℤ),
    x = y + 7 →
    y = z + 15 →
    z = w + 25 →
    w = 90 →
    x = 137 := by
  sorry

end x_value_l3467_346719


namespace min_common_perimeter_of_specific_isosceles_triangles_l3467_346725

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Checks if two isosceles triangles are noncongruent -/
def noncongruent (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.leg ≠ t2.leg ∨ t1.base ≠ t2.base

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ :=
  2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 2 : ℝ) * Real.sqrt ((t.leg : ℝ)^2 - (t.base / 2 : ℝ)^2)

/-- Theorem stating the minimum common perimeter of two specific isosceles triangles -/
theorem min_common_perimeter_of_specific_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    noncongruent t1 t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t2.base = 4 * t1.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      noncongruent s1 s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s2.base = 4 * s1.base →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 1180 :=
  sorry

end min_common_perimeter_of_specific_isosceles_triangles_l3467_346725


namespace fraction_sum_simplification_l3467_346755

theorem fraction_sum_simplification : (3 : ℚ) / 462 + 28 / 42 = 311 / 462 := by sorry

end fraction_sum_simplification_l3467_346755


namespace sum_probability_is_thirteen_sixteenths_l3467_346729

/-- Represents an n-sided die -/
structure Die (n : ℕ) where
  sides : Fin n → ℕ
  valid : ∀ i, sides i ∈ Finset.range n.succ

/-- The 8-sided die -/
def eight_sided_die : Die 8 :=
  { sides := λ i => i.val + 1,
    valid := by sorry }

/-- The 6-sided die -/
def six_sided_die : Die 6 :=
  { sides := λ i => i.val + 1,
    valid := by sorry }

/-- The set of all possible outcomes when rolling two dice -/
def outcomes : Finset (Fin 8 × Fin 6) :=
  Finset.product (Finset.univ : Finset (Fin 8)) (Finset.univ : Finset (Fin 6))

/-- The set of favorable outcomes (sum ≤ 10) -/
def favorable_outcomes : Finset (Fin 8 × Fin 6) :=
  outcomes.filter (λ p => eight_sided_die.sides p.1 + six_sided_die.sides p.2 ≤ 10)

/-- The probability of the sum being less than or equal to 10 -/
def probability : ℚ :=
  favorable_outcomes.card / outcomes.card

theorem sum_probability_is_thirteen_sixteenths :
  probability = 13 / 16 := by sorry

end sum_probability_is_thirteen_sixteenths_l3467_346729


namespace pebble_collection_proof_l3467_346791

def initial_pebbles : ℕ := 3
def collection_days : ℕ := 15
def first_day_collection : ℕ := 2
def daily_increase : ℕ := 1

def total_pebbles : ℕ := initial_pebbles + (collection_days * (2 * first_day_collection + (collection_days - 1) * daily_increase)) / 2

theorem pebble_collection_proof :
  total_pebbles = 138 := by
  sorry

end pebble_collection_proof_l3467_346791


namespace workshop_participation_l3467_346737

theorem workshop_participation (total : ℕ) (A B C : ℕ) (at_least_two : ℕ) 
  (h_total : total = 25)
  (h_A : A = 15)
  (h_B : B = 14)
  (h_C : C = 11)
  (h_at_least_two : at_least_two = 12)
  (h_sum : A + B + C ≥ total + at_least_two) :
  ∃ (x y z a b c : ℕ), 
    x + y + z + a + b + c = total ∧
    a + b + c = at_least_two ∧
    x + a + c = A ∧
    y + a + b = B ∧
    z + b + c = C ∧
    0 = total - (x + y + z + a + b + c) :=
by sorry

end workshop_participation_l3467_346737


namespace quadratic_function_theorem_l3467_346785

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The property that the range of a function is [0, +∞) -/
def HasNonnegativeRange (f : ℝ → ℝ) : Prop :=
  ∀ y, (∃ x, f x = y) → y ≥ 0

/-- The solution set of f(x) < c is an open interval of length 8 -/
def HasSolutionSetOfLength8 (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∃ m, ∀ x, f x < c ↔ m < x ∧ x < m + 8

theorem quadratic_function_theorem (a b : ℝ) :
  HasNonnegativeRange (QuadraticFunction a b) →
  HasSolutionSetOfLength8 (QuadraticFunction a b) 16 := by sorry

end quadratic_function_theorem_l3467_346785


namespace range_of_m_l3467_346756

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}

def B (m : ℝ) : Set ℝ := {x | x^2 - (m+3)*x + m^2 = 0}

theorem range_of_m : 
  ∀ m : ℝ, (A ∪ (Set.univ \ B m) = Set.univ) ↔ (m < -1 ∨ m ≥ 3) :=
sorry

end range_of_m_l3467_346756


namespace min_attempts_to_open_safe_l3467_346726

/-- Represents a sequence of 7 digits -/
def Code := Fin 7 → Fin 10

/-- Checks if all digits in a code are different -/
def all_different (c : Code) : Prop :=
  ∀ i j : Fin 7, i ≠ j → c i ≠ c j

/-- Checks if at least one digit in the attempt matches the secret code in the same position -/
def has_match (secret : Code) (attempt : Code) : Prop :=
  ∃ i : Fin 7, secret i = attempt i

/-- Represents a sequence of attempts to open the safe -/
def AttemptSequence (n : ℕ) := Fin n → Code

/-- Checks if a sequence of attempts guarantees opening the safe for any possible secret code -/
def guarantees_opening (attempts : AttemptSequence n) : Prop :=
  ∀ secret : Code, all_different secret →
    ∃ attempt ∈ Set.range attempts, all_different attempt ∧ has_match secret attempt

/-- The main theorem: 6 attempts are sufficient and necessary to guarantee opening the safe -/
theorem min_attempts_to_open_safe :
  (∃ attempts : AttemptSequence 6, guarantees_opening attempts) ∧
  (∀ n < 6, ¬∃ attempts : AttemptSequence n, guarantees_opening attempts) :=
sorry

end min_attempts_to_open_safe_l3467_346726


namespace extreme_points_condition_l3467_346778

/-- The function f(x) = ln x + ax^2 - 2x has two distinct extreme points
    if and only if 0 < a < 1/2, where x > 0 -/
theorem extreme_points_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (∀ x : ℝ, x > 0 → (((1 : ℝ) / x + 2 * a * x - 2 = 0) ↔ (x = x₁ ∨ x = x₂))))
  ↔ (0 < a ∧ a < (1 : ℝ) / 2) :=
by sorry


end extreme_points_condition_l3467_346778


namespace ce_length_l3467_346711

/-- Given a triangle ABC, this function returns true if the triangle is right-angled -/
def is_right_triangle (A B C : ℝ × ℝ) : Prop := sorry

/-- Given three points A, B, C, this function returns the measure of angle ABC in degrees -/
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given two points A and B, this function returns the distance between them -/
def distance (A B : ℝ × ℝ) : ℝ := sorry

theorem ce_length (A B C D E : ℝ × ℝ) 
  (h1 : is_right_triangle A B E)
  (h2 : is_right_triangle B C E)
  (h3 : is_right_triangle C D E)
  (h4 : angle_measure A E B = 60)
  (h5 : angle_measure B E C = 60)
  (h6 : angle_measure C E D = 60)
  (h7 : distance A E = 36) :
  distance C E = 9 := by sorry

end ce_length_l3467_346711


namespace solution_set_g_range_of_a_l3467_346732

-- Define the functions f and g
def f (a x : ℝ) := |2*x - a| + |2*x + 3|
def g (x : ℝ) := |x - 1| + 2

-- Theorem for part (1)
theorem solution_set_g (x : ℝ) : 
  |g x| < 5 ↔ -2 < x ∧ x < 4 := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) → 
  (a ≥ -1 ∨ a ≤ -5) := by sorry

end solution_set_g_range_of_a_l3467_346732


namespace x_plus_2y_squared_value_l3467_346795

theorem x_plus_2y_squared_value (x y : ℝ) :
  8 * y^4 + 4 * x^2 * y^2 + 4 * x * y^2 + 2 * x^3 + 2 * y^2 + 2 * x = x^2 + 1 →
  x + 2 * y^2 = 1/2 := by
  sorry

end x_plus_2y_squared_value_l3467_346795


namespace total_cars_count_l3467_346787

/-- Represents the number of cars counted by each person -/
structure CarCounts where
  jared : ℕ
  ann : ℕ
  alfred : ℕ
  bella : ℕ

/-- Calculates the total number of cars counted by all people -/
def total_count (counts : CarCounts) : ℕ :=
  counts.jared + counts.ann + counts.alfred + counts.bella

/-- Theorem stating the total count of cars after Alfred's recount -/
theorem total_cars_count (counts : CarCounts) :
  counts.jared = 300 ∧
  counts.ann = counts.jared + counts.jared * 15 / 100 ∧
  counts.alfred = counts.ann - 7 + (counts.ann - 7) * 12 / 100 ∧
  counts.bella = counts.jared + counts.jared * 20 / 100 ∧
  counts.bella = counts.alfred - counts.alfred * 10 / 100 →
  total_count counts = 1365 := by
  sorry

#eval total_count { jared := 300, ann := 345, alfred := 379, bella := 341 }

end total_cars_count_l3467_346787


namespace mary_marbles_left_l3467_346734

/-- The number of yellow marbles Mary has left after a series of exchanges -/
def marblesLeft (initial : ℝ) (giveJoan : ℝ) (receiveJoan : ℝ) (giveSam : ℝ) : ℝ :=
  initial - giveJoan + receiveJoan - giveSam

/-- Theorem stating that Mary will have 4.7 yellow marbles left -/
theorem mary_marbles_left :
  marblesLeft 9.5 2.3 1.1 3.6 = 4.7 := by
  sorry

end mary_marbles_left_l3467_346734


namespace expression_evaluation_l3467_346760

theorem expression_evaluation : -24 + 12 * (10 / 5) = 0 := by
  sorry

end expression_evaluation_l3467_346760


namespace boys_in_class_l3467_346758

/-- Given a class with a 4:3 ratio of girls to boys and 49 total students,
    prove that the number of boys is 21. -/
theorem boys_in_class (girls boys : ℕ) : 
  4 * boys = 3 * girls →  -- ratio of girls to boys is 4:3
  girls + boys = 49 →     -- total number of students is 49
  boys = 21 :=            -- prove that the number of boys is 21
by sorry

end boys_in_class_l3467_346758


namespace line_x_intercept_x_intercept_is_four_l3467_346700

/-- A line passing through two points (1, 3) and (5, -1) has x-intercept 4 -/
theorem line_x_intercept : ℝ → ℝ → Prop :=
  fun (slope : ℝ) (x_intercept : ℝ) =>
    (slope = ((-1) - 3) / (5 - 1)) ∧
    (3 = slope * (1 - x_intercept)) ∧
    (x_intercept = 4)

/-- The x-intercept of the line passing through (1, 3) and (5, -1) is 4 -/
theorem x_intercept_is_four : ∃ (slope : ℝ), line_x_intercept slope 4 := by
  sorry

end line_x_intercept_x_intercept_is_four_l3467_346700


namespace complex_sum_parts_zero_l3467_346757

theorem complex_sum_parts_zero (b : ℝ) : 
  let z : ℂ := 2 - b * I
  (z.re + z.im = 0) → b = 2 := by
  sorry

end complex_sum_parts_zero_l3467_346757


namespace outfit_choices_l3467_346750

/-- The number of color options for each item type -/
def num_colors : ℕ := 6

/-- The number of item types in an outfit -/
def num_items : ℕ := 4

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_colors ^ num_items

/-- The number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- The number of valid outfits (excluding those with all items of the same color) -/
def valid_outfits : ℕ := total_combinations - same_color_outfits

theorem outfit_choices :
  valid_outfits = 1290 :=
sorry

end outfit_choices_l3467_346750


namespace max_abs_sum_on_circle_l3467_346780

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ |x| + |y| ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 4 ∧ |x₀| + |y₀| = M :=
by sorry

end max_abs_sum_on_circle_l3467_346780


namespace largest_angle_of_triangle_l3467_346718

theorem largest_angle_of_triangle (y : ℝ) : 
  45 + 60 + y = 180 →
  max (max 45 60) y = 75 := by
sorry

end largest_angle_of_triangle_l3467_346718


namespace isosceles_triangle_area_l3467_346728

/-- An isosceles triangle with given altitude and perimeter has area 75 -/
theorem isosceles_triangle_area (base altitudeToBase equalSide : ℝ) : 
  base > 0 → 
  altitudeToBase > 0 → 
  equalSide > 0 → 
  altitudeToBase = 10 → 
  base + 2 * equalSide = 40 → 
  (1/2) * base * altitudeToBase = 75 := by
  sorry

end isosceles_triangle_area_l3467_346728


namespace right_and_obtuse_angles_in_clerts_l3467_346788

-- Define the number of clerts in a full Martian circle
def martian_full_circle : ℕ := 600

-- Define Earth angles in degrees
def earth_right_angle : ℕ := 90
def earth_obtuse_angle : ℕ := 135
def earth_full_circle : ℕ := 360

-- Define the conversion function from Earth degrees to Martian clerts
def earth_to_martian (earth_angle : ℕ) : ℕ :=
  (earth_angle * martian_full_circle) / earth_full_circle

-- Theorem statement
theorem right_and_obtuse_angles_in_clerts :
  earth_to_martian earth_right_angle = 150 ∧
  earth_to_martian earth_obtuse_angle = 225 := by
  sorry


end right_and_obtuse_angles_in_clerts_l3467_346788


namespace line_intercepts_sum_l3467_346736

/-- Given a line with equation y + 3 = -2(x + 5), 
    the sum of its x-intercept and y-intercept is -39/2 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y + 3 = -2*(x + 5)) → 
  (∃ x_int y_int : ℝ, 
    (y_int + 3 = -2*(x_int + 5)) ∧ 
    (0 + 3 = -2*(x_int + 5)) ∧ 
    (y_int + 3 = -2*(0 + 5)) ∧ 
    (x_int + y_int = -39/2)) :=
by sorry

end line_intercepts_sum_l3467_346736


namespace xy_product_range_l3467_346716

theorem xy_product_range (x y : ℝ) : 
  x^2 * y^2 + x^2 - 10*x*y - 8*x + 16 = 0 → 0 ≤ x*y ∧ x*y ≤ 10 := by
  sorry

end xy_product_range_l3467_346716


namespace reciprocal_of_negative_two_thirds_l3467_346786

theorem reciprocal_of_negative_two_thirds :
  ((-2 : ℚ) / 3)⁻¹ = -3 / 2 := by
  sorry

end reciprocal_of_negative_two_thirds_l3467_346786


namespace matrix_value_example_l3467_346779

def matrix_value (p q r s : ℤ) : ℤ := p * s - q * r

theorem matrix_value_example : matrix_value 4 5 2 3 = 2 := by sorry

end matrix_value_example_l3467_346779


namespace tan_alpha_value_l3467_346710

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.tan (2 * α) = Real.sin α / (2 + Real.cos α)) : 
  Real.tan α = -Real.sqrt 15 := by
sorry

end tan_alpha_value_l3467_346710


namespace bucket_weight_l3467_346766

/-- Given a bucket with unknown weight and unknown full water weight,
    if the total weight is p when it's three-quarters full and q when it's one-third full,
    then the total weight when it's completely full is (1/5)(8p - 3q). -/
theorem bucket_weight (p q : ℝ) : 
  (∃ (x y : ℝ), x + 3/4 * y = p ∧ x + 1/3 * y = q) → 
  (∃ (x y : ℝ), x + 3/4 * y = p ∧ x + 1/3 * y = q ∧ x + y = 1/5 * (8*p - 3*q)) :=
by sorry

end bucket_weight_l3467_346766


namespace complex_equation_solution_l3467_346709

theorem complex_equation_solution (z : ℂ) :
  (-3 + 4 * Complex.I) * z = 25 * Complex.I → z = 4 + 3 * Complex.I := by
sorry

end complex_equation_solution_l3467_346709


namespace rectangular_lot_area_l3467_346739

/-- Represents a rectangular lot with given properties -/
structure RectangularLot where
  width : ℝ
  length : ℝ
  length_constraint : length = 2 * width + 35
  perimeter_constraint : 2 * (width + length) = 850

/-- The area of a rectangular lot -/
def area (lot : RectangularLot) : ℝ := lot.width * lot.length

/-- Theorem stating that a rectangular lot with the given properties has an area of 38350 square feet -/
theorem rectangular_lot_area : 
  ∀ (lot : RectangularLot), area lot = 38350 := by
  sorry

end rectangular_lot_area_l3467_346739


namespace inequality_proof_l3467_346762

theorem inequality_proof (x : ℝ) : x > 4 → 3 * x + 5 < 5 * x - 3 := by
  sorry

end inequality_proof_l3467_346762


namespace degree_to_radian_90_l3467_346735

theorem degree_to_radian_90 : 
  (90 : ℝ) * (π / 180) = π / 2 := by sorry

end degree_to_radian_90_l3467_346735


namespace symmetry_about_y_axis_l3467_346769

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x
def g (x : ℝ) : ℝ := x^2 + 2*x

-- State the theorem
theorem symmetry_about_y_axis (x : ℝ) : 
  (∀ (y : ℝ), f x = y ↔ g (-x) = y) → g x = x^2 + 2*x :=
by sorry

end symmetry_about_y_axis_l3467_346769


namespace connie_marbles_l3467_346784

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 183

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 593

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem connie_marbles : initial_marbles = 776 := by
  sorry

end connie_marbles_l3467_346784


namespace parallel_vectors_x_value_l3467_346745

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (4, x) (-4, 4) → x = -4 := by
  sorry

end parallel_vectors_x_value_l3467_346745


namespace tan_eleven_pi_fourths_l3467_346771

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by
  sorry

end tan_eleven_pi_fourths_l3467_346771


namespace pet_shop_inventory_l3467_346708

/-- Represents the pet shop inventory problem --/
theorem pet_shop_inventory (num_kittens : ℕ) (puppy_cost kitten_cost total_value : ℕ) :
  num_kittens = 4 →
  puppy_cost = 20 →
  kitten_cost = 15 →
  total_value = 100 →
  ∃ (num_puppies : ℕ), num_puppies = 2 ∧ num_puppies * puppy_cost + num_kittens * kitten_cost = total_value :=
by
  sorry

end pet_shop_inventory_l3467_346708


namespace min_value_expression_l3467_346744

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) ≥ 3 ∧
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) = 3 ↔ y = x * Real.sqrt 2 :=
by sorry

end min_value_expression_l3467_346744


namespace prime_sum_product_l3467_346717

theorem prime_sum_product (p q : ℕ) : 
  Prime p → Prime q → p + q = 91 → p * q = 178 := by sorry

end prime_sum_product_l3467_346717


namespace line_parallel_to_plane_l3467_346721

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (m n : Line) (α β : Plane) 
  (h1 : ¬ (m = n)) -- m and n are non-overlapping
  (h2 : ¬ (α = β)) -- α and β are non-overlapping
  (h3 : intersect α β n) -- α intersects β at n
  (h4 : ¬ in_plane m α) -- m is not in α
  (h5 : parallel m n) -- m is parallel to n
  : parallel_plane_line α m := by sorry

end line_parallel_to_plane_l3467_346721


namespace table_sum_difference_l3467_346763

/-- Represents a cell in the N × N table -/
structure Cell (N : ℕ) where
  row : Fin N
  col : Fin N

/-- The rule for placing numbers in the table -/
def placeNumber (N : ℕ) (n : Fin (N^2)) : Cell N → Prop :=
  sorry

/-- The sum of numbers in a given column -/
def columnSum (N : ℕ) (col : Fin N) : ℕ :=
  sorry

/-- The sum of numbers in a given row -/
def rowSum (N : ℕ) (row : Fin N) : ℕ :=
  sorry

/-- The column containing N² -/
def lastColumn (N : ℕ) : Fin N :=
  sorry

/-- The row containing 1 -/
def firstRow (N : ℕ) : Fin N :=
  sorry

theorem table_sum_difference (N : ℕ) :
  columnSum N (lastColumn N) - rowSum N (firstRow N) = N^2 - N :=
sorry

end table_sum_difference_l3467_346763


namespace andy_stencils_l3467_346752

/-- Calculates the number of stencils painted given the following conditions:
  * Hourly wage
  * Pay per racquet strung
  * Pay per grommet change
  * Pay per stencil painted
  * Hours worked
  * Number of racquets strung
  * Number of grommet sets changed
  * Total earnings -/
def stencils_painted (hourly_wage : ℚ) (pay_per_racquet : ℚ) (pay_per_grommet : ℚ) 
  (pay_per_stencil : ℚ) (hours_worked : ℚ) (racquets_strung : ℕ) (grommets_changed : ℕ) 
  (total_earnings : ℚ) : ℕ :=
  sorry

theorem andy_stencils : 
  stencils_painted 9 15 10 1 8 7 2 202 = 5 :=
sorry

end andy_stencils_l3467_346752


namespace johns_allowance_l3467_346774

theorem johns_allowance (allowance : ℚ) : 
  (allowance * (2/5) * (2/3) = 64/100) → allowance = 24/10 := by
  sorry

end johns_allowance_l3467_346774


namespace evaluate_expression_l3467_346748

theorem evaluate_expression : 
  (30 ^ 20 : ℝ) / (90 ^ 10) = 10 ^ 10 := by
  sorry

#check evaluate_expression

end evaluate_expression_l3467_346748


namespace m_range_l3467_346773

theorem m_range (p q : Prop) (m : ℝ) 
  (hp : ∀ x : ℝ, 2*x - x^2 < m)
  (hq : m^2 - 2*m - 3 ≥ 0)
  (hnp : ¬(¬p))
  (hpq : ¬(p ∧ q)) :
  1 < m ∧ m < 3 := by
sorry

end m_range_l3467_346773


namespace travel_distance_calculation_l3467_346723

theorem travel_distance_calculation (total_distance sea_distance : ℕ) 
  (h1 : total_distance = 601)
  (h2 : sea_distance = 150) :
  total_distance - sea_distance = 451 :=
by sorry

end travel_distance_calculation_l3467_346723


namespace min_difference_f_g_l3467_346707

noncomputable def f (x : ℝ) := Real.exp x

noncomputable def g (x : ℝ) := Real.log (x / 2) + 1 / 2

theorem min_difference_f_g :
  ∀ a : ℝ, ∃ b : ℝ, b > 0 ∧ f a = g b ∧
  (∀ c : ℝ, c > 0 ∧ f a = g c → b - a ≤ c - a) ∧
  b - a = 2 + Real.log 2 :=
sorry

end min_difference_f_g_l3467_346707


namespace suzanna_textbooks_pages_l3467_346746

/-- Calculates the total number of pages in Suzanna's textbooks -/
def total_pages (history : ℕ) : ℕ :=
  let geography := history + 70
  let math := (history + geography) / 2
  let science := 2 * history
  let literature := history + geography - 30
  let economics := math + literature + 25
  history + geography + math + science + literature + economics

/-- Theorem stating that the total number of pages in Suzanna's textbooks is 1845 -/
theorem suzanna_textbooks_pages : total_pages 160 = 1845 := by
  sorry

end suzanna_textbooks_pages_l3467_346746


namespace hyperbolic_matrix_det_is_one_cosh_sq_sub_sinh_sq_l3467_346777

open Matrix Real

/-- The determinant of a specific 3x3 matrix involving hyperbolic functions is 1 -/
theorem hyperbolic_matrix_det_is_one (α β : ℝ) : 
  det !![cosh α * cosh β, cosh α * sinh β, -sinh α;
         -sinh β, cosh β, 0;
         sinh α * cosh β, sinh α * sinh β, cosh α] = 1 := by
  sorry

/-- The fundamental hyperbolic identity -/
theorem cosh_sq_sub_sinh_sq (x : ℝ) : cosh x * cosh x - sinh x * sinh x = 1 := by
  sorry

end hyperbolic_matrix_det_is_one_cosh_sq_sub_sinh_sq_l3467_346777


namespace range_of_a_range_of_m_l3467_346767

-- Define the sets A, B, C, and D
def A : Set ℝ := {x | x^2 + 3*x - 4 ≥ 0}
def B : Set ℝ := {x | (x-2)/x ≤ 0}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < 1+a}
def D (m : ℝ) : Set ℝ := {x | x^2 - (2*m+1/2)*x + m*(m+1/2) ≤ 0}

-- Part 1
theorem range_of_a :
  ∀ a : ℝ, (C a ⊆ (A ∩ B)) ↔ a ≥ 1/2 :=
sorry

-- Part 2
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ D m → x ∈ A ∩ B) ∧
           (∃ x : ℝ, x ∈ A ∩ B ∧ x ∉ D m) ↔
  1 ≤ m ∧ m ≤ 3/2 :=
sorry

end range_of_a_range_of_m_l3467_346767


namespace geometric_sequence_min_a3_l3467_346742

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_min_a3 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 - a 1 = 1 →
  (∀ b : ℕ → ℝ, is_geometric_sequence b → (∀ n : ℕ, b n > 0) → b 2 - b 1 = 1 → a 3 ≤ b 3) →
  ∀ n : ℕ, a n = 2^(n - 1) := by
sorry

end geometric_sequence_min_a3_l3467_346742


namespace parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l3467_346783

/-- The axis of symmetry of a parabola y = ax^2 + bx + c is the vertical line x = -b/(2a) -/
theorem parabola_axis_of_symmetry (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  ∀ x, f (-b / (2 * a) + x) = f (-b / (2 * a) - x) :=
by sorry

/-- The axis of symmetry of the parabola y = x^2 - 2x - 3 is the line x = 1 -/
theorem specific_parabola_axis_of_symmetry :
  let f : ℝ → ℝ := λ x => x^2 - 2 * x - 3
  ∀ x, f (1 + x) = f (1 - x) :=
by sorry

end parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l3467_346783


namespace cubic_equation_natural_roots_l3467_346789

theorem cubic_equation_natural_roots (p : ℝ) :
  (∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    5 * (x : ℝ)^3 - 5*(p+1)*(x : ℝ)^2 + (71*p-1)*(x : ℝ) + 1 = 66*p ∧
    5 * (y : ℝ)^3 - 5*(p+1)*(y : ℝ)^2 + (71*p-1)*(y : ℝ) + 1 = 66*p ∧
    5 * (z : ℝ)^3 - 5*(p+1)*(z : ℝ)^2 + (71*p-1)*(z : ℝ) + 1 = 66*p) ↔
  p = 76 := by
sorry

end cubic_equation_natural_roots_l3467_346789


namespace janets_crayons_l3467_346775

theorem janets_crayons (michelle_initial : ℕ) (michelle_final : ℕ) (janet_initial : ℕ) : 
  michelle_initial = 2 → 
  michelle_final = 4 → 
  michelle_final = michelle_initial + janet_initial → 
  janet_initial = 2 := by
sorry

end janets_crayons_l3467_346775


namespace circle_triangle_area_relation_l3467_346770

theorem circle_triangle_area_relation :
  ∀ (A B C : ℝ),
  (15 : ℝ)^2 + 20^2 = 25^2 →  -- Right triangle condition
  A > 0 ∧ B > 0 ∧ C > 0 →  -- Areas are positive
  C ≥ A ∧ C ≥ B →  -- C is the largest area
  A + B + (1/2 * 15 * 20) = (π * 25^2) / 8 →  -- Area relation
  A + B + 150 = C :=
by sorry

end circle_triangle_area_relation_l3467_346770


namespace roots_of_polynomials_l3467_346765

theorem roots_of_polynomials (r : ℝ) : 
  r^2 - 2*r - 1 = 0 → r^5 - 12*r^4 - 29*r - 12 = 0 := by
  sorry

#check roots_of_polynomials

end roots_of_polynomials_l3467_346765


namespace smallest_subtrahend_for_multiple_of_five_l3467_346798

theorem smallest_subtrahend_for_multiple_of_five :
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → m < n → ¬(∃ k : ℤ, 425 - m = 5 * k)) ∧ (∃ k : ℤ, 425 - n = 5 * k) :=
by sorry

end smallest_subtrahend_for_multiple_of_five_l3467_346798


namespace distance_between_points_l3467_346706

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (4, -6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 10 := by
  sorry

end distance_between_points_l3467_346706


namespace line_slope_is_two_l3467_346713

/-- A line in the xy-plane with y-intercept 2 and passing through (239, 480) has slope 2 -/
theorem line_slope_is_two :
  ∀ (m : ℝ) (f : ℝ → ℝ),
  (∀ x, f x = m * x + 2) →  -- Line equation with y-intercept 2
  f 239 = 480 →            -- Line passes through (239, 480)
  m = 2 :=                 -- Slope is 2
by
  sorry

end line_slope_is_two_l3467_346713


namespace thirty_in_base_6_l3467_346790

/-- Converts a decimal number to its base 6 representation -/
def to_base_6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Converts a list of digits in base 6 to a natural number -/
def from_base_6 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 6 + d) 0

theorem thirty_in_base_6 :
  to_base_6 30 = [5, 0] ∧ from_base_6 [5, 0] = 30 :=
sorry

end thirty_in_base_6_l3467_346790


namespace divisibility_theorem_l3467_346764

theorem divisibility_theorem (a : ℤ) : 
  (2 ∣ a^2 - a) ∧ (3 ∣ a^3 - a) := by
  sorry

end divisibility_theorem_l3467_346764


namespace polygon_area_l3467_346751

-- Define the polygon as a list of points
def polygon : List (ℕ × ℕ) := [(0, 0), (5, 0), (5, 2), (3, 2), (3, 3), (2, 3), (2, 2), (0, 2), (0, 0)]

-- Define a function to calculate the area of a polygon given its vertices
def calculatePolygonArea (vertices : List (ℕ × ℕ)) : ℕ := sorry

-- Theorem statement
theorem polygon_area : calculatePolygonArea polygon = 11 := by sorry

end polygon_area_l3467_346751


namespace win_by_fourth_round_prob_l3467_346724

/-- The probability of winning a single round in Rock, Paper, Scissors -/
def win_prob : ℚ := 1 / 3

/-- The number of rounds needed to win the game -/
def rounds_to_win : ℕ := 3

/-- The total number of rounds played -/
def total_rounds : ℕ := 4

/-- The probability of winning by the fourth round in a "best of five" Rock, Paper, Scissors game -/
theorem win_by_fourth_round_prob :
  (Nat.choose (total_rounds - 1) (rounds_to_win - 1) : ℚ) *
  win_prob ^ (rounds_to_win - 1) *
  (1 - win_prob) ^ (total_rounds - rounds_to_win) *
  win_prob = 2 / 27 := by
  sorry

end win_by_fourth_round_prob_l3467_346724


namespace existence_of_special_integers_l3467_346761

theorem existence_of_special_integers : ∃ (m n : ℤ), 
  (∃ (k₁ : ℤ), n^2 = k₁ * m) ∧
  (∃ (k₂ : ℤ), m^3 = k₂ * n^2) ∧
  (∃ (k₃ : ℤ), n^4 = k₃ * m^3) ∧
  (∃ (k₄ : ℤ), m^5 = k₄ * n^4) ∧
  (∀ (k₅ : ℤ), n^6 ≠ k₅ * m^5) ∧
  m = 32 ∧ n = 16 := by
sorry

end existence_of_special_integers_l3467_346761


namespace parallelogram_roots_l3467_346715

theorem parallelogram_roots (a : ℝ) : 
  (∃ z₁ z₂ z₃ z₄ : ℂ, 
    z₁^4 - 8*z₁^3 + 13*a*z₁^2 - 2*(3*a^2 + 2*a - 4)*z₁ - 2 = 0 ∧
    z₂^4 - 8*z₂^3 + 13*a*z₂^2 - 2*(3*a^2 + 2*a - 4)*z₂ - 2 = 0 ∧
    z₃^4 - 8*z₃^3 + 13*a*z₃^2 - 2*(3*a^2 + 2*a - 4)*z₃ - 2 = 0 ∧
    z₄^4 - 8*z₄^3 + 13*a*z₄^2 - 2*(3*a^2 + 2*a - 4)*z₄ - 2 = 0 ∧
    (z₁ + z₃ = z₂ + z₄) ∧ (z₁ - z₂ = z₄ - z₃)) ↔
  a^2 + (2/3)*a - 49*(1/3) = 0 :=
sorry

end parallelogram_roots_l3467_346715


namespace remaining_area_calculation_l3467_346712

theorem remaining_area_calculation : 
  let large_square_side : ℝ := 3
  let small_square_side : ℝ := 1
  let triangle_base : ℝ := 1
  let triangle_height : ℝ := 3
  large_square_side ^ 2 - (small_square_side ^ 2 + (triangle_base * triangle_height / 2)) = 6.5 := by
  sorry

end remaining_area_calculation_l3467_346712


namespace three_similar_points_l3467_346738

-- Define the trapezoid ABCD
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define a point P on AB
def PointOnAB (t : Trapezoid) (x : ℝ) : ℝ × ℝ :=
  (x * t.B.1 + (1 - x) * t.A.1, x * t.B.2 + (1 - x) * t.A.2)

-- Define the similarity condition
def IsSimilar (t : Trapezoid) (x : ℝ) : Prop :=
  let P := PointOnAB t x
  ∃ k : ℝ, k > 0 ∧
    (P.1 - t.A.1)^2 + (P.2 - t.A.2)^2 = k * ((t.C.1 - P.1)^2 + (t.C.2 - P.2)^2) ∧
    (t.D.1 - t.A.1)^2 + (t.D.2 - t.A.2)^2 = k * ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)

-- Theorem statement
theorem three_similar_points (t : Trapezoid) 
  (h1 : t.B.1 - t.A.1 = 7) 
  (h2 : t.D.2 - t.A.2 = 2) 
  (h3 : t.C.1 - t.B.1 = 3) 
  (h4 : t.A.2 = t.B.2) 
  (h5 : t.C.2 = t.D.2) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, 0 ≤ x ∧ x ≤ 1 ∧ IsSimilar t x :=
sorry

end three_similar_points_l3467_346738


namespace mutually_exclusive_not_opposite_mutually_exclusive_but_not_opposite_l3467_346772

/-- Represents the contents of a pencil case -/
structure PencilCase where
  pencils : ℕ
  pens : ℕ

/-- Represents the outcome of selecting two items -/
inductive Selection
  | TwoPencils
  | OnePencilOnePen
  | TwoPens

/-- Defines the pencil case with 2 pencils and 2 pens -/
def case : PencilCase := ⟨2, 2⟩

/-- Predicate for exactly one pen being selected -/
def exactlyOnePen (s : Selection) : Prop :=
  s = Selection.OnePencilOnePen

/-- Predicate for exactly two pencils being selected -/
def exactlyTwoPencils (s : Selection) : Prop :=
  s = Selection.TwoPencils

/-- Theorem stating that "Exactly 1 pen" and "Exactly 2 pencils" are mutually exclusive -/
theorem mutually_exclusive :
  ∀ s : Selection, ¬(exactlyOnePen s ∧ exactlyTwoPencils s) :=
sorry

/-- Theorem stating that "Exactly 1 pen" and "Exactly 2 pencils" are not opposite events -/
theorem not_opposite :
  ∃ s : Selection, ¬(exactlyOnePen s ∨ exactlyTwoPencils s) :=
sorry

/-- Main theorem combining the above results -/
theorem mutually_exclusive_but_not_opposite :
  (∀ s : Selection, ¬(exactlyOnePen s ∧ exactlyTwoPencils s)) ∧
  (∃ s : Selection, ¬(exactlyOnePen s ∨ exactlyTwoPencils s)) :=
sorry

end mutually_exclusive_not_opposite_mutually_exclusive_but_not_opposite_l3467_346772


namespace inductive_reasoning_not_comparison_l3467_346727

/-- Represents different types of reasoning --/
inductive ReasoningType
| Deductive
| Inductive
| Analogical
| Plausibility

/-- Represents the process of reasoning --/
structure Reasoning where
  type : ReasoningType
  process : String
  conclusion_certainty : Bool

/-- Definition of deductive reasoning --/
def deductive_reasoning : Reasoning :=
  { type := ReasoningType.Deductive,
    process := "from general to specific",
    conclusion_certainty := true }

/-- Definition of inductive reasoning --/
def inductive_reasoning : Reasoning :=
  { type := ReasoningType.Inductive,
    process := "from specific to general",
    conclusion_certainty := false }

/-- Definition of analogical reasoning --/
def analogical_reasoning : Reasoning :=
  { type := ReasoningType.Analogical,
    process := "comparing characteristics of different things",
    conclusion_certainty := false }

/-- Theorem stating that inductive reasoning is not about comparing characteristics of two types of things --/
theorem inductive_reasoning_not_comparison : 
  inductive_reasoning.process ≠ "reasoning between the characteristics of two types of things" := by
  sorry


end inductive_reasoning_not_comparison_l3467_346727


namespace system_solutions_correct_l3467_346799

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℝ, 3 * x + 2 * y = 10 ∧ x / 2 - (y + 1) / 3 = 1 ∧ x = 3 ∧ y = 1/2) ∧
  -- System 2
  (∃ x y : ℝ, 4 * x - 5 * y = 3 ∧ (x - 2 * y) / 0.4 = 0.6 ∧ x = 1.6 ∧ y = 0.68) :=
by sorry

end system_solutions_correct_l3467_346799


namespace f_is_cubic_l3467_346704

/-- A polynomial function of degree 4 -/
def f (a₀ a₁ a₂ a₃ a₄ : ℝ) (x : ℝ) : ℝ := a₀*x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄

/-- The function reaches its maximum at x = -1 -/
def max_at_neg_one (a₀ a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∀ x, f a₀ a₁ a₂ a₃ a₄ x ≤ f a₀ a₁ a₂ a₃ a₄ (-1)

/-- The graph of y = f(x + 1) is symmetric about (-1, 0) -/
def symmetric_about_neg_one (a₀ a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∀ x, f a₀ a₁ a₂ a₃ a₄ (x + 1) = f a₀ a₁ a₂ a₃ a₄ (-x + 1)

theorem f_is_cubic (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  max_at_neg_one a₀ a₁ a₂ a₃ a₄ →
  symmetric_about_neg_one a₀ a₁ a₂ a₃ a₄ →
  ∀ x, f a₀ a₁ a₂ a₃ a₄ x = x^3 - x :=
sorry

end f_is_cubic_l3467_346704


namespace complex_absolute_value_l3467_346722

/-- Given a complex number ω = 5 + 4i, prove that |ω^2 + 4ω + 41| = 2√2009 -/
theorem complex_absolute_value (ω : ℂ) (h : ω = 5 + 4 * I) : 
  Complex.abs (ω^2 + 4*ω + 41) = 2 * Real.sqrt 2009 := by
  sorry

end complex_absolute_value_l3467_346722


namespace similar_transformation_l3467_346792

structure Square where
  diagonal : ℝ

structure Transformation where
  area_after : ℝ
  is_similar : Bool

def original_square : Square := { diagonal := 2 }

def transformation : Transformation := { area_after := 4, is_similar := true }

theorem similar_transformation (s : Square) (t : Transformation) :
  s.diagonal = 2 ∧ t.area_after = 4 → t.is_similar = true := by
  sorry

end similar_transformation_l3467_346792


namespace pamela_skittles_l3467_346714

theorem pamela_skittles (initial_skittles : ℕ) (given_skittles : ℕ) : 
  initial_skittles = 50 → given_skittles = 7 → initial_skittles - given_skittles = 43 := by
sorry

end pamela_skittles_l3467_346714


namespace enemy_plane_hit_probability_l3467_346781

-- Define the probabilities of hitting for Person A and Person B
def prob_A_hits : ℝ := 0.6
def prob_B_hits : ℝ := 0.5

-- Define the event of the plane being hit
def plane_hit (prob_A prob_B : ℝ) : Prop :=
  1 - (1 - prob_A) * (1 - prob_B) = 0.8

-- State the theorem
theorem enemy_plane_hit_probability :
  plane_hit prob_A_hits prob_B_hits :=
by sorry

end enemy_plane_hit_probability_l3467_346781


namespace expressions_equality_l3467_346740

theorem expressions_equality (a b c : ℝ) :
  a + 2 * b * c = (a + b) * (a + 2 * c) ↔ a + b + 2 * c = 0 := by
  sorry

end expressions_equality_l3467_346740


namespace cube_less_than_self_l3467_346720

theorem cube_less_than_self (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : a^3 < a := by
  sorry

end cube_less_than_self_l3467_346720


namespace power_sum_unique_solution_l3467_346782

theorem power_sum_unique_solution (k : ℕ+) :
  (∃ (n : ℕ) (m : ℕ), m > 1 ∧ 3^k.val + 5^k.val = n^m) ↔ k = 1 := by
  sorry

end power_sum_unique_solution_l3467_346782


namespace certain_number_problem_l3467_346743

theorem certain_number_problem (x : ℝ) : 
  (0.90 * x = 0.50 * 1080) → x = 600 := by
  sorry

end certain_number_problem_l3467_346743


namespace sum_of_squares_l3467_346753

theorem sum_of_squares (a b c : ℝ) : 
  a * b + b * c + a * c = 131 → a + b + c = 19 → a^2 + b^2 + c^2 = 99 := by
sorry

end sum_of_squares_l3467_346753


namespace carnival_walk_distance_l3467_346759

def total_distance : Real := 0.75
def car_to_entrance : Real := 0.33
def entrance_to_rides : Real := 0.33

theorem carnival_walk_distance : 
  total_distance - (car_to_entrance + entrance_to_rides) = 0.09 := by
  sorry

end carnival_walk_distance_l3467_346759


namespace cylinder_plane_intersection_l3467_346768

/-- The equation of the curve formed by intersecting a cylinder with a plane -/
theorem cylinder_plane_intersection
  (r h : ℝ) -- radius and height of the cylinder
  (α : ℝ) -- angle between cutting plane and base plane
  (hr : r > 0)
  (hh : h > 0)
  (hα : 0 < α ∧ α < π/2) :
  ∃ f : ℝ → ℝ,
    (∀ x, 0 < x → x < 2*π*r →
      f x = r * Real.tan α * Real.sin (x/r - π/2)) ∧
    (∀ x, f x = 0 → (x = 0 ∨ x = 2*π*r)) :=
sorry

end cylinder_plane_intersection_l3467_346768


namespace three_propositions_l3467_346731

theorem three_propositions :
  (∀ a b : ℝ, |a - b| < 1 → |a| < |b| + 1) ∧
  (∀ a b : ℝ, |a + b| - 2 * |a| ≤ |a - b|) ∧
  (∀ x y : ℝ, |x| < 2 ∧ |y| > 3 → |x / y| < 2 / 3) := by
  sorry

end three_propositions_l3467_346731


namespace geometric_sequence_problem_l3467_346797

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 125 * r = b ∧ b * r = 60 / 49) → 
  b = 50 * Real.sqrt 3 / 7 := by
sorry

end geometric_sequence_problem_l3467_346797


namespace range_of_even_quadratic_l3467_346730

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - 2*a*x + b

-- Define the property of being even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem range_of_even_quadratic (a b : ℝ) :
  (∀ x ∈ Set.Icc (-2*b) (3*b - 1), f a b x ∈ Set.Icc 1 5) ∧
  is_even (f a b) →
  Set.range (f a b) = Set.Icc 1 5 :=
sorry

end range_of_even_quadratic_l3467_346730


namespace unique_triplet_l3467_346705

theorem unique_triplet : 
  ∃! (a b c : ℕ), 2 ≤ a ∧ a < b ∧ b < c ∧ 
  (((a - 1) * (b - 1) * (c - 1)) ∣ (a * b * c - 1)) ∧
  a = 4 ∧ b = 5 ∧ c = 6 := by
sorry

end unique_triplet_l3467_346705


namespace evaluate_expression_l3467_346793

theorem evaluate_expression (b : ℕ) (h : b = 2) : (b^3 * b^4) - 10 = 118 := by
  sorry

end evaluate_expression_l3467_346793


namespace cube_edge_length_l3467_346741

theorem cube_edge_length (volume : ℝ) (edge_length : ℝ) :
  volume = 2744 ∧ volume = edge_length ^ 3 → edge_length = 14 := by
  sorry

end cube_edge_length_l3467_346741


namespace zeros_of_derivative_form_arithmetic_progression_l3467_346794

/-- A fourth-degree polynomial whose zeros form an arithmetic progression -/
def ArithmeticZerosPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ (a r : ℝ) (α : ℝ), α ≠ 0 ∧ r > 0 ∧
    ∀ x, f x = α * (x - a) * (x - (a + r)) * (x - (a + 2*r)) * (x - (a + 3*r))

/-- The zeros of a polynomial form an arithmetic progression -/
def ZerosFormArithmeticProgression (f : ℝ → ℝ) : Prop :=
  ∃ (a d : ℝ), ∀ x, f x = 0 → ∃ n : ℕ, x = a + n * d

/-- The main theorem -/
theorem zeros_of_derivative_form_arithmetic_progression
  (f : ℝ → ℝ) (hf : ArithmeticZerosPolynomial f) :
  ZerosFormArithmeticProgression f' :=
sorry

end zeros_of_derivative_form_arithmetic_progression_l3467_346794
