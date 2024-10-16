import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_for_diophantine_equation_l1649_164933

theorem no_solution_for_diophantine_equation (d : ℤ) (h : d % 4 = 3) :
  ∀ (x y : ℕ), x^2 - d * y^2 ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_diophantine_equation_l1649_164933


namespace NUMINAMATH_CALUDE_initial_number_problem_l1649_164934

theorem initial_number_problem (x : ℝ) : 8 * x - 4 = 2.625 → x = 0.828125 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_problem_l1649_164934


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l1649_164916

theorem quadratic_factorization_sum (a w c d : ℝ) : 
  (∀ x, 6 * x^2 + x - 12 = (a * x + w) * (c * x + d)) →
  |a| + |w| + |c| + |d| = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l1649_164916


namespace NUMINAMATH_CALUDE_james_arthur_muffin_ratio_muffin_baking_problem_l1649_164940

theorem james_arthur_muffin_ratio : ℕ → ℕ → ℕ
  | arthur_muffins, james_muffins =>
    james_muffins / arthur_muffins

theorem muffin_baking_problem (arthur_muffins james_muffins : ℕ) 
  (h1 : arthur_muffins = 115)
  (h2 : james_muffins = 1380) :
  james_arthur_muffin_ratio arthur_muffins james_muffins = 12 := by
  sorry

end NUMINAMATH_CALUDE_james_arthur_muffin_ratio_muffin_baking_problem_l1649_164940


namespace NUMINAMATH_CALUDE_max_sum_solution_l1649_164950

theorem max_sum_solution : ∃ (a b : ℕ), 
  (2 * a * b + 3 * b = b^2 + 6 * a + 6) ∧ 
  (∀ (x y : ℕ), (2 * x * y + 3 * y = y^2 + 6 * x + 6) → (x + y ≤ a + b)) ∧
  a = 5 ∧ b = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_solution_l1649_164950


namespace NUMINAMATH_CALUDE_room_space_is_400_l1649_164973

/-- The total space of a room with bookshelves and reserved space for desk and walking -/
def room_space (num_shelves : ℕ) (shelf_space desk_space : ℝ) : ℝ :=
  num_shelves * shelf_space + desk_space

/-- Theorem: The room space is 400 square feet -/
theorem room_space_is_400 :
  room_space 3 80 160 = 400 :=
by sorry

end NUMINAMATH_CALUDE_room_space_is_400_l1649_164973


namespace NUMINAMATH_CALUDE_combined_girls_average_is_85_l1649_164920

/-- Represents the average scores and student counts for two high schools -/
structure SchoolData where
  adams_boys_avg : ℝ
  adams_girls_avg : ℝ
  adams_combined_avg : ℝ
  baker_boys_avg : ℝ
  baker_girls_avg : ℝ
  baker_combined_avg : ℝ
  combined_boys_avg : ℝ
  adams_boys_count : ℝ
  adams_girls_count : ℝ
  baker_boys_count : ℝ
  baker_girls_count : ℝ

/-- Theorem stating that the combined girls' average score for both schools is 85 -/
theorem combined_girls_average_is_85 (data : SchoolData)
  (h1 : data.adams_boys_avg = 72)
  (h2 : data.adams_girls_avg = 78)
  (h3 : data.adams_combined_avg = 75)
  (h4 : data.baker_boys_avg = 84)
  (h5 : data.baker_girls_avg = 91)
  (h6 : data.baker_combined_avg = 85)
  (h7 : data.combined_boys_avg = 80)
  (h8 : data.adams_boys_count = data.adams_girls_count)
  (h9 : data.baker_boys_count = 6 * data.baker_girls_count / 7)
  (h10 : data.adams_boys_count = data.baker_boys_count) :
  (data.adams_girls_avg * data.adams_girls_count + data.baker_girls_avg * data.baker_girls_count) /
  (data.adams_girls_count + data.baker_girls_count) = 85 := by
  sorry


end NUMINAMATH_CALUDE_combined_girls_average_is_85_l1649_164920


namespace NUMINAMATH_CALUDE_vector_product_l1649_164911

/-- Given vectors a and b, if |a| = 2 and a ⊥ b, then mn = -6 -/
theorem vector_product (m n : ℝ) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (2, n)
  (a.1^2 + a.2^2 = 4) → -- |a| = 2
  (a.1 * b.1 + a.2 * b.2 = 0) → -- a ⊥ b
  m * n = -6 := by
sorry

end NUMINAMATH_CALUDE_vector_product_l1649_164911


namespace NUMINAMATH_CALUDE_min_balls_theorem_l1649_164926

/-- Given a box with black and white balls, calculate the minimum number of balls to draw. -/
def min_balls_to_draw (black_balls white_balls : ℕ) : ℕ × ℕ :=
  (3, black_balls + 2)

/-- Theorem: For a box with 100 black and 100 white balls, the minimum number of balls
    to draw to ensure at least 2 of the same color is 3, and to ensure at least 2 white
    balls is 102. -/
theorem min_balls_theorem :
  let (same_color, two_white) := min_balls_to_draw 100 100
  same_color = 3 ∧ two_white = 102 := by sorry

end NUMINAMATH_CALUDE_min_balls_theorem_l1649_164926


namespace NUMINAMATH_CALUDE_jungsoos_number_is_420_75_l1649_164986

/-- Jinho's number is defined as the sum of 1 multiplied by 4, 0.1 multiplied by 2, and 0.001 multiplied by 7 -/
def jinhos_number : ℝ := 1 * 4 + 0.1 * 2 + 0.001 * 7

/-- Younghee's number is defined as 100 multiplied by Jinho's number -/
def younghees_number : ℝ := 100 * jinhos_number

/-- Jungsoo's number is defined as Younghee's number plus 0.05 -/
def jungsoos_number : ℝ := younghees_number + 0.05

/-- Theorem stating that Jungsoo's number equals 420.75 -/
theorem jungsoos_number_is_420_75 : jungsoos_number = 420.75 := by sorry

end NUMINAMATH_CALUDE_jungsoos_number_is_420_75_l1649_164986


namespace NUMINAMATH_CALUDE_elective_course_selection_l1649_164938

theorem elective_course_selection (type_A : ℕ) (type_B : ℕ) : 
  type_A = 4 → type_B = 3 → (type_A + type_B : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_elective_course_selection_l1649_164938


namespace NUMINAMATH_CALUDE_john_total_spent_l1649_164949

def silver_amount : ℝ := 1.5
def gold_amount : ℝ := 2 * silver_amount
def silver_price_per_ounce : ℝ := 20
def gold_price_per_ounce : ℝ := 50 * silver_price_per_ounce

def total_spent : ℝ := silver_amount * silver_price_per_ounce + gold_amount * gold_price_per_ounce

theorem john_total_spent :
  total_spent = 3030 := by sorry

end NUMINAMATH_CALUDE_john_total_spent_l1649_164949


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l1649_164957

theorem no_real_roots_quadratic (a : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + a ≠ 0) ↔ a > 9/4 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l1649_164957


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1649_164937

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x + a / x
  else if x < 0 then -(Real.log (-x) + a / (-x))
  else 0

-- Define the theorem
theorem possible_values_of_a (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) →  -- f is odd
  (∃ x₁ x₂ x₃ x₄ : ℝ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧  -- x₁ < x₂ < x₃ < x₄
    x₁ + x₄ = 0 ∧  -- x₁ + x₄ = 0
    ∃ r : ℝ,  -- Existence of common ratio r for geometric sequence
      f a x₂ = r * f a x₁ ∧
      f a x₃ = r * f a x₂ ∧
      f a x₄ = r * f a x₃ ∧
    ∃ d : ℝ,  -- Existence of common difference d for arithmetic sequence
      x₂ = x₁ + d ∧
      x₃ = x₂ + d ∧
      x₄ = x₃ + d) →
  a ≤ Real.sqrt 3 / (2 * Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1649_164937


namespace NUMINAMATH_CALUDE_price_difference_is_70_l1649_164990

-- Define the pricing structures and discount rates
def shop_x_base_price : ℝ := 1.25
def shop_y_base_price : ℝ := 2.75
def shop_x_discount_rate_80plus : ℝ := 0.10
def shop_y_bulk_price_80plus : ℝ := 2.00

-- Define the number of copies
def num_copies : ℕ := 80

-- Calculate the price for Shop X
def shop_x_price (copies : ℕ) : ℝ :=
  shop_x_base_price * copies * (1 - shop_x_discount_rate_80plus)

-- Calculate the price for Shop Y
def shop_y_price (copies : ℕ) : ℝ :=
  shop_y_bulk_price_80plus * copies

-- Theorem to prove
theorem price_difference_is_70 :
  shop_y_price num_copies - shop_x_price num_copies = 70 := by
  sorry


end NUMINAMATH_CALUDE_price_difference_is_70_l1649_164990


namespace NUMINAMATH_CALUDE_marbles_given_to_brother_l1649_164941

def initial_marbles : ℕ := 12
def current_marbles : ℕ := 7

theorem marbles_given_to_brother :
  initial_marbles - current_marbles = 5 := by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_brother_l1649_164941


namespace NUMINAMATH_CALUDE_probability_perfect_square_sum_l1649_164997

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def dice_sum_outcomes : ℕ := 64

def perfect_square_sums : List ℕ := [4, 9, 16]

def ways_to_get_sum (sum : ℕ) : ℕ :=
  if sum = 4 then 3
  else if sum = 9 then 8
  else if sum = 16 then 1
  else 0

def total_favorable_outcomes : ℕ :=
  (perfect_square_sums.map ways_to_get_sum).sum

theorem probability_perfect_square_sum :
  (total_favorable_outcomes : ℚ) / dice_sum_outcomes = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_perfect_square_sum_l1649_164997


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1649_164961

/-- An arithmetic sequence with common difference d -/
def ArithmeticSequence (d : ℝ) (a : ℕ → ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- The property that the sum of any two distinct terms is a term in the sequence -/
def SumPropertyHolds (a : ℕ → ℝ) : Prop :=
  ∀ s t, s ≠ t → ∃ k, a s + a t = a k

/-- The theorem stating the equivalence of the sum property and the existence of m -/
theorem arithmetic_sequence_sum_property (d : ℝ) (a : ℕ → ℝ) :
  ArithmeticSequence d a →
  (SumPropertyHolds a ↔ ∃ m : ℤ, m ≥ -1 ∧ a 1 = m * d) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1649_164961


namespace NUMINAMATH_CALUDE_cosine_equality_l1649_164974

theorem cosine_equality (n : ℤ) (hn : 0 ≤ n ∧ n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (331 * π / 180) → n = 29 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l1649_164974


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1649_164992

def M : Set ℝ := {-1, 1, 2}
def N : Set ℝ := {x : ℝ | x^2 - 5*x + 4 = 0}

theorem union_of_M_and_N : M ∪ N = {-1, 1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1649_164992


namespace NUMINAMATH_CALUDE_expand_expression_l1649_164913

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 5) = x^4 - 4*x^2 - 45 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1649_164913


namespace NUMINAMATH_CALUDE_power_five_mod_eighteen_l1649_164967

theorem power_five_mod_eighteen : 5^100 % 18 = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_eighteen_l1649_164967


namespace NUMINAMATH_CALUDE_theater_revenue_l1649_164914

theorem theater_revenue (total_seats : ℕ) (adult_price child_price : ℕ) (child_tickets : ℕ) :
  total_seats = 80 →
  adult_price = 12 →
  child_price = 5 →
  child_tickets = 63 →
  (total_seats = child_tickets + (total_seats - child_tickets)) →
  child_tickets * child_price + (total_seats - child_tickets) * adult_price = 519 := by
  sorry

end NUMINAMATH_CALUDE_theater_revenue_l1649_164914


namespace NUMINAMATH_CALUDE_nested_expression_sum_l1649_164919

def nested_expression : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * (1 + nested_expression n)

theorem nested_expression_sum : nested_expression 8 = 1022 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_sum_l1649_164919


namespace NUMINAMATH_CALUDE_semicircle_radius_l1649_164901

/-- The radius of a semi-circle given its perimeter -/
theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 108) :
  ∃ r : ℝ, r = perimeter / (Real.pi + 2) := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l1649_164901


namespace NUMINAMATH_CALUDE_toothpaste_problem_l1649_164989

/-- Represents the amount of toothpaste used by Anne's mom at each brushing -/
def moms_toothpaste_usage : ℝ := 2

/-- The problem statement -/
theorem toothpaste_problem (
  total_toothpaste : ℝ)
  (dads_usage : ℝ)
  (kids_usage : ℝ)
  (brushings_per_day : ℕ)
  (days_until_empty : ℕ)
  (h1 : total_toothpaste = 105)
  (h2 : dads_usage = 3)
  (h3 : kids_usage = 1)
  (h4 : brushings_per_day = 3)
  (h5 : days_until_empty = 5)
  : moms_toothpaste_usage * (brushings_per_day : ℝ) * days_until_empty +
    dads_usage * (brushings_per_day : ℝ) * days_until_empty +
    2 * kids_usage * (brushings_per_day : ℝ) * days_until_empty =
    total_toothpaste :=
by sorry

end NUMINAMATH_CALUDE_toothpaste_problem_l1649_164989


namespace NUMINAMATH_CALUDE_intersection_condition_l1649_164977

/-- Given a line y = kx + 2k and a circle x^2 + y^2 + mx + 4 = 0,
    if the line has at least one intersection point with the circle, then m > 4 -/
theorem intersection_condition (k m : ℝ) : 
  (∃ x y : ℝ, y = k * x + 2 * k ∧ x^2 + y^2 + m * x + 4 = 0) → m > 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l1649_164977


namespace NUMINAMATH_CALUDE_pond_length_l1649_164960

/-- Given a rectangular field with length 24 meters and width 12 meters, 
    containing a square pond whose area is 1/8 of the field's area,
    prove that the length of the pond is 6 meters. -/
theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_length : ℝ) : 
  field_length = 24 →
  field_width = 12 →
  field_length = 2 * field_width →
  pond_length^2 = (field_length * field_width) / 8 →
  pond_length = 6 :=
by sorry

end NUMINAMATH_CALUDE_pond_length_l1649_164960


namespace NUMINAMATH_CALUDE_surface_is_one_sheet_hyperboloid_l1649_164945

/-- The equation of the surface -/
def surface_equation (x y z : ℝ) : Prop :=
  x^2 - 2*x - 3*y^2 + 12*y + 2*z^2 + 12*z - 11 = 0

/-- The standard form of a one-sheet hyperboloid -/
def one_sheet_hyperboloid (a b c : ℝ) (x y z : ℝ) : Prop :=
  (x - a)^2 / 18 - (y - b)^2 / 6 + (z - c)^2 / 9 = 1

/-- Theorem stating that the surface equation represents a one-sheet hyperboloid -/
theorem surface_is_one_sheet_hyperboloid :
  ∀ x y z : ℝ, surface_equation x y z ↔ one_sheet_hyperboloid 1 2 (-3) x y z :=
by sorry

end NUMINAMATH_CALUDE_surface_is_one_sheet_hyperboloid_l1649_164945


namespace NUMINAMATH_CALUDE_pickle_problem_l1649_164968

/-- Pickle Problem -/
theorem pickle_problem (jars cucumbers initial_vinegar pickles_per_cucumber pickles_per_jar remaining_vinegar : ℕ)
  (h1 : jars = 4)
  (h2 : cucumbers = 10)
  (h3 : initial_vinegar = 100)
  (h4 : pickles_per_cucumber = 6)
  (h5 : pickles_per_jar = 12)
  (h6 : remaining_vinegar = 60) :
  (initial_vinegar - remaining_vinegar) / jars = 10 := by
  sorry


end NUMINAMATH_CALUDE_pickle_problem_l1649_164968


namespace NUMINAMATH_CALUDE_max_value_constraint_l1649_164958

theorem max_value_constraint (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 4) :
  (10 * x + 3 * y + 15 * z)^2 ≤ 3220 / 36 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1649_164958


namespace NUMINAMATH_CALUDE_election_votes_calculation_l1649_164943

theorem election_votes_calculation (total_votes : ℕ) :
  let valid_votes_percentage : ℚ := 85 / 100
  let candidate_a_percentage : ℚ := 75 / 100
  let candidate_a_votes : ℕ := 357000
  (↑candidate_a_votes : ℚ) = candidate_a_percentage * (valid_votes_percentage * ↑total_votes) →
  total_votes = 560000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l1649_164943


namespace NUMINAMATH_CALUDE_acute_angles_tangent_sum_l1649_164995

theorem acute_angles_tangent_sum (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  (1 + Real.tan α) * (1 + Real.tan β) = 2 →
  α + β = π/4 := by
sorry

end NUMINAMATH_CALUDE_acute_angles_tangent_sum_l1649_164995


namespace NUMINAMATH_CALUDE_factors_of_30_to_4th_l1649_164962

theorem factors_of_30_to_4th (h : 30 = 2 * 3 * 5) :
  (Finset.filter (fun d => d ≠ 1 ∧ d ≠ 30^4) (Nat.divisors (30^4))).card = 123 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_30_to_4th_l1649_164962


namespace NUMINAMATH_CALUDE_meaningful_fraction_l1649_164954

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 4)) ↔ x ≠ 4 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l1649_164954


namespace NUMINAMATH_CALUDE_max_value_sum_and_reciprocal_l1649_164972

theorem max_value_sum_and_reciprocal (nums : Finset ℝ) (x : ℝ) :
  (Finset.card nums = 11) →
  (∀ y ∈ nums, y > 0) →
  (x ∈ nums) →
  (Finset.sum nums id = 102) →
  (Finset.sum nums (λ y => 1 / y) = 102) →
  (x + 1 / x ≤ 10304 / 102) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_and_reciprocal_l1649_164972


namespace NUMINAMATH_CALUDE_largest_two_digit_multiple_of_3_and_5_l1649_164935

theorem largest_two_digit_multiple_of_3_and_5 : 
  ∃ n : ℕ, n = 90 ∧ 
  n ≥ 10 ∧ n < 100 ∧ 
  n % 3 = 0 ∧ n % 5 = 0 ∧
  ∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ m % 3 = 0 ∧ m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_multiple_of_3_and_5_l1649_164935


namespace NUMINAMATH_CALUDE_part_one_part_two_l1649_164976

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x - b * x^2

-- Part 1
theorem part_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, f a b x ≤ 1) → a ≤ 2 * Real.sqrt b :=
sorry

-- Part 2
theorem part_two (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  (∀ x ∈ Set.Icc 0 1, |f a b x| ≤ 1) ↔ (b - 1 ≤ a ∧ a ≤ 2 * Real.sqrt b) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1649_164976


namespace NUMINAMATH_CALUDE_range_of_f_on_I_l1649_164931

-- Define the function
def f (x : ℝ) : ℝ := -x^2 - 2*x + 3

-- Define the interval
def I : Set ℝ := {x | -5 ≤ x ∧ x ≤ 0}

-- State the theorem
theorem range_of_f_on_I :
  {y | ∃ x ∈ I, f x = y} = {y | -12 ≤ y ∧ y ≤ 4} := by sorry

end NUMINAMATH_CALUDE_range_of_f_on_I_l1649_164931


namespace NUMINAMATH_CALUDE_robot_trap_theorem_l1649_164904

theorem robot_trap_theorem (ε : ℝ) (hε : ε > 0) : 
  ∃ m l : ℕ+, |m.val * Real.sqrt 2 - l.val| < ε := by
sorry

end NUMINAMATH_CALUDE_robot_trap_theorem_l1649_164904


namespace NUMINAMATH_CALUDE_robotics_club_enrollment_l1649_164978

theorem robotics_club_enrollment (total : ℕ) (cs : ℕ) (elec : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : cs = 50)
  (h3 : elec = 35)
  (h4 : both = 25) :
  total - (cs + elec - both) = 20 :=
by sorry

end NUMINAMATH_CALUDE_robotics_club_enrollment_l1649_164978


namespace NUMINAMATH_CALUDE_circle_intersection_equation_l1649_164979

noncomputable def circle_equation (t : ℝ) (x y : ℝ) : Prop :=
  (x - t)^2 + (y - 2/t)^2 = t^2 + (2/t)^2

theorem circle_intersection_equation :
  ∀ t : ℝ,
  t ≠ 0 →
  circle_equation t 0 0 →
  (∃ a : ℝ, a ≠ 0 ∧ circle_equation t a 0) →
  (∃ b : ℝ, b ≠ 0 ∧ circle_equation t 0 b) →
  (∀ x y : ℝ, 2*x + y = 4 → circle_equation t x y → 
    ∃ m n : ℝ, circle_equation t m n ∧ 2*m + n = 4 ∧ m^2 + n^2 = x^2 + y^2) →
  circle_equation 2 x y ∧ (x - 2)^2 + (y - 1)^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_equation_l1649_164979


namespace NUMINAMATH_CALUDE_largest_coeff_is_correct_l1649_164929

/-- The coefficient of the term with the largest binomial coefficient in the expansion of (1/2 + 2x)^10 -/
def largest_coeff : ℕ := 252

/-- The binomial expansion of (1/2 + 2x)^10 -/
def expansion : ℕ → ℕ
| 0 => 1  -- Placeholder for the actual expansion
| n + 1 => n + 1  -- Placeholder for the actual expansion

theorem largest_coeff_is_correct :
  ∃ k, k ∈ Finset.range 11 ∧
    (∀ j ∈ Finset.range 11, Nat.choose 10 k ≥ Nat.choose 10 j) ∧
    expansion k = largest_coeff :=
by sorry

end NUMINAMATH_CALUDE_largest_coeff_is_correct_l1649_164929


namespace NUMINAMATH_CALUDE_not_all_data_has_regression_equation_l1649_164918

-- Define the basic concepts
def DataSet : Type := Set (ℝ × ℝ)
def RegressionEquation : Type := ℝ → ℝ

-- Define the properties mentioned in the problem
def hasCorrelation (d : DataSet) : Prop := sorry
def hasCausalRelationship (d : DataSet) : Prop := sorry
def canBeRepresentedByScatterPlot (d : DataSet) : Prop := sorry
def hasLinearCorrelation (d : DataSet) : Prop := sorry
def hasRegressionEquation (d : DataSet) : Prop := sorry

-- Define the statements from the problem
axiom correlation_not_causation : 
  ∀ d : DataSet, hasCorrelation d → ¬ (hasCausalRelationship d)

axiom scatter_plot_reflects_correlation : 
  ∀ d : DataSet, hasCorrelation d → canBeRepresentedByScatterPlot d

axiom regression_line_best_represents : 
  ∀ d : DataSet, hasLinearCorrelation d → hasRegressionEquation d

-- The theorem to be proved
theorem not_all_data_has_regression_equation :
  ¬ (∀ d : DataSet, hasRegressionEquation d) := by
  sorry

end NUMINAMATH_CALUDE_not_all_data_has_regression_equation_l1649_164918


namespace NUMINAMATH_CALUDE_min_value_expression_l1649_164915

theorem min_value_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1649_164915


namespace NUMINAMATH_CALUDE_multiple_of_six_is_multiple_of_three_l1649_164969

theorem multiple_of_six_is_multiple_of_three (n : ℤ) :
  (∃ k : ℤ, n = 6 * k) → (∃ m : ℤ, n = 3 * m) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_six_is_multiple_of_three_l1649_164969


namespace NUMINAMATH_CALUDE_circle_tangent_properties_l1649_164902

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y + 3 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-3, 0)

-- Define the tangent line l
def line_l (x y : ℝ) : Prop := ∃ k, y = k * (x + 3)

-- Theorem statement
theorem circle_tangent_properties :
  ∃ (center : ℝ × ℝ) (radius : ℝ) (y_intercept : ℝ),
    -- The center of M
    center = (-2, 1) ∧
    -- The radius of M
    radius = Real.sqrt 2 ∧
    -- The y-intercept of line l
    y_intercept = -3 ∧
    -- Line l is tangent to circle M at point P
    (∀ x y, circle_M x y → line_l x y → (x, y) = point_P) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_properties_l1649_164902


namespace NUMINAMATH_CALUDE_soccer_ball_distribution_l1649_164906

/-- The number of ways to distribute n identical balls into k boxes --/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute balls into numbered boxes with constraints --/
def distributeWithConstraints (totalBalls numBoxes : ℕ) : ℕ :=
  let remainingBalls := totalBalls - (numBoxes * (numBoxes + 1) / 2)
  distribute remainingBalls numBoxes

theorem soccer_ball_distribution :
  distributeWithConstraints 9 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_distribution_l1649_164906


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1649_164952

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 2*a^2 + 2*a - 3 = 0) → 
  (b^3 - 2*b^2 + 2*b - 3 = 0) → 
  (c^3 - 2*c^2 + 2*c - 3 = 0) → 
  a^3 + b^3 + c^3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1649_164952


namespace NUMINAMATH_CALUDE_combined_weight_is_1170_l1649_164964

/-- The weight Tony can lift in "the curl" exercise -/
def curl_weight : ℝ := 90

/-- The weight Tony can lift in "the military press" exercise -/
def military_press_weight : ℝ := 2 * curl_weight

/-- The weight Tony can lift in "the squat" exercise -/
def squat_weight : ℝ := 5 * military_press_weight

/-- The weight Tony can lift in "the bench press" exercise -/
def bench_press_weight : ℝ := 1.5 * military_press_weight

/-- The combined weight Tony can lift in the squat and bench press exercises -/
def combined_weight : ℝ := squat_weight + bench_press_weight

theorem combined_weight_is_1170 : combined_weight = 1170 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_is_1170_l1649_164964


namespace NUMINAMATH_CALUDE_twice_difference_l1649_164927

/-- Given two real numbers m and n, prove that 2(m-n) is equivalent to twice the difference between m and n -/
theorem twice_difference (m n : ℝ) : 2 * (m - n) = 2 * m - 2 * n := by
  sorry

end NUMINAMATH_CALUDE_twice_difference_l1649_164927


namespace NUMINAMATH_CALUDE_first_three_digits_after_decimal_l1649_164908

/-- The first three digits to the right of the decimal point in (2^10 + 1)^(4/3) are 320. -/
theorem first_three_digits_after_decimal (x : ℝ) : x = (2^10 + 1)^(4/3) →
  ∃ n : ℕ, x - ↑n = 0.320 + r ∧ 0 ≤ r ∧ r < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_first_three_digits_after_decimal_l1649_164908


namespace NUMINAMATH_CALUDE_horses_added_correct_horses_added_l1649_164983

theorem horses_added (initial_horses : ℕ) (drinking_water : ℕ) (bathing_water : ℕ) 
  (total_days : ℕ) (total_water : ℕ) : ℕ :=
  let water_per_horse := drinking_water + bathing_water
  let initial_daily_water := initial_horses * water_per_horse
  let initial_total_water := initial_daily_water * total_days
  let new_horses_water := total_water - initial_total_water
  let new_horses_daily_water := new_horses_water / total_days
  new_horses_daily_water / water_per_horse

theorem correct_horses_added :
  horses_added 3 5 2 28 1568 = 5 := by
  sorry

end NUMINAMATH_CALUDE_horses_added_correct_horses_added_l1649_164983


namespace NUMINAMATH_CALUDE_polygon_sides_count_l1649_164903

theorem polygon_sides_count (n : ℕ) (h : (n - 2) * 180 = 3 * 360) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l1649_164903


namespace NUMINAMATH_CALUDE_prob_more_ones_than_fives_five_dice_l1649_164996

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define a function to calculate the probability
def prob_more_ones_than_fives (n : ℕ) (s : ℕ) : ℚ :=
  190 / (s^n : ℚ)

-- Theorem statement
theorem prob_more_ones_than_fives_five_dice : 
  prob_more_ones_than_fives num_dice num_sides = 190 / 7776 := by
  sorry


end NUMINAMATH_CALUDE_prob_more_ones_than_fives_five_dice_l1649_164996


namespace NUMINAMATH_CALUDE_log_problem_l1649_164955

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_problem (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 1) :
  a = 2 ∧ 
  f a 1 = 0 ∧ 
  ∀ x > 0, f a x < 1 ↔ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l1649_164955


namespace NUMINAMATH_CALUDE_ice_cream_sales_theorem_l1649_164939

/-- The number of ice cream cones sold on Tuesday -/
def tuesday_sales : ℕ := 12000

/-- The number of ice cream cones sold on Wednesday -/
def wednesday_sales : ℕ := 2 * tuesday_sales

/-- The total number of ice cream cones sold on Tuesday and Wednesday -/
def total_sales : ℕ := tuesday_sales + wednesday_sales

theorem ice_cream_sales_theorem : total_sales = 36000 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_theorem_l1649_164939


namespace NUMINAMATH_CALUDE_initial_ratio_is_four_to_one_l1649_164925

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℝ
  water : ℝ

/-- The initial mixture before adding water -/
def initial_mixture : Mixture := sorry

/-- The final mixture after adding water -/
def final_mixture : Mixture := sorry

theorem initial_ratio_is_four_to_one :
  -- Initial mixture volume is 45 litres
  initial_mixture.milk + initial_mixture.water = 45 →
  -- 9 litres of water added
  final_mixture.water = initial_mixture.water + 9 →
  -- Final ratio of milk to water is 2:1
  final_mixture.milk / final_mixture.water = 2 →
  -- Prove that the initial ratio of milk to water was 4:1
  initial_mixture.milk / initial_mixture.water = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_is_four_to_one_l1649_164925


namespace NUMINAMATH_CALUDE_four_person_greeting_card_distribution_l1649_164924

def greeting_card_distribution (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * greeting_card_distribution (n - 1)

theorem four_person_greeting_card_distribution :
  greeting_card_distribution 4 = 9 :=
sorry

end NUMINAMATH_CALUDE_four_person_greeting_card_distribution_l1649_164924


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1649_164946

/-- Given a point and a line, this theorem proves that the equation
    x - 2y + 7 = 0 represents a line passing through the given point
    and perpendicular to the given line. -/
theorem perpendicular_line_equation (x y : ℝ) :
  let point : ℝ × ℝ := (-1, 3)
  let given_line := {(x, y) : ℝ × ℝ | 2 * x + y + 3 = 0}
  let perpendicular_line := {(x, y) : ℝ × ℝ | x - 2 * y + 7 = 0}
  (point ∈ perpendicular_line) ∧
  (∀ (v w : ℝ × ℝ), v ∈ given_line → w ∈ given_line → v ≠ w →
    let slope_given := (w.2 - v.2) / (w.1 - v.1)
    let slope_perp := (y - 3) / (x - (-1))
    slope_given * slope_perp = -1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1649_164946


namespace NUMINAMATH_CALUDE_equation_solution_l1649_164907

theorem equation_solution : ∃ x : ℝ, 6*x - 3*2*x - 2*3*x + 6 = 0 ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1649_164907


namespace NUMINAMATH_CALUDE_buses_passed_count_l1649_164991

/-- Represents the frequency of bus departures in minutes -/
def dallas_departure_frequency : ℕ := 60
def houston_departure_frequency : ℕ := 60

/-- Represents the offset of Houston departures from the hour in minutes -/
def houston_departure_offset : ℕ := 45

/-- Represents the trip duration in hours -/
def trip_duration : ℕ := 6

/-- Represents the number of Dallas-bound buses passed by a Houston-bound bus -/
def buses_passed : ℕ := 11

theorem buses_passed_count :
  buses_passed = 11 := by sorry

end NUMINAMATH_CALUDE_buses_passed_count_l1649_164991


namespace NUMINAMATH_CALUDE_pet_store_cages_l1649_164921

/-- Calculates the number of cages needed for a given number of animals and cage capacity -/
def cages_needed (animals : ℕ) (capacity : ℕ) : ℕ :=
  (animals + capacity - 1) / capacity

theorem pet_store_cages : 
  let initial_puppies : ℕ := 13
  let initial_kittens : ℕ := 10
  let initial_birds : ℕ := 15
  let sold_puppies : ℕ := 7
  let sold_kittens : ℕ := 4
  let sold_birds : ℕ := 5
  let puppy_capacity : ℕ := 2
  let kitten_capacity : ℕ := 3
  let bird_capacity : ℕ := 4
  let remaining_puppies := initial_puppies - sold_puppies
  let remaining_kittens := initial_kittens - sold_kittens
  let remaining_birds := initial_birds - sold_birds
  let total_cages := cages_needed remaining_puppies puppy_capacity + 
                     cages_needed remaining_kittens kitten_capacity + 
                     cages_needed remaining_birds bird_capacity
  total_cages = 8 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l1649_164921


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l1649_164912

/-- Given a right triangle with sides 15 and 20, similar to a larger triangle
    where one side is twice a rectangle's shorter side (30), 
    prove the perimeter of the larger triangle is 240. -/
theorem similar_triangle_perimeter : 
  ∀ (small_triangle large_triangle : Set ℝ) 
    (rectangle : Set (ℝ × ℝ)),
  (∃ a b c : ℝ, small_triangle = {a, b, c} ∧ 
    a = 15 ∧ b = 20 ∧ c^2 = a^2 + b^2) →
  (∃ x y : ℝ, rectangle = {(30, 60), (x, y)}) →
  (∃ d e f : ℝ, large_triangle = {d, e, f} ∧
    d = 2 * 30 ∧ 
    (d / 15 = e / 20 ∧ d / 15 = f / (15^2 + 20^2).sqrt)) →
  (∃ p : ℝ, p = d + e + f ∧ p = 240) :=
by sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l1649_164912


namespace NUMINAMATH_CALUDE_bird_ratio_l1649_164981

/-- Represents the number of birds caught by a cat during the day. -/
def birds_day : ℕ := 8

/-- Represents the total number of birds caught by a cat. -/
def birds_total : ℕ := 24

/-- Represents the number of birds caught by a cat at night. -/
def birds_night : ℕ := birds_total - birds_day

/-- The theorem states that the ratio of birds caught at night to birds caught during the day is 2:1. -/
theorem bird_ratio : birds_night / birds_day = 2 := by
  sorry

end NUMINAMATH_CALUDE_bird_ratio_l1649_164981


namespace NUMINAMATH_CALUDE_equation_solution_l1649_164965

theorem equation_solution (k : ℤ) : 
  let n : ℚ := -5 + 1024 * k
  (5/4) * n + 5/4 = n := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1649_164965


namespace NUMINAMATH_CALUDE_binary_110_equals_6_l1649_164982

def binary_to_decimal (b₂ b₁ b₀ : ℕ) : ℕ :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_110_equals_6 :
  binary_to_decimal 1 1 0 = 6 := by sorry

end NUMINAMATH_CALUDE_binary_110_equals_6_l1649_164982


namespace NUMINAMATH_CALUDE_circle_area_relation_l1649_164932

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circle_area_relation :
  ∀ (r_A r_B : ℝ),
  circle_area r_A = 9 →
  r_A = r_B / 2 →
  circle_area r_B = 36 := by
sorry

end NUMINAMATH_CALUDE_circle_area_relation_l1649_164932


namespace NUMINAMATH_CALUDE_focus_directrix_distance_l1649_164963

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Theorem statement
theorem focus_directrix_distance :
  let focus_y := 1 / 16
  let directrix_y := -1 / 16
  |focus_y - directrix_y| = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_focus_directrix_distance_l1649_164963


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1649_164971

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  totalPopulation : ℕ
  stratumPopulation : ℕ
  stratumSample : ℕ
  totalSample : ℕ

/-- The stratified sampling is proportional if the ratio of the stratum in the population
    equals the ratio of the stratum in the sample -/
def isProportional (s : StratifiedSample) : Prop :=
  s.stratumPopulation * s.totalSample = s.totalPopulation * s.stratumSample

theorem stratified_sample_size 
  (s : StratifiedSample) 
  (h1 : s.totalPopulation = 12000)
  (h2 : s.stratumPopulation = 3600)
  (h3 : s.stratumSample = 60)
  (h4 : isProportional s) :
  s.totalSample = 200 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l1649_164971


namespace NUMINAMATH_CALUDE_prob_no_consecutive_heads_10_l1649_164928

/-- Number of valid sequences without two consecutive heads for n coin tosses -/
def f (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | n + 2 => f (n + 1) + f n

/-- Probability of no two consecutive heads in n coin tosses -/
def prob_no_consecutive_heads (n : ℕ) : ℚ :=
  (f n : ℚ) / (2^n : ℚ)

/-- Theorem: The probability of no two heads appearing consecutively in 10 coin tosses is 9/64 -/
theorem prob_no_consecutive_heads_10 : prob_no_consecutive_heads 10 = 9/64 := by
  sorry

#eval prob_no_consecutive_heads 10

end NUMINAMATH_CALUDE_prob_no_consecutive_heads_10_l1649_164928


namespace NUMINAMATH_CALUDE_area_triangle_XMY_l1649_164985

/-- Triangle XMY with given dimensions --/
structure TriangleXMY where
  YM : ℝ
  MX : ℝ
  YZ : ℝ

/-- The area of triangle XMY is 3 square miles --/
theorem area_triangle_XMY (t : TriangleXMY) (h1 : t.YM = 2) (h2 : t.MX = 3) (h3 : t.YZ = 5) :
  (1 / 2) * t.YM * t.MX = 3 := by
  sorry


end NUMINAMATH_CALUDE_area_triangle_XMY_l1649_164985


namespace NUMINAMATH_CALUDE_value_of_b_l1649_164970

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 4) : b = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l1649_164970


namespace NUMINAMATH_CALUDE_green_blue_difference_l1649_164923

/-- Represents a hexagonal figure with blue and green tiles -/
structure HexFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Calculates the number of tiles needed for a two-layer border of a hexagon -/
def border_tiles : ℕ := 6 * 6

/-- Represents the new figure after adding a border -/
def new_figure (fig : HexFigure) : HexFigure :=
  { blue_tiles := fig.blue_tiles,
    green_tiles := fig.green_tiles + border_tiles }

/-- The main theorem to prove -/
theorem green_blue_difference (fig : HexFigure) 
  (h1 : fig.blue_tiles = 20) 
  (h2 : fig.green_tiles = 8) : 
  (new_figure fig).green_tiles - (new_figure fig).blue_tiles = 24 := by
  sorry

#check green_blue_difference

end NUMINAMATH_CALUDE_green_blue_difference_l1649_164923


namespace NUMINAMATH_CALUDE_triangle_ratio_l1649_164922

theorem triangle_ratio (a b c : ℝ) (A : ℝ) (S : ℝ) : 
  A = π/3 →  -- 60° in radians
  b = 1 → 
  S = Real.sqrt 3 → 
  S = (1/2) * b * c * Real.sin A → 
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A → 
  a / Real.sin A = 2 * Real.sqrt 39 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l1649_164922


namespace NUMINAMATH_CALUDE_y_coordinate_order_l1649_164988

-- Define the quadratic function
def f (x : ℝ) (b : ℝ) : ℝ := -x^2 + 2*x + b

-- Define the points A, B, C
def A (b : ℝ) : ℝ × ℝ := (4, f 4 b)
def B (b : ℝ) : ℝ × ℝ := (-1, f (-1) b)
def C (b : ℝ) : ℝ × ℝ := (1, f 1 b)

-- Theorem stating the order of y-coordinates
theorem y_coordinate_order (b : ℝ) :
  (A b).2 < (B b).2 ∧ (B b).2 < (C b).2 :=
by sorry

end NUMINAMATH_CALUDE_y_coordinate_order_l1649_164988


namespace NUMINAMATH_CALUDE_stall_owner_earnings_l1649_164942

/-- Represents the number of yellow balls in the bag -/
def yellow_balls : ℕ := 3

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 3

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := yellow_balls + white_balls

/-- Represents the number of balls drawn in each event -/
def balls_drawn : ℕ := 3

/-- Represents the probability of drawing 3 yellow balls -/
def prob_3_yellow : ℚ := 1 / 20

/-- Represents the probability of drawing 3 white balls -/
def prob_3_white : ℚ := 1 / 20

/-- Represents the probability of drawing balls of the same color -/
def prob_same_color : ℚ := prob_3_yellow + prob_3_white

/-- Represents the amount won when drawing 3 balls of the same color -/
def win_amount : ℤ := 10

/-- Represents the amount lost when drawing 3 balls of different colors -/
def loss_amount : ℤ := 2

/-- Represents the number of draws per day -/
def draws_per_day : ℕ := 80

/-- Represents the number of days in a month -/
def days_in_month : ℕ := 30

/-- Theorem: The stall owner's expected earnings in a month are $1920 -/
theorem stall_owner_earnings : 
  (draws_per_day * days_in_month * 
    (prob_same_color * win_amount - (1 - prob_same_color) * loss_amount)) = 1920 := by
  sorry

end NUMINAMATH_CALUDE_stall_owner_earnings_l1649_164942


namespace NUMINAMATH_CALUDE_students_left_l1649_164994

theorem students_left (initial_students new_students final_students : ℕ) :
  initial_students = 8 →
  new_students = 8 →
  final_students = 11 →
  initial_students + new_students - final_students = 5 := by
sorry

end NUMINAMATH_CALUDE_students_left_l1649_164994


namespace NUMINAMATH_CALUDE_total_weight_of_tickets_l1649_164953

-- Define the given conditions
def loose_boxes : ℕ := 9
def tickets_per_box : ℕ := 5
def weight_per_box : ℝ := 1.2
def boxes_per_case : ℕ := 10
def cases : ℕ := 2

-- Define the theorem
theorem total_weight_of_tickets :
  (loose_boxes + cases * boxes_per_case) * weight_per_box = 34.8 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_tickets_l1649_164953


namespace NUMINAMATH_CALUDE_broken_seashells_l1649_164948

/-- Given the total number of seashells and the number of unbroken seashells,
    calculate the number of broken seashells. -/
theorem broken_seashells (total : ℕ) (unbroken : ℕ) (h : unbroken ≤ total) :
  total - unbroken = total - unbroken :=
by sorry

end NUMINAMATH_CALUDE_broken_seashells_l1649_164948


namespace NUMINAMATH_CALUDE_six_coin_flip_probability_l1649_164987

theorem six_coin_flip_probability : 
  let n : ℕ := 6  -- number of coins
  let p : ℚ := 1 / 2  -- probability of heads for a fair coin
  2 * p^n = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_six_coin_flip_probability_l1649_164987


namespace NUMINAMATH_CALUDE_intersection_y_intercept_sum_l1649_164980

/-- Given two lines that intersect at a point, prove the sum of their y-intercepts. -/
theorem intersection_y_intercept_sum (a b : ℝ) : 
  (∃ x y : ℝ, x = (1/3)*y + a ∧ y = (1/3)*x + b ∧ x = 3 ∧ y = -1) →
  a + b = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_sum_l1649_164980


namespace NUMINAMATH_CALUDE_cafe_problem_l1649_164999

/-- The number of local offices that ordered sandwiches -/
def num_offices : ℕ := 3

/-- The number of sandwiches ordered by each office -/
def sandwiches_per_office : ℕ := 10

/-- The number of sandwiches ordered by each customer in half of the group -/
def sandwiches_per_customer : ℕ := 4

/-- The total number of sandwiches made by the café -/
def total_sandwiches : ℕ := 54

/-- The number of customers in the group that arrived at the café -/
def num_customers : ℕ := 12

theorem cafe_problem :
  num_offices * sandwiches_per_office +
  (num_customers / 2) * sandwiches_per_customer =
  total_sandwiches :=
by sorry

end NUMINAMATH_CALUDE_cafe_problem_l1649_164999


namespace NUMINAMATH_CALUDE_path_cost_calculation_l1649_164905

/-- Represents the dimensions and cost parameters of a field with a path around it. -/
structure FieldWithPath where
  field_length : ℝ
  field_width : ℝ
  path_width : ℝ
  path_area : ℝ
  cost_per_sqm : ℝ

/-- Calculates the total cost of constructing a path around a field. -/
def total_path_cost (f : FieldWithPath) : ℝ :=
  f.path_area * f.cost_per_sqm

/-- Theorem stating that the total cost of constructing the path is Rs. 3037.44. -/
theorem path_cost_calculation (f : FieldWithPath)
  (h1 : f.field_length = 75)
  (h2 : f.field_width = 55)
  (h3 : f.path_width = 2.8)
  (h4 : f.path_area = 1518.72)
  (h5 : f.cost_per_sqm = 2) :
  total_path_cost f = 3037.44 := by
  sorry

#check path_cost_calculation

end NUMINAMATH_CALUDE_path_cost_calculation_l1649_164905


namespace NUMINAMATH_CALUDE_complex_square_roots_l1649_164900

theorem complex_square_roots : 
  ∀ z : ℂ, z^2 = -99 - 40*I ↔ z = 2 - 10*I ∨ z = -2 + 10*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_roots_l1649_164900


namespace NUMINAMATH_CALUDE_contacts_per_dollar_theorem_l1649_164975

/-- Represents a box of contacts with quantity and price -/
structure ContactBox where
  quantity : ℕ
  price : ℚ

/-- Calculates the number of contacts per dollar for a given box -/
def contactsPerDollar (box : ContactBox) : ℚ :=
  box.quantity / box.price

/-- Theorem stating that the number of contacts equal to $1 worth in the box 
    with the lower cost per contact is 3 -/
theorem contacts_per_dollar_theorem (box1 box2 : ContactBox) 
  (h1 : box1.quantity = 50 ∧ box1.price = 25)
  (h2 : box2.quantity = 99 ∧ box2.price = 33) :
  let betterBox := if contactsPerDollar box1 > contactsPerDollar box2 then box1 else box2
  contactsPerDollar betterBox = 3 := by
  sorry

end NUMINAMATH_CALUDE_contacts_per_dollar_theorem_l1649_164975


namespace NUMINAMATH_CALUDE_swap_digits_theorem_l1649_164993

/-- Represents a two-digit number with digits a and b -/
structure TwoDigitNumber where
  a : ℕ
  b : ℕ
  a_less_than_ten : a < 10
  b_less_than_ten : b < 10

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : ℕ := 10 * n.a + n.b

/-- The value of a two-digit number with swapped digits -/
def TwoDigitNumber.swapped_value (n : TwoDigitNumber) : ℕ := 10 * n.b + n.a

/-- Theorem stating that swapping digits in a two-digit number results in 10b + a -/
theorem swap_digits_theorem (n : TwoDigitNumber) : 
  n.swapped_value = 10 * n.b + n.a := by sorry

end NUMINAMATH_CALUDE_swap_digits_theorem_l1649_164993


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1649_164944

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Theorem: In an arithmetic sequence, if a_4 + a_6 + a_8 + a_10 + a_12 = 120, then 2a_10 - a_12 = 24 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) : 
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1649_164944


namespace NUMINAMATH_CALUDE_equation_solutions_l1649_164951

theorem equation_solutions :
  (∀ x : ℝ, 2 * (2 * x - 1)^2 = 32 ↔ x = 5/2 ∨ x = -3/2) ∧
  (∀ x : ℝ, -x^2 + 2*x + 1 = 0 ↔ x = -1 - Real.sqrt 2 ∨ x = -1 + Real.sqrt 2) ∧
  (∀ x : ℝ, (x - 3)^2 + 2*x*(x - 3) = 0 ↔ x = 3 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1649_164951


namespace NUMINAMATH_CALUDE_password_probability_l1649_164998

/-- Represents the set of symbols used in the password -/
def SymbolSet : Finset Char := {'!', '@', '#', '$', '%'}

/-- Represents the set of favorable symbols -/
def FavorableSymbols : Finset Char := {'$', '%', '@'}

/-- Represents the set of two-digit numbers (00 to 99) -/
def TwoDigitNumbers : Finset Nat := Finset.range 100

/-- Represents the set of even two-digit numbers -/
def EvenTwoDigitNumbers : Finset Nat := TwoDigitNumbers.filter (fun n => n % 2 = 0)

/-- The probability of Alice's password meeting the specific criteria -/
theorem password_probability : 
  (EvenTwoDigitNumbers.card : ℚ) / TwoDigitNumbers.card * 
  (FavorableSymbols.card : ℚ) / SymbolSet.card * 
  (EvenTwoDigitNumbers.card : ℚ) / TwoDigitNumbers.card = 3 / 20 := by
  sorry


end NUMINAMATH_CALUDE_password_probability_l1649_164998


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1649_164966

/-- Represents the number of students in a stratum -/
structure Stratum where
  size : ℕ

/-- Represents a sample from a stratum -/
structure Sample where
  size : ℕ

/-- Calculates the total sample size for stratified sampling -/
def calculateTotalSampleSize (male : Stratum) (female : Stratum) (femaleSample : Sample) : ℕ :=
  (femaleSample.size * (male.size + female.size)) / female.size

/-- Theorem: Given the specified conditions, the total sample size is 176 -/
theorem stratified_sampling_theorem (male : Stratum) (female : Stratum) (femaleSample : Sample)
    (h1 : male.size = 1200)
    (h2 : female.size = 1000)
    (h3 : femaleSample.size = 80) :
    calculateTotalSampleSize male female femaleSample = 176 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1649_164966


namespace NUMINAMATH_CALUDE_sum_product_plus_one_positive_l1649_164947

theorem sum_product_plus_one_positive (a b c : ℝ) 
  (ha : abs a < 1) (hb : abs b < 1) (hc : abs c < 1) : 
  a * b + b * c + c * a + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_plus_one_positive_l1649_164947


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1649_164959

theorem smaller_number_proof (x y : ℝ) : 
  x + y = 70 ∧ y = 3 * x + 10 → x = 15 ∧ x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1649_164959


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l1649_164917

theorem quadratic_form_k_value :
  ∃ (a h : ℝ), ∀ x : ℝ, 9 * x^2 - 12 * x = a * (x - h)^2 - 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l1649_164917


namespace NUMINAMATH_CALUDE_function_properties_l1649_164930

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x - a * x^2 - Real.log x

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 - 2 * a * x - 1 / x

theorem function_properties (a : ℝ) (h_a : a > 0) :
  -- Part 1: If f'(1) = -2, then a = 1
  (f_derivative a 1 = -2) → a = 1 ∧
  -- Part 2: When a ≥ 1/8, f(x) is monotonically decreasing
  (a ≥ 1/8 → ∀ x > 0, f_derivative a x ≤ 0) :=
sorry

end

end NUMINAMATH_CALUDE_function_properties_l1649_164930


namespace NUMINAMATH_CALUDE_chord_length_l1649_164956

/-- The length of the chord intercepted by a line on a circle -/
theorem chord_length (t : ℝ) : 
  let line : ℝ → ℝ × ℝ := λ t => (1 + 2*t, 2 + t)
  let circle : ℝ × ℝ → Prop := λ p => p.1^2 + p.2^2 = 9
  let chord_length := 
    Real.sqrt (4 * (9 - (3 / Real.sqrt 5)^2))
  chord_length = 12/5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l1649_164956


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1649_164909

-- Define the conditions p and q
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := x ≥ 2

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬p is a sufficient but not necessary condition for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  ¬(∀ x : ℝ, not_q x → not_p x) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1649_164909


namespace NUMINAMATH_CALUDE_worker_d_rate_l1649_164910

-- Define work rates for workers a, b, c, and d
variable (A B C D : ℚ)

-- Define the conditions
def condition1 : Prop := A + B = 1 / 15
def condition2 : Prop := A + B + C = 1 / 12
def condition3 : Prop := C + D = 1 / 20

-- Theorem statement
theorem worker_d_rate 
  (h1 : condition1 A B) 
  (h2 : condition2 A B C) 
  (h3 : condition3 C D) : 
  D = 1 / 30 := by sorry

end NUMINAMATH_CALUDE_worker_d_rate_l1649_164910


namespace NUMINAMATH_CALUDE_period_length_divides_totient_l1649_164936

-- Define L(m) as the period length of the decimal expansion of 1/m
def L (m : ℕ) : ℕ := sorry

-- State the theorem
theorem period_length_divides_totient (m : ℕ) (h : Nat.gcd m 10 = 1) : 
  L m ∣ Nat.totient m := by sorry

end NUMINAMATH_CALUDE_period_length_divides_totient_l1649_164936


namespace NUMINAMATH_CALUDE_f_sum_symmetric_l1649_164984

/-- Given a function f(x) = ax^7 - bx^5 + cx^3 + 2, prove that f(5) + f(-5) = 4 -/
theorem f_sum_symmetric (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^7 - b * x^5 + c * x^3 + 2
  f 5 + f (-5) = 4 := by sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l1649_164984
