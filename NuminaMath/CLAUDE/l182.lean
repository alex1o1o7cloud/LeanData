import Mathlib

namespace sequence_properties_l182_18275

def sequence_a (n : ℕ+) : ℚ := (1 / 3) ^ n.val

def sum_S (n : ℕ+) : ℚ := (1 / 2) * (1 - (1 / 3) ^ n.val)

def arithmetic_sequence_condition (t : ℚ) : Prop :=
  let S₁ := sum_S 1
  let S₂ := sum_S 2
  let S₃ := sum_S 3
  S₁ + 3 * (S₂ + S₃) = 2 * (S₁ + S₂) * t

theorem sequence_properties :
  (∀ n : ℕ+, sum_S (n + 1) - sum_S n = (1 / 3) ^ (n + 1).val) →
  (∀ n : ℕ+, sequence_a n = (1 / 3) ^ n.val) ∧
  (∀ n : ℕ+, sum_S n = (1 / 2) * (1 - (1 / 3) ^ n.val)) ∧
  (∃ t : ℚ, arithmetic_sequence_condition t ∧ t = 2) := by
  sorry

end sequence_properties_l182_18275


namespace square_field_perimeter_l182_18232

theorem square_field_perimeter (a p : ℝ) (h1 : a ≥ 0) (h2 : p > 0) 
  (h3 : 6 * a = 6 * (2 * p + 9)) (h4 : a = a^2) : p = 36 := by
  sorry

end square_field_perimeter_l182_18232


namespace matrix_det_plus_five_l182_18274

theorem matrix_det_plus_five (M : Matrix (Fin 2) (Fin 2) ℤ) :
  M = ![![7, -2], ![-3, 6]] →
  M.det + 5 = 41 := by
sorry

end matrix_det_plus_five_l182_18274


namespace divisibility_by_nineteen_l182_18221

theorem divisibility_by_nineteen (k : ℕ) : 19 ∣ (2^(26*k + 2) + 3) := by
  sorry

end divisibility_by_nineteen_l182_18221


namespace x_greater_than_y_l182_18228

theorem x_greater_than_y (x y : ℝ) (h : y = (1 - 0.9444444444444444) * x) : 
  x = 18 * y := by sorry

end x_greater_than_y_l182_18228


namespace solution_set_f_gt_7_min_m2_plus_n2_l182_18225

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

-- Theorem for the solution set of f(x) > 7
theorem solution_set_f_gt_7 :
  {x : ℝ | f x > 7} = {x : ℝ | x > 4 ∨ x < -3} := by sorry

-- Theorem for the minimum value of m^2 + n^2 and the values of m and n
theorem min_m2_plus_n2 (m n : ℝ) (hn : n > 0) (h_min : ∀ x, f x ≥ m + n) :
  m^2 + n^2 ≥ 9/2 ∧ (m^2 + n^2 = 9/2 ↔ m = 3/2 ∧ n = 3/2) := by sorry

end solution_set_f_gt_7_min_m2_plus_n2_l182_18225


namespace odd_cube_plus_three_square_minus_linear_minus_three_divisible_by_48_l182_18231

theorem odd_cube_plus_three_square_minus_linear_minus_three_divisible_by_48 (x : ℤ) (h : ∃ k : ℤ, x = 2*k + 1) :
  ∃ m : ℤ, x^3 + 3*x^2 - x - 3 = 48*m := by
sorry

end odd_cube_plus_three_square_minus_linear_minus_three_divisible_by_48_l182_18231


namespace f_two_zeros_iff_a_in_range_l182_18285

-- Define the function f(x) = 2x³ - ax² + 1
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - a * x^2 + 1

-- Define the interval [1/2, 2]
def interval : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}

-- Define the condition of having exactly two zeros in the interval
def has_two_zeros (a : ℝ) : Prop :=
  ∃ x y, x ∈ interval ∧ y ∈ interval ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
  ∀ z, z ∈ interval ∧ f a z = 0 → z = x ∨ z = y

-- State the theorem
theorem f_two_zeros_iff_a_in_range :
  ∀ a : ℝ, has_two_zeros a ↔ 3/2 < a ∧ a ≤ 17/4 :=
sorry

end f_two_zeros_iff_a_in_range_l182_18285


namespace mrs_heine_purchase_l182_18235

/-- Calculates the total number of items purchased for dogs given the number of dogs,
    biscuits per dog, and boots per dog. -/
def total_items (num_dogs : ℕ) (biscuits_per_dog : ℕ) (boots_per_dog : ℕ) : ℕ :=
  num_dogs * (biscuits_per_dog + boots_per_dog)

/-- Proves that Mrs. Heine will buy 18 items in total for her dogs. -/
theorem mrs_heine_purchase : 
  let num_dogs : ℕ := 2
  let biscuits_per_dog : ℕ := 5
  let boots_per_set : ℕ := 4
  total_items num_dogs biscuits_per_dog boots_per_set = 18 := by
  sorry


end mrs_heine_purchase_l182_18235


namespace range_of_y_minus_2x_l182_18222

theorem range_of_y_minus_2x (x y : ℝ) 
  (hx : -2 ≤ x ∧ x ≤ 1) 
  (hy : 2 ≤ y ∧ y ≤ 4) : 
  0 ≤ y - 2*x ∧ y - 2*x ≤ 8 := by
  sorry

end range_of_y_minus_2x_l182_18222


namespace imaginary_product_real_part_l182_18217

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem imaginary_product_real_part (z : ℂ) (a b : ℝ) 
  (h1 : is_purely_imaginary z) 
  (h2 : (3 * Complex.I) * z = Complex.mk a b) : 
  a = -3 := by
  sorry

end imaginary_product_real_part_l182_18217


namespace percent_relation_l182_18233

theorem percent_relation (a b : ℝ) (h : a = 1.25 * b) : 4 * b = 3.2 * a := by
  sorry

end percent_relation_l182_18233


namespace quadratic_inequality_minimum_l182_18281

theorem quadratic_inequality_minimum (a b : ℝ) (h1 : Set.Icc 1 4 = {x : ℝ | x^2 - 5*a*x + b ≤ 0}) :
  let t (x y : ℝ) := a/x + b/y
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → t x y ≥ 9/2 :=
by
  sorry

end quadratic_inequality_minimum_l182_18281


namespace solve_equation_l182_18293

theorem solve_equation (y : ℝ) (h : (9 / y^2) = (y / 36)) : y = (324 : ℝ)^(1/3) := by
  sorry

end solve_equation_l182_18293


namespace slope_is_two_l182_18289

/-- A linear function y = kx + b where y increases by 6 when x increases by 3 -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  increase_property : ∀ (x : ℝ), (k * (x + 3) + b) - (k * x + b) = 6

/-- Theorem: The slope k of the linear function is equal to 2 -/
theorem slope_is_two (f : LinearFunction) : f.k = 2 := by
  sorry

end slope_is_two_l182_18289


namespace stock_price_after_two_years_l182_18262

/-- The final stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease)

/-- Theorem stating the final stock price after two years -/
theorem stock_price_after_two_years :
  final_stock_price 120 0.80 0.30 = 151.20 := by
  sorry


end stock_price_after_two_years_l182_18262


namespace factor_expression_l182_18252

theorem factor_expression (a : ℝ) : 53 * a^2 + 159 * a = 53 * a * (a + 3) := by
  sorry

end factor_expression_l182_18252


namespace pupils_like_only_maths_l182_18202

/-- Represents the number of pupils in various categories -/
structure ClassData where
  total : ℕ
  likesMaths : ℕ
  likesEnglish : ℕ
  likesBoth : ℕ
  likesNeither : ℕ

/-- The main theorem stating the number of pupils who like only Maths -/
theorem pupils_like_only_maths (c : ClassData) : 
  c.total = 30 ∧ 
  c.likesMaths = 20 ∧ 
  c.likesEnglish = 18 ∧ 
  c.likesBoth = 2 * c.likesNeither →
  c.likesMaths - c.likesBoth = 4 := by
  sorry


end pupils_like_only_maths_l182_18202


namespace hexagon_shell_arrangements_l182_18279

/-- The number of rotational symmetries in a regular hexagon -/
def hexagon_rotations : ℕ := 6

/-- The number of distinct points on the hexagon (corners and midpoints) -/
def total_points : ℕ := 12

/-- The number of distinct sea shells -/
def total_shells : ℕ := 12

/-- The number of distinct arrangements of sea shells on a regular hexagon,
    considering only rotational equivalence -/
def distinct_arrangements : ℕ := (Nat.factorial total_shells) / hexagon_rotations

theorem hexagon_shell_arrangements :
  distinct_arrangements = 79833600 := by
  sorry

end hexagon_shell_arrangements_l182_18279


namespace distribution_methods_l182_18261

/-- Represents the number of ways to distribute books to students -/
def distribute_books (novels : ℕ) (picture_books : ℕ) (students : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of distribution methods -/
theorem distribution_methods :
  distribute_books 2 2 3 = 12 :=
by sorry

end distribution_methods_l182_18261


namespace arithmetic_sequence_sum_l182_18239

def is_arithmetic (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 6 = 32 →
  a 4 + a 5 = 55 := by
sorry

end arithmetic_sequence_sum_l182_18239


namespace sequence_nature_l182_18282

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem sequence_nature (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n^2) →
  (∀ n, a n = S n - S (n-1)) →
  arithmetic_sequence a 2 :=
sorry

end sequence_nature_l182_18282


namespace sin_theta_value_l182_18276

theorem sin_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 4 * Real.cos θ) 
  (h2 : 0 < θ) 
  (h3 : θ < Real.pi) : 
  Real.sin θ = (-5 + Real.sqrt 41) / 4 := by
sorry

end sin_theta_value_l182_18276


namespace contrapositive_of_true_implication_l182_18206

theorem contrapositive_of_true_implication (h : ∀ x : ℝ, x < 0 → x^2 > 0) :
  ∀ x : ℝ, x^2 ≤ 0 → x ≥ 0 := by
  sorry

end contrapositive_of_true_implication_l182_18206


namespace fruit_salad_red_grapes_l182_18278

theorem fruit_salad_red_grapes (green_grapes : ℕ) : 
  let red_grapes := 3 * green_grapes + 7
  let raspberries := green_grapes - 5
  green_grapes + red_grapes + raspberries = 102 →
  red_grapes = 67 := by
  sorry

end fruit_salad_red_grapes_l182_18278


namespace tangent_line_length_l182_18201

/-- The length of a tangent line from a point to a circle --/
theorem tangent_line_length 
  (l : ℝ → ℝ → Prop) 
  (C : ℝ → ℝ → Prop) 
  (a : ℝ) :
  (∀ x y, l x y ↔ x + a * y - 1 = 0) →
  (∀ x y, C x y ↔ x^2 + y^2 - 4*x - 2*y + 1 = 0) →
  (∀ x y, l x y → C x y → x = 2 ∧ y = 1) →
  l (-4) a →
  ∃ B : ℝ × ℝ, C B.1 B.2 ∧ 
    (B.1 + 4)^2 + (B.2 + 1)^2 = 36 :=
by sorry

end tangent_line_length_l182_18201


namespace faster_train_speed_l182_18267

/-- Proves the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed (train_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  crossing_time = 8 →
  let relative_speed := 2 * train_length / crossing_time
  let slower_speed := relative_speed / 3
  let faster_speed := 2 * slower_speed
  faster_speed = 50 / 3 := by
  sorry

end faster_train_speed_l182_18267


namespace initial_cards_count_l182_18204

/-- The initial number of baseball cards Fred had -/
def initial_cards : ℕ := sorry

/-- The number of baseball cards Keith bought -/
def cards_bought : ℕ := 22

/-- The number of baseball cards Fred has left -/
def cards_left : ℕ := 18

/-- Theorem stating that the initial number of cards is 40 -/
theorem initial_cards_count : initial_cards = 40 := by sorry

end initial_cards_count_l182_18204


namespace quadratic_solution_implies_value_l182_18249

theorem quadratic_solution_implies_value (a b : ℝ) : 
  (1 : ℝ)^2 + a * 1 + 2 * b = 0 → 2023 - a - 2 * b = 2024 := by
  sorry

end quadratic_solution_implies_value_l182_18249


namespace card_distribution_events_l182_18251

structure Card where
  color : String
  deriving Repr

structure Person where
  name : String
  deriving Repr

def distribute_cards (cards : List Card) (people : List Person) : List (Person × Card) :=
  sorry

def event_A_red (distribution : List (Person × Card)) : Prop :=
  sorry

def event_B_red (distribution : List (Person × Card)) : Prop :=
  sorry

def mutually_exclusive (event1 event2 : List (Person × Card) → Prop) : Prop :=
  sorry

def opposite_events (event1 event2 : List (Person × Card) → Prop) : Prop :=
  sorry

theorem card_distribution_events :
  let cards := [Card.mk "red", Card.mk "black", Card.mk "blue", Card.mk "white"]
  let people := [Person.mk "A", Person.mk "B", Person.mk "C", Person.mk "D"]
  let distributions := distribute_cards cards people
  mutually_exclusive event_A_red event_B_red ∧
  ¬(opposite_events event_A_red event_B_red) :=
by
  sorry

end card_distribution_events_l182_18251


namespace unique_digit_multiplication_l182_18237

theorem unique_digit_multiplication (B : ℕ) : 
  (B < 10) →                           -- B is a single digit
  (B2 : ℕ) →                           -- B2 is a natural number
  (B2 = 10 * B + 2) →                  -- B2 is a two-digit number ending in 2
  (7 * B < 100) →                      -- 7B is a two-digit number
  (B2 * (70 + B) = 6396) →             -- The multiplication equation
  (B = 8) := by sorry

end unique_digit_multiplication_l182_18237


namespace polynomial_simplification_l182_18291

theorem polynomial_simplification (x : ℝ) :
  (10 * x^3 - 30 * x^2 + 40 * x - 5) - (3 * x^3 - 7 * x^2 - 5 * x + 10) =
  7 * x^3 - 23 * x^2 + 45 * x - 15 := by
  sorry

end polynomial_simplification_l182_18291


namespace complex_number_location_l182_18294

theorem complex_number_location (z : ℂ) (h : (1 - Complex.I) * z = Complex.I ^ 2013) :
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end complex_number_location_l182_18294


namespace bumper_car_line_count_l182_18260

/-- The number of people waiting in line for bumper cars after changes -/
def total_people_waiting (initial1 initial2 initial3 left1 left2 left3 joined1 joined2 joined3 : ℕ) : ℕ :=
  (initial1 - left1 + joined1) + (initial2 - left2 + joined2) + (initial3 - left3 + joined3)

/-- Theorem stating the total number of people waiting in line for bumper cars after changes -/
theorem bumper_car_line_count : 
  total_people_waiting 7 12 15 4 3 5 8 10 7 = 47 := by
  sorry

end bumper_car_line_count_l182_18260


namespace circle_area_l182_18241

/-- The area of the circle defined by the equation 3x^2 + 3y^2 - 12x + 9y + 27 = 0 is equal to 61π/4 -/
theorem circle_area (x y : ℝ) : 
  (3 * x^2 + 3 * y^2 - 12 * x + 9 * y + 27 = 0) → 
  (∃ (center : ℝ × ℝ) (radius : ℝ), 
    ((x - center.1)^2 + (y - center.2)^2 = radius^2) ∧ 
    (π * radius^2 = 61 * π / 4)) :=
by sorry

end circle_area_l182_18241


namespace rectangle_area_increase_rectangle_area_increase_percentage_l182_18263

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.3 * l) * (1.2 * w) = 1.56 * (l * w) := by
  sorry

theorem rectangle_area_increase_percentage (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  ((1.3 * l) * (1.2 * w) - l * w) / (l * w) = 0.56 := by
  sorry

end rectangle_area_increase_rectangle_area_increase_percentage_l182_18263


namespace relationship_depends_on_b_relationship_only_b_l182_18265

theorem relationship_depends_on_b (a b : ℝ) : 
  (a + b) - (a - b) = 2 * b :=
sorry

theorem relationship_only_b (a b : ℝ) : 
  (a + b > a - b ↔ b > 0) ∧
  (a + b < a - b ↔ b < 0) ∧
  (a + b = a - b ↔ b = 0) :=
sorry

end relationship_depends_on_b_relationship_only_b_l182_18265


namespace sequence_term_l182_18259

/-- The sum of the first n terms of a sequence -/
def S (n : ℕ) : ℤ := n^2 - 3*n

/-- The nth term of the sequence -/
def a (n : ℕ) : ℤ := 2*n - 4

theorem sequence_term (n : ℕ) (h : n ≥ 1) : 
  a n = S n - S (n-1) :=
sorry

end sequence_term_l182_18259


namespace magnitude_of_z_to_fourth_l182_18214

-- Define the complex number
def z : ℂ := 4 - 3 * Complex.I

-- State the theorem
theorem magnitude_of_z_to_fourth : Complex.abs (z^4) = 625 := by
  sorry

end magnitude_of_z_to_fourth_l182_18214


namespace fixed_point_of_exponential_function_l182_18257

/-- Given a > 0 and a ≠ 1, the function f(x) = a^(x+1) + 1 always passes through the point (-1, 2) -/
theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) + 1
  f (-1) = 2 := by
  sorry

end fixed_point_of_exponential_function_l182_18257


namespace calculate_mixed_fraction_expression_l182_18223

theorem calculate_mixed_fraction_expression : 
  (47 * ((2 + 2/3) - (3 + 1/4))) / ((3 + 1/2) + (2 + 1/5)) = -(4 + 25/38) := by
  sorry

end calculate_mixed_fraction_expression_l182_18223


namespace bead_removal_proof_l182_18213

theorem bead_removal_proof (total_beads : ℕ) (parts : ℕ) (final_beads : ℕ) (x : ℕ) : 
  total_beads = 39 →
  parts = 3 →
  final_beads = 6 →
  2 * ((total_beads / parts) - x) = final_beads →
  x = 10 := by
sorry

end bead_removal_proof_l182_18213


namespace emily_small_gardens_l182_18255

def number_of_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

theorem emily_small_gardens :
  number_of_small_gardens 42 36 2 = 3 := by
  sorry

end emily_small_gardens_l182_18255


namespace triangle_angle_sine_cosine_equivalence_l182_18211

theorem triangle_angle_sine_cosine_equivalence (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) :
  (Real.sin A > Real.sin B) ↔ (Real.cos A + Real.cos (A + C) < 0) := by
  sorry

end triangle_angle_sine_cosine_equivalence_l182_18211


namespace unique_valid_n_l182_18246

/-- The set of numbers {1, 16, 27} -/
def S : Finset ℕ := {1, 16, 27}

/-- Condition: The product of any two distinct members of S increased by 9 is a perfect square -/
axiom distinct_product_square (a b : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hab : a ≠ b) :
  ∃ k : ℕ, a * b + 9 = k^2

/-- Definition: n is a positive integer for which n+9, 16n+9, and 27n+9 are perfect squares -/
def is_valid (n : ℕ) : Prop :=
  n > 0 ∧ 
  (∃ k : ℕ, n + 9 = k^2) ∧ 
  (∃ l : ℕ, 16 * n + 9 = l^2) ∧ 
  (∃ m : ℕ, 27 * n + 9 = m^2)

/-- Theorem: 280 is the unique positive integer satisfying the conditions -/
theorem unique_valid_n : 
  is_valid 280 ∧ ∀ n : ℕ, is_valid n → n = 280 :=
by sorry

end unique_valid_n_l182_18246


namespace count_special_numbers_is_360_l182_18229

/-- A function that counts 4-digit numbers beginning with 2 and having exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits : Finset ℕ := Finset.range 10
  let non_two_digits : Finset ℕ := digits.erase 2

  let case1 := 3 * non_two_digits.card * (non_two_digits.card - 1)
  let case2 := 3 * non_two_digits.card * (non_two_digits.card - 1)

  case1 + case2

/-- Theorem stating that the count of special numbers is 360 -/
theorem count_special_numbers_is_360 : count_special_numbers = 360 := by
  sorry

end count_special_numbers_is_360_l182_18229


namespace anne_distance_l182_18290

/-- Calculates the distance traveled given time and speed -/
def distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Proves that wandering for 5 hours at 4 miles per hour results in a distance of 20 miles -/
theorem anne_distance : distance 5 4 = 20 := by
  sorry

end anne_distance_l182_18290


namespace brian_read_75_chapters_l182_18272

/-- The total number of chapters Brian read -/
def total_chapters : ℕ :=
  let book1 : ℕ := 20
  let book2 : ℕ := 15
  let book3 : ℕ := 15
  let first_three : ℕ := book1 + book2 + book3
  let book4 : ℕ := first_three / 2
  book1 + book2 + book3 + book4

/-- Proof that Brian read 75 chapters in total -/
theorem brian_read_75_chapters : total_chapters = 75 := by
  sorry

end brian_read_75_chapters_l182_18272


namespace equation_has_two_solutions_l182_18244

-- Define the equation
def equation (x : ℝ) : Prop := |x - 2| = |x - 4| + |x - 6|

-- Define the set of solutions
def solution_set : Set ℝ := {x : ℝ | equation x}

-- Theorem statement
theorem equation_has_two_solutions : 
  ∃ (a b : ℝ), a ≠ b ∧ solution_set = {a, b} :=
sorry

end equation_has_two_solutions_l182_18244


namespace trapezoid_segment_length_l182_18236

/-- Given a trapezoid ABCD where:
    1. The ratio of the area of triangle ABC to the area of triangle ADC is 5:2
    2. AB + CD = 280 cm
    Prove that AB = 200 cm -/
theorem trapezoid_segment_length (AB CD : ℝ) (h : ℝ) : 
  (AB * h / 2) / (CD * h / 2) = 5 / 2 →
  AB + CD = 280 →
  AB = 200 := by
sorry

end trapezoid_segment_length_l182_18236


namespace three_students_same_group_l182_18286

/-- The number of students in the school -/
def total_students : ℕ := 900

/-- The number of lunch groups -/
def num_groups : ℕ := 4

/-- The size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- The probability of three specific students being in the same lunch group -/
def prob_same_group : ℚ := 1 / 16

theorem three_students_same_group :
  let n := total_students
  let k := num_groups
  let g := group_size
  prob_same_group = (g / n) * ((g - 1) / (n - 1)) * ((g - 2) / (n - 2)) :=
sorry

end three_students_same_group_l182_18286


namespace prob_10_or_9_prob_less_than_7_l182_18271

-- Define the probabilities
def p_10 : ℝ := 0.21
def p_9 : ℝ := 0.23
def p_8 : ℝ := 0.25
def p_7 : ℝ := 0.28

-- Theorem for the first question
theorem prob_10_or_9 : p_10 + p_9 = 0.44 := by sorry

-- Theorem for the second question
theorem prob_less_than_7 : 1 - (p_10 + p_9 + p_8 + p_7) = 0.03 := by sorry

end prob_10_or_9_prob_less_than_7_l182_18271


namespace quadratic_constant_term_l182_18264

theorem quadratic_constant_term (m : ℝ) : 
  (∀ x, (m - 3) * x^2 - 3 * x + m^2 = 9) → m = -3 :=
by
  sorry

end quadratic_constant_term_l182_18264


namespace tv_monthly_payment_l182_18297

/-- Calculates the monthly payment for a discounted television purchase with installments -/
theorem tv_monthly_payment 
  (original_price : ℝ) 
  (discount_rate : ℝ) 
  (first_installment : ℝ) 
  (num_installments : ℕ) 
  (h1 : original_price = 480) 
  (h2 : discount_rate = 0.05) 
  (h3 : first_installment = 150) 
  (h4 : num_installments = 3) : 
  ∃ (monthly_payment : ℝ), 
    monthly_payment = (original_price * (1 - discount_rate) - first_installment) / num_installments ∧ 
    monthly_payment = 102 := by
  sorry

end tv_monthly_payment_l182_18297


namespace go_stones_count_l182_18250

/-- Calculates the total number of go stones given the number of stones per bundle,
    the number of bundles of black stones, and the number of white stones. -/
def total_go_stones (stones_per_bundle : ℕ) (black_bundles : ℕ) (white_stones : ℕ) : ℕ :=
  stones_per_bundle * black_bundles + white_stones

/-- Proves that the total number of go stones is 46 given the specified conditions. -/
theorem go_stones_count : total_go_stones 10 3 16 = 46 := by
  sorry

end go_stones_count_l182_18250


namespace largest_digit_divisible_by_six_l182_18243

/-- The function that constructs the number 5678N from a single digit N -/
def constructNumber (N : ℕ) : ℕ := 5678 * 10 + N

/-- Predicate to check if a natural number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := n < 10

/-- Theorem stating that 4 is the largest single-digit number N such that 5678N is divisible by 6 -/
theorem largest_digit_divisible_by_six :
  ∀ N : ℕ, isSingleDigit N → N > 4 → ¬(constructNumber N % 6 = 0) :=
by sorry

end largest_digit_divisible_by_six_l182_18243


namespace room_width_l182_18248

/-- Given a rectangular room with length 21 m, surrounded by a 2 m wide veranda on all sides,
    and the veranda area is 148 m², prove that the width of the room is 12 m. -/
theorem room_width (room_length : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_length = 21 →
  veranda_width = 2 →
  veranda_area = 148 →
  ∃ (room_width : ℝ),
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) -
    room_length * room_width = veranda_area ∧
    room_width = 12 := by
  sorry

end room_width_l182_18248


namespace deposit_percentage_l182_18256

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) : 
  deposit = 55 → remaining = 495 → (deposit / (deposit + remaining)) * 100 = 10 := by
  sorry

end deposit_percentage_l182_18256


namespace hyperbola_equation_l182_18270

/-- The equation of a hyperbola sharing foci with a given ellipse and passing through a specific point -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (a b : ℝ), (x^2 / 9 + y^2 / 5 = 1) ∧ 
   (x^2 / a^2 - y^2 / b^2 = 1) ∧
   (3^2 / a^2 - 2 / b^2 = 1) ∧
   (a^2 + b^2 = 4)) →
  (x^2 / 3 - y^2 = 1) :=
by sorry

end hyperbola_equation_l182_18270


namespace fraction_problem_l182_18254

theorem fraction_problem : ∃ x : ℚ, (65 / 100 * 40 : ℚ) = x * 25 + 6 ∧ x = 4 / 5 := by
  sorry

end fraction_problem_l182_18254


namespace johns_initial_contribution_l182_18284

theorem johns_initial_contribution 
  (total_initial : ℝ) 
  (total_final : ℝ) 
  (john_initial : ℝ) 
  (kelly_initial : ℝ) 
  (luke_initial : ℝ) 
  (h1 : total_initial = 1200)
  (h2 : total_final = 1800)
  (h3 : total_initial = john_initial + kelly_initial + luke_initial)
  (h4 : total_final = (john_initial - 200) + 3 * kelly_initial + 3 * luke_initial) :
  john_initial = 800 := by
sorry

end johns_initial_contribution_l182_18284


namespace line_A2A3_tangent_to_circle_M_l182_18224

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = x

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point on the parabola
def point_on_parabola (A : ℝ × ℝ) : Prop := parabola_C A.1 A.2

-- Define a line tangent to the circle
def line_tangent_to_circle (A B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), circle_M (A.1 + t * (B.1 - A.1)) (A.2 + t * (B.2 - A.2))

-- Main theorem
theorem line_A2A3_tangent_to_circle_M (A₁ A₂ A₃ : ℝ × ℝ) :
  point_on_parabola A₁ →
  point_on_parabola A₂ →
  point_on_parabola A₃ →
  line_tangent_to_circle A₁ A₂ →
  line_tangent_to_circle A₁ A₃ →
  line_tangent_to_circle A₂ A₃ :=
sorry

end line_A2A3_tangent_to_circle_M_l182_18224


namespace insects_eaten_by_geckos_and_lizards_l182_18283

/-- The number of insects eaten by geckos and lizards -/
def total_insects_eaten (num_geckos : ℕ) (insects_per_gecko : ℕ) (num_lizards : ℕ) : ℕ :=
  num_geckos * insects_per_gecko + num_lizards * (2 * insects_per_gecko)

/-- Theorem stating the total number of insects eaten in the given scenario -/
theorem insects_eaten_by_geckos_and_lizards :
  total_insects_eaten 5 6 3 = 66 := by
  sorry


end insects_eaten_by_geckos_and_lizards_l182_18283


namespace gas_volume_at_20C_l182_18268

/-- Represents the volume of a gas at a given temperature -/
structure GasVolume where
  temp : ℝ  -- temperature in Celsius
  vol : ℝ   -- volume in cubic centimeters

/-- Represents the relationship between temperature change and volume change -/
structure VolumeChange where
  temp_change : ℝ  -- temperature change in Celsius
  vol_change : ℝ   -- volume change in cubic centimeters

theorem gas_volume_at_20C 
  (initial : GasVolume)
  (change : VolumeChange)
  (h1 : initial.temp = 30)
  (h2 : initial.vol = 36)
  (h3 : change.temp_change = 2)
  (h4 : change.vol_change = 3) :
  ∃ (final : GasVolume), 
    final.temp = 20 ∧ 
    final.vol = 21 :=
sorry

end gas_volume_at_20C_l182_18268


namespace final_result_l182_18287

def loop_calculation (i : ℕ) (S : ℕ) : ℕ :=
  if i < 9 then S else loop_calculation (i - 1) (S * i)

theorem final_result :
  loop_calculation 11 1 = 990 := by
  sorry

end final_result_l182_18287


namespace divisibility_by_three_l182_18227

theorem divisibility_by_three (d : Nat) : 
  d ≤ 9 → (15780 + d) % 3 = 0 ↔ d = 0 ∨ d = 3 ∨ d = 6 ∨ d = 9 := by
  sorry

end divisibility_by_three_l182_18227


namespace geometry_problem_l182_18203

-- Define the points
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 2)
def P : ℝ × ℝ := (2, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the line parallel to AB passing through P
def line_parallel_AB_through_P (x y : ℝ) : Prop :=
  x + 2*y - 8 = 0

-- Define the circumscribed circle of triangle OAB
def circle_OAB (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 5

-- Theorem statement
theorem geometry_problem :
  (∀ x y : ℝ, line_parallel_AB_through_P x y ↔ 
    (y - P.2 = ((B.2 - A.2) / (B.1 - A.1)) * (x - P.1))) ∧
  (∀ x y : ℝ, circle_OAB x y ↔ 
    ((x - ((A.1 + B.1) / 2))^2 + (y - ((A.2 + B.2) / 2))^2 = 
     ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4)) :=
sorry

end geometry_problem_l182_18203


namespace shirt_price_proof_l182_18212

-- Define the original prices
def original_shirt_price : ℝ := 60
def original_jacket_price : ℝ := 90

-- Define the reduction rate
def reduction_rate : ℝ := 0.2

-- Define the number of items bought
def num_shirts : ℕ := 5
def num_jackets : ℕ := 10

-- Define the total cost after reduction
def total_cost : ℝ := 960

-- Theorem statement
theorem shirt_price_proof :
  (1 - reduction_rate) * (num_shirts * original_shirt_price + num_jackets * original_jacket_price) = total_cost :=
by sorry

end shirt_price_proof_l182_18212


namespace tan_half_angle_second_quadrant_l182_18253

/-- If α is an angle in the second quadrant and 3sinα + 4cosα = 0, then tan(α/2) = 2 -/
theorem tan_half_angle_second_quadrant (α : Real) : 
  π/2 < α ∧ α < π → -- α is in the second quadrant
  3 * Real.sin α + 4 * Real.cos α = 0 → -- given equation
  Real.tan (α/2) = 2 := by sorry

end tan_half_angle_second_quadrant_l182_18253


namespace last_three_sum_l182_18292

theorem last_three_sum (a : Fin 7 → ℝ) 
  (h1 : (a 0 + a 1 + a 2 + a 3) / 4 = 13)
  (h2 : (a 3 + a 4 + a 5 + a 6) / 4 = 15)
  (h3 : a 3 ^ 2 = a 6)
  (h4 : a 6 = 25) :
  a 4 + a 5 + a 6 = 55 := by
sorry

end last_three_sum_l182_18292


namespace absolute_value_inequality_l182_18269

theorem absolute_value_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end absolute_value_inequality_l182_18269


namespace optimal_zongzi_purchase_l182_18266

/-- Represents the unit price and quantity of zongzi --/
structure Zongzi where
  unit_price : ℝ
  quantity : ℕ

/-- Represents the shopping mall's zongzi purchase plan --/
structure ZongziPurchasePlan where
  zongzi_a : Zongzi
  zongzi_b : Zongzi

/-- Defines the conditions of the zongzi purchase problem --/
def zongzi_problem (plan : ZongziPurchasePlan) : Prop :=
  let a := plan.zongzi_a
  let b := plan.zongzi_b
  (3000 / a.unit_price - 3360 / b.unit_price = 40) ∧
  (b.unit_price = 1.2 * a.unit_price) ∧
  (a.quantity + b.quantity = 2200) ∧
  (a.unit_price * a.quantity ≤ b.unit_price * b.quantity)

/-- Theorem stating the optimal solution to the zongzi purchase problem --/
theorem optimal_zongzi_purchase :
  ∃ (plan : ZongziPurchasePlan),
    zongzi_problem plan ∧
    plan.zongzi_a.unit_price = 5 ∧
    plan.zongzi_b.unit_price = 6 ∧
    plan.zongzi_a.quantity = 1200 ∧
    plan.zongzi_b.quantity = 1000 ∧
    plan.zongzi_a.unit_price * plan.zongzi_a.quantity +
    plan.zongzi_b.unit_price * plan.zongzi_b.quantity = 12000 :=
  sorry

end optimal_zongzi_purchase_l182_18266


namespace card_count_proof_l182_18230

/-- The ratio of Xiao Ming's counting speed to Xiao Hua's -/
def speed_ratio : ℚ := 6 / 4

/-- The number of cards Xiao Hua counted before forgetting -/
def forgot_count : ℕ := 48

/-- The number of cards Xiao Hua counted after starting over -/
def final_count : ℕ := 112

/-- The number of cards left in the box after Xiao Hua's final count -/
def remaining_cards : ℕ := 1

/-- The original number of cards in the box -/
def original_cards : ℕ := 353

theorem card_count_proof :
  (speed_ratio * forgot_count).num.toNat + final_count + remaining_cards = original_cards :=
sorry

end card_count_proof_l182_18230


namespace avocados_for_guacamole_l182_18209

/-- The number of avocados needed for one serving of guacamole -/
def avocados_per_serving (initial_avocados sister_avocados total_servings : ℕ) : ℕ :=
  (initial_avocados + sister_avocados) / total_servings

/-- Theorem stating that 3 avocados are needed for one serving of guacamole -/
theorem avocados_for_guacamole :
  avocados_per_serving 5 4 3 = 3 := by
  sorry

end avocados_for_guacamole_l182_18209


namespace certain_number_theorem_l182_18258

theorem certain_number_theorem (a x : ℕ) (h1 : a = 105) (h2 : a^3 = x * 25 * 45 * 49) : x = 21 := by
  sorry

end certain_number_theorem_l182_18258


namespace investment_problem_l182_18296

theorem investment_problem (x : ℝ) : 
  x > 0 ∧ 
  0.07 * x + 0.27 * 1500 = 0.22 * (x + 1500) → 
  x = 500 := by
sorry

end investment_problem_l182_18296


namespace contest_team_mistakes_l182_18200

/-- The number of incorrect answers for a team in a math contest -/
def team_incorrect_answers (total_questions : ℕ) (riley_mistakes : ℕ) (ofelia_correct_offset : ℕ) : ℕ :=
  let riley_correct := total_questions - riley_mistakes
  let ofelia_correct := riley_correct / 2 + ofelia_correct_offset
  let ofelia_mistakes := total_questions - ofelia_correct
  riley_mistakes + ofelia_mistakes

/-- Theorem stating the number of incorrect answers for Riley and Ofelia's team -/
theorem contest_team_mistakes :
  team_incorrect_answers 35 3 5 = 17 := by
  sorry

end contest_team_mistakes_l182_18200


namespace sport_water_amount_l182_18288

/-- Represents the ratio of ingredients in a flavored drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- Standard formulation of the flavored drink -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

/-- Sport formulation of the flavored drink -/
def sport_ratio : DrinkRatio :=
  { flavoring := standard_ratio.flavoring,
    corn_syrup := standard_ratio.corn_syrup / 3,
    water := standard_ratio.water * 2 }

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 5

/-- Theorem: The amount of water in the sport formulation is 75 ounces -/
theorem sport_water_amount :
  (sport_ratio.water / sport_ratio.flavoring) * (sport_corn_syrup / sport_ratio.corn_syrup) * sport_ratio.flavoring = 75 := by
  sorry

end sport_water_amount_l182_18288


namespace pet_store_puppies_l182_18245

theorem pet_store_puppies (initial_birds initial_puppies initial_cats initial_spiders : ℕ)
  (sold_birds adopted_puppies loose_spiders : ℕ) (final_total : ℕ) :
  initial_birds = 12 →
  initial_cats = 5 →
  initial_spiders = 15 →
  sold_birds = initial_birds / 2 →
  adopted_puppies = 3 →
  loose_spiders = 7 →
  final_total = 25 →
  final_total = initial_birds - sold_birds + initial_cats + 
                (initial_spiders - loose_spiders) + (initial_puppies - adopted_puppies) →
  initial_puppies = 9 :=
by sorry

end pet_store_puppies_l182_18245


namespace fifth_power_prime_solution_l182_18238

theorem fifth_power_prime_solution :
  ∀ (x y p : ℕ+),
  (x^2 + y) * (y^2 + x) = p^5 ∧ Nat.Prime p.val →
  ((x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2)) :=
sorry

end fifth_power_prime_solution_l182_18238


namespace parallelogram_height_l182_18218

theorem parallelogram_height (area base height : ℝ) : 
  area = 231 ∧ base = 21 ∧ area = base * height → height = 11 := by
  sorry

end parallelogram_height_l182_18218


namespace brown_hat_fraction_l182_18240

theorem brown_hat_fraction (H : ℝ) (H_pos : H > 0) : ∃ B : ℝ,
  B > 0 ∧ B < 1 ∧
  (1/5 * B * H) / (1/3 * H) = 0.15 ∧
  B = 1/4 := by
sorry

end brown_hat_fraction_l182_18240


namespace problem_statement_l182_18247

theorem problem_statement (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b)^2016 = 1 := by
  sorry

end problem_statement_l182_18247


namespace roden_fish_count_l182_18205

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := 15

/-- The number of blue fish Roden bought -/
def blue_fish : ℕ := 7

/-- The total number of fish Roden bought -/
def total_fish : ℕ := gold_fish + blue_fish

theorem roden_fish_count : total_fish = 22 := by
  sorry

end roden_fish_count_l182_18205


namespace special_matrix_vector_product_l182_18207

def matrix_vector_op (a b c d e f : ℝ) : ℝ × ℝ :=
  (a * e + b * f, c * e + d * f)

theorem special_matrix_vector_product 
  (α β : ℝ) 
  (h1 : α + β = Real.pi) 
  (h2 : α - β = Real.pi / 2) : 
  matrix_vector_op (Real.sin α) (Real.cos α) (Real.cos α) (Real.sin α) (Real.cos β) (Real.sin β) = (0, 0) := by
  sorry

end special_matrix_vector_product_l182_18207


namespace min_value_sum_reciprocals_l182_18215

-- Define the line equation
def line_eq (a b x y : ℝ) : Prop := a * x - b * y + 8 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-2, 2)

-- Theorem statement
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line_passes_center : line_eq a b (circle_center.1) (circle_center.2)) :
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → line_eq a' b' (circle_center.1) (circle_center.2) → 
    1/a + 1/b ≤ 1/a' + 1/b') ∧ 
  (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ line_eq a' b' (circle_center.1) (circle_center.2) ∧ 
    1/a' + 1/b' = 1) :=
sorry

end min_value_sum_reciprocals_l182_18215


namespace point_on_curve_l182_18299

theorem point_on_curve : ∃ θ : ℝ, 1 + Real.sin θ = 3/2 ∧ Real.sin (2*θ) = Real.sqrt 3 / 2 := by
  sorry

end point_on_curve_l182_18299


namespace fraction_subtraction_l182_18277

theorem fraction_subtraction (x : ℝ) : 
  x * 8000 - (1 / 20) * (1 / 100) * 8000 = 796 → x = 0.1 := by
  sorry

end fraction_subtraction_l182_18277


namespace stating_perfect_match_equation_l182_18298

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 22

/-- Represents the number of screws a worker can produce per day -/
def screws_per_worker : ℕ := 1200

/-- Represents the number of nuts a worker can produce per day -/
def nuts_per_worker : ℕ := 2000

/-- Represents the number of nuts needed for each screw -/
def nuts_per_screw : ℕ := 2

/-- 
Theorem stating that for a perfect match of products, 
the equation 2 × 1200(22 - x) = 2000x must hold, 
where x is the number of workers assigned to produce nuts
-/
theorem perfect_match_equation (x : ℕ) (h : x ≤ total_workers) : 
  2 * screws_per_worker * (total_workers - x) = nuts_per_worker * x := by
  sorry

end stating_perfect_match_equation_l182_18298


namespace min_voters_for_tall_giraffe_win_l182_18220

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingSystem :=
  (total_voters : ℕ)
  (num_districts : ℕ)
  (precincts_per_district : ℕ)
  (voters_per_precinct : ℕ)
  (h_total : total_voters = num_districts * precincts_per_district * voters_per_precinct)

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingSystem) : ℕ :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win := (vs.precincts_per_district + 1) / 2
  let voters_to_win_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win * voters_to_win_precinct

/-- The theorem stating the minimum number of voters required for the Tall giraffe to win -/
theorem min_voters_for_tall_giraffe_win (vs : VotingSystem) 
  (h_voters : vs.total_voters = 135)
  (h_districts : vs.num_districts = 5)
  (h_precincts : vs.precincts_per_district = 9)
  (h_voters_per_precinct : vs.voters_per_precinct = 3) :
  min_voters_to_win vs = 30 := by
  sorry

#eval min_voters_to_win { total_voters := 135, num_districts := 5, precincts_per_district := 9, voters_per_precinct := 3, h_total := rfl }

end min_voters_for_tall_giraffe_win_l182_18220


namespace guests_not_responded_l182_18208

def total_guests : ℕ := 200
def yes_percentage : ℚ := 83 / 100
def no_percentage : ℚ := 9 / 100

theorem guests_not_responded : 
  (total_guests : ℚ) - 
  (yes_percentage * total_guests + no_percentage * total_guests) = 16 := by
  sorry

end guests_not_responded_l182_18208


namespace greatest_of_three_consecutive_integers_l182_18210

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 36) : 
  max x (max (x + 1) (x + 2)) = 13 := by
sorry

end greatest_of_three_consecutive_integers_l182_18210


namespace forty_knocks_to_knicks_l182_18242

-- Define the units
def Knick : Type := ℚ
def Knack : Type := ℚ
def Knock : Type := ℚ

-- Define the conversion rates
def knicks_to_knacks : ℚ := 3 / 8
def knacks_to_knocks : ℚ := 5 / 4

-- Theorem statement
theorem forty_knocks_to_knicks :
  (40 : ℚ) * knacks_to_knocks⁻¹ * knicks_to_knacks⁻¹ = 128 / 3 := by
  sorry

end forty_knocks_to_knicks_l182_18242


namespace coefficient_x5_expansion_l182_18273

/-- The coefficient of x^5 in the expansion of (1+x)^2(1-x)^5 is -1 -/
theorem coefficient_x5_expansion : Int := by
  sorry

end coefficient_x5_expansion_l182_18273


namespace smallest_w_l182_18219

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_w (w : ℕ) : w ≥ 676 ↔ 
  (is_divisible (1452 * w) (2^4)) ∧ 
  (is_divisible (1452 * w) (3^3)) ∧ 
  (is_divisible (1452 * w) (13^3)) := by
  sorry

end smallest_w_l182_18219


namespace tate_education_years_l182_18226

/-- The total years Tate spent in high school and college -/
def totalEducationYears (normalHighSchoolYears : ℕ) : ℕ :=
  let tateHighSchoolYears := normalHighSchoolYears - 1
  let tertiaryEducationYears := 3 * tateHighSchoolYears
  tateHighSchoolYears + tertiaryEducationYears

/-- Theorem stating that Tate's total education years is 12 -/
theorem tate_education_years :
  totalEducationYears 4 = 12 := by
  sorry

end tate_education_years_l182_18226


namespace mark_young_fish_count_l182_18295

/-- Calculates the total number of young fish given the number of tanks, pregnant fish per tank, and young per fish. -/
def total_young_fish (num_tanks : ℕ) (fish_per_tank : ℕ) (young_per_fish : ℕ) : ℕ :=
  num_tanks * fish_per_tank * young_per_fish

/-- Proves that given 5 tanks, 6 pregnant fish per tank, and 25 young per fish, the total number of young fish is 750. -/
theorem mark_young_fish_count :
  total_young_fish 5 6 25 = 750 := by
  sorry

end mark_young_fish_count_l182_18295


namespace g_of_seven_l182_18234

theorem g_of_seven (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 2) = 5 * x + 4) : g 7 = 19 := by
  sorry

end g_of_seven_l182_18234


namespace lunchroom_students_l182_18216

theorem lunchroom_students (students_per_table : ℕ) (num_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : num_tables = 34) : 
  students_per_table * num_tables = 204 := by
  sorry

end lunchroom_students_l182_18216


namespace solve_complex_equation_l182_18280

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (w : ℂ) : Prop :=
  2 + 3 * i * w = 4 - 2 * i * w

-- State the theorem
theorem solve_complex_equation :
  ∃ w : ℂ, equation w ∧ w = -2 * i / 5 :=
by sorry

end solve_complex_equation_l182_18280
