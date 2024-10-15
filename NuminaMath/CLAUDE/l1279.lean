import Mathlib

namespace NUMINAMATH_CALUDE_picture_distance_l1279_127944

theorem picture_distance (wall_width picture_width : ℝ) 
  (hw : wall_width = 25)
  (hp : picture_width = 3) :
  (wall_width - picture_width) / 2 = 11 := by
sorry

end NUMINAMATH_CALUDE_picture_distance_l1279_127944


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l1279_127980

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 36) : 
  max x (max (x + 1) (x + 2)) = 13 := by
sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l1279_127980


namespace NUMINAMATH_CALUDE_calculate_mixed_fraction_expression_l1279_127988

theorem calculate_mixed_fraction_expression : 
  (47 * ((2 + 2/3) - (3 + 1/4))) / ((3 + 1/2) + (2 + 1/5)) = -(4 + 25/38) := by
  sorry

end NUMINAMATH_CALUDE_calculate_mixed_fraction_expression_l1279_127988


namespace NUMINAMATH_CALUDE_trig_identity_l1279_127923

theorem trig_identity : 
  (3 / (Real.sin (20 * π / 180))^2) - (1 / (Real.cos (20 * π / 180))^2) + 64 * (Real.sin (20 * π / 180))^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1279_127923


namespace NUMINAMATH_CALUDE_mrs_heine_purchase_l1279_127938

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


end NUMINAMATH_CALUDE_mrs_heine_purchase_l1279_127938


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l1279_127939

/-- Given a trapezoid ABCD where:
    1. The ratio of the area of triangle ABC to the area of triangle ADC is 5:2
    2. AB + CD = 280 cm
    Prove that AB = 200 cm -/
theorem trapezoid_segment_length (AB CD : ℝ) (h : ℝ) : 
  (AB * h / 2) / (CD * h / 2) = 5 / 2 →
  AB + CD = 280 →
  AB = 200 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l1279_127939


namespace NUMINAMATH_CALUDE_bacteria_growth_l1279_127999

theorem bacteria_growth (n : ℕ) : n = 4 ↔ (n > 0 ∧ 5 * 3^n > 200 ∧ ∀ m : ℕ, m > 0 → m < n → 5 * 3^m ≤ 200) :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_l1279_127999


namespace NUMINAMATH_CALUDE_sequence_term_l1279_127974

/-- The sum of the first n terms of a sequence -/
def S (n : ℕ) : ℤ := n^2 - 3*n

/-- The nth term of the sequence -/
def a (n : ℕ) : ℤ := 2*n - 4

theorem sequence_term (n : ℕ) (h : n ≥ 1) : 
  a n = S n - S (n-1) :=
sorry

end NUMINAMATH_CALUDE_sequence_term_l1279_127974


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l1279_127924

theorem arithmetic_sequence_inequality (a b c : ℝ) (h1 : b - a = c - b) (h2 : b - a ≠ 0) :
  ¬ (∀ a b c : ℝ, a^3*b + b^3*c + c^3*a ≥ a^4 + b^4 + c^4) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l1279_127924


namespace NUMINAMATH_CALUDE_bead_removal_proof_l1279_127934

theorem bead_removal_proof (total_beads : ℕ) (parts : ℕ) (final_beads : ℕ) (x : ℕ) : 
  total_beads = 39 →
  parts = 3 →
  final_beads = 6 →
  2 * ((total_beads / parts) - x) = final_beads →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_bead_removal_proof_l1279_127934


namespace NUMINAMATH_CALUDE_sin_120_degrees_l1279_127927

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l1279_127927


namespace NUMINAMATH_CALUDE_pupils_like_only_maths_l1279_127953

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


end NUMINAMATH_CALUDE_pupils_like_only_maths_l1279_127953


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_l1279_127911

/-- An isosceles triangle with perimeter 13 and one side 3 has a base of 3 -/
theorem isosceles_triangle_base (a b c : ℝ) : 
  a + b + c = 13 →  -- perimeter is 13
  a = b →           -- isosceles condition
  (a = 3 ∨ b = 3 ∨ c = 3) →  -- one side is 3
  c = 3 :=          -- base is 3
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_l1279_127911


namespace NUMINAMATH_CALUDE_magnitude_of_z_to_fourth_l1279_127935

-- Define the complex number
def z : ℂ := 4 - 3 * Complex.I

-- State the theorem
theorem magnitude_of_z_to_fourth : Complex.abs (z^4) = 625 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_to_fourth_l1279_127935


namespace NUMINAMATH_CALUDE_proposition_equivalence_l1279_127929

theorem proposition_equivalence (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ x^2 - 2*x > m) ↔ m < 3 := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l1279_127929


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_72_l1279_127998

theorem five_digit_divisible_by_72 (a b : Nat) : 
  (a < 10 ∧ b < 10) →
  (a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b) % 72 = 0 ↔ 
  (a = 3 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_72_l1279_127998


namespace NUMINAMATH_CALUDE_initial_cards_count_l1279_127936

/-- The initial number of baseball cards Fred had -/
def initial_cards : ℕ := sorry

/-- The number of baseball cards Keith bought -/
def cards_bought : ℕ := 22

/-- The number of baseball cards Fred has left -/
def cards_left : ℕ := 18

/-- Theorem stating that the initial number of cards is 40 -/
theorem initial_cards_count : initial_cards = 40 := by sorry

end NUMINAMATH_CALUDE_initial_cards_count_l1279_127936


namespace NUMINAMATH_CALUDE_symmetric_points_ratio_l1279_127993

/-- Given two points A and B symmetric about a line ax + y - b = 0, prove that a/b = 1/3 -/
theorem symmetric_points_ratio (a b : ℝ) : 
  let A : ℝ × ℝ := (-1, 3)
  let B : ℝ × ℝ := (3, 5)
  (∀ (x y : ℝ), (x = -1 ∧ y = 3) ∨ (x = 3 ∧ y = 5) → a * x + y - b = 0) →
  (∀ (x y : ℝ), a * x + y - b = 0 → a * ((3 - (-1))/2 + (-1)) + ((5 - 3)/2 + 3) - b = 0) →
  a / b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_ratio_l1279_127993


namespace NUMINAMATH_CALUDE_unique_digit_multiplication_l1279_127940

theorem unique_digit_multiplication (B : ℕ) : 
  (B < 10) →                           -- B is a single digit
  (B2 : ℕ) →                           -- B2 is a natural number
  (B2 = 10 * B + 2) →                  -- B2 is a two-digit number ending in 2
  (7 * B < 100) →                      -- 7B is a two-digit number
  (B2 * (70 + B) = 6396) →             -- The multiplication equation
  (B = 8) := by sorry

end NUMINAMATH_CALUDE_unique_digit_multiplication_l1279_127940


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l1279_127906

/-- Represents a triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle satisfies the triangle inequality. -/
def Triangle.isValid (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Checks if a triangle is isosceles. -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle. -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar. -/
def areSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem similar_triangle_perimeter 
  (t1 t2 : Triangle) 
  (h1 : t1.isValid)
  (h2 : t2.isValid)
  (h3 : t1.isIsosceles)
  (h4 : t2.isIsosceles)
  (h5 : areSimilar t1 t2)
  (h6 : t1.a = 8 ∧ t1.b = 24 ∧ t1.c = 24)
  (h7 : t2.a = 40) : 
  t2.perimeter = 280 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l1279_127906


namespace NUMINAMATH_CALUDE_card_distribution_events_l1279_127997

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

end NUMINAMATH_CALUDE_card_distribution_events_l1279_127997


namespace NUMINAMATH_CALUDE_range_of_a_l1279_127932

theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ+, ((1 - a) * n - a) * Real.log a < 0) ↔ (0 < a ∧ a < 1/2) ∨ a > 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1279_127932


namespace NUMINAMATH_CALUDE_same_color_probability_l1279_127994

/-- The probability of drawing two balls of the same color from a bag containing black and white balls -/
theorem same_color_probability (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = black_balls + white_balls)
  (h2 : black_balls = 7)
  (h3 : white_balls = 8) :
  (black_balls * (black_balls - 1) + white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1)) = 7 / 15 := by
sorry

end NUMINAMATH_CALUDE_same_color_probability_l1279_127994


namespace NUMINAMATH_CALUDE_integer_triple_sum_equation_l1279_127922

theorem integer_triple_sum_equation : ∃ (x y z : ℕ),
  1000 < x ∧ x < y ∧ y < z ∧ z < 2000 ∧
  (1 : ℚ) / 2 + 1 / 3 + 1 / 7 + 1 / x + 1 / y + 1 / z + 1 / 45 = 1 ∧
  x = 1806 ∧ y = 1892 ∧ z = 1980 := by
  sorry

end NUMINAMATH_CALUDE_integer_triple_sum_equation_l1279_127922


namespace NUMINAMATH_CALUDE_shirt_price_proof_l1279_127982

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

end NUMINAMATH_CALUDE_shirt_price_proof_l1279_127982


namespace NUMINAMATH_CALUDE_contrapositive_of_true_implication_l1279_127977

theorem contrapositive_of_true_implication (h : ∀ x : ℝ, x < 0 → x^2 > 0) :
  ∀ x : ℝ, x^2 ≤ 0 → x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_of_true_implication_l1279_127977


namespace NUMINAMATH_CALUDE_solution_set_f_gt_7_min_m2_plus_n2_l1279_127969

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

-- Theorem for the solution set of f(x) > 7
theorem solution_set_f_gt_7 :
  {x : ℝ | f x > 7} = {x : ℝ | x > 4 ∨ x < -3} := by sorry

-- Theorem for the minimum value of m^2 + n^2 and the values of m and n
theorem min_m2_plus_n2 (m n : ℝ) (hn : n > 0) (h_min : ∀ x, f x ≥ m + n) :
  m^2 + n^2 ≥ 9/2 ∧ (m^2 + n^2 = 9/2 ↔ m = 3/2 ∧ n = 3/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_7_min_m2_plus_n2_l1279_127969


namespace NUMINAMATH_CALUDE_count_special_numbers_is_360_l1279_127964

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

end NUMINAMATH_CALUDE_count_special_numbers_is_360_l1279_127964


namespace NUMINAMATH_CALUDE_guests_not_responded_l1279_127948

def total_guests : ℕ := 200
def yes_percentage : ℚ := 83 / 100
def no_percentage : ℚ := 9 / 100

theorem guests_not_responded : 
  (total_guests : ℚ) - 
  (yes_percentage * total_guests + no_percentage * total_guests) = 16 := by
  sorry

end NUMINAMATH_CALUDE_guests_not_responded_l1279_127948


namespace NUMINAMATH_CALUDE_sandcastle_height_difference_l1279_127959

theorem sandcastle_height_difference (miki_height sister_height : ℝ) 
  (h1 : miki_height = 0.83)
  (h2 : sister_height = 0.5) :
  miki_height - sister_height = 0.33 := by sorry

end NUMINAMATH_CALUDE_sandcastle_height_difference_l1279_127959


namespace NUMINAMATH_CALUDE_factor_expression_l1279_127972

theorem factor_expression (a : ℝ) : 53 * a^2 + 159 * a = 53 * a * (a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1279_127972


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1279_127990

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

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1279_127990


namespace NUMINAMATH_CALUDE_proportion_third_number_l1279_127918

theorem proportion_third_number (y : ℝ) : 
  (0.6 : ℝ) / 0.96 = y / 8 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_third_number_l1279_127918


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1279_127956

theorem algebraic_expression_equality (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (x + 1) / (x - 1) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1279_127956


namespace NUMINAMATH_CALUDE_bumper_car_line_count_l1279_127985

/-- The number of people waiting in line for bumper cars after changes -/
def total_people_waiting (initial1 initial2 initial3 left1 left2 left3 joined1 joined2 joined3 : ℕ) : ℕ :=
  (initial1 - left1 + joined1) + (initial2 - left2 + joined2) + (initial3 - left3 + joined3)

/-- Theorem stating the total number of people waiting in line for bumper cars after changes -/
theorem bumper_car_line_count : 
  total_people_waiting 7 12 15 4 3 5 8 10 7 = 47 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_count_l1279_127985


namespace NUMINAMATH_CALUDE_x_four_coefficient_range_l1279_127921

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^4 in the expansion of (1+x+mx^2)^10
def coefficient (m : ℝ) : ℝ := binomial 10 4 + binomial 10 2 * m^2 + binomial 10 1 * binomial 9 2 * m

-- State the theorem
theorem x_four_coefficient_range :
  {m : ℝ | coefficient m > -330} = {m : ℝ | m < -6 ∨ m > -2} := by sorry

end NUMINAMATH_CALUDE_x_four_coefficient_range_l1279_127921


namespace NUMINAMATH_CALUDE_avocados_for_guacamole_l1279_127949

/-- The number of avocados needed for one serving of guacamole -/
def avocados_per_serving (initial_avocados sister_avocados total_servings : ℕ) : ℕ :=
  (initial_avocados + sister_avocados) / total_servings

/-- Theorem stating that 3 avocados are needed for one serving of guacamole -/
theorem avocados_for_guacamole :
  avocados_per_serving 5 4 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_avocados_for_guacamole_l1279_127949


namespace NUMINAMATH_CALUDE_fifth_power_prime_solution_l1279_127989

theorem fifth_power_prime_solution :
  ∀ (x y p : ℕ+),
  (x^2 + y) * (y^2 + x) = p^5 ∧ Nat.Prime p.val →
  ((x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2)) :=
sorry

end NUMINAMATH_CALUDE_fifth_power_prime_solution_l1279_127989


namespace NUMINAMATH_CALUDE_emily_small_gardens_l1279_127983

def number_of_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

theorem emily_small_gardens :
  number_of_small_gardens 42 36 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_emily_small_gardens_l1279_127983


namespace NUMINAMATH_CALUDE_lunchroom_students_l1279_127962

theorem lunchroom_students (students_per_table : ℕ) (num_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : num_tables = 34) : 
  students_per_table * num_tables = 204 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_students_l1279_127962


namespace NUMINAMATH_CALUDE_problem_statement_l1279_127986

theorem problem_statement (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b)^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1279_127986


namespace NUMINAMATH_CALUDE_bowling_record_difference_l1279_127904

theorem bowling_record_difference (old_record : ℕ) (players : ℕ) (rounds : ℕ) (current_score : ℕ) : 
  old_record = 287 →
  players = 4 →
  rounds = 10 →
  current_score = 10440 →
  old_record - (((old_record * players * rounds) - current_score) / players) = 27 := by
sorry

end NUMINAMATH_CALUDE_bowling_record_difference_l1279_127904


namespace NUMINAMATH_CALUDE_geometry_problem_l1279_127954

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

end NUMINAMATH_CALUDE_geometry_problem_l1279_127954


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1279_127995

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ + q = 0 ∧ 
    x₂^2 + p*x₂ + q = 0 ∧ 
    |x₁ - x₂| = 2) →
  p = Real.sqrt (4*q + 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1279_127995


namespace NUMINAMATH_CALUDE_glasses_in_larger_box_l1279_127979

theorem glasses_in_larger_box 
  (small_box : ℕ) 
  (total_boxes : ℕ) 
  (average_glasses : ℕ) :
  small_box = 12 → 
  total_boxes = 2 → 
  average_glasses = 15 → 
  ∃ large_box : ℕ, 
    (small_box + large_box) / total_boxes = average_glasses ∧ 
    large_box = 18 := by
sorry

end NUMINAMATH_CALUDE_glasses_in_larger_box_l1279_127979


namespace NUMINAMATH_CALUDE_approximation_problem_l1279_127978

def is_close (x y : ℝ) (ε : ℝ) : Prop := |x - y| ≤ ε

theorem approximation_problem :
  (∀ n : ℕ, 5 ≤ n ∧ n ≤ 9 → is_close (5 * n * 18) 1200 90) ∧
  (∀ m : ℕ, 0 ≤ m ∧ m ≤ 2 → is_close ((3 * 10 + m) * 9 / 5) 60 5) :=
sorry

end NUMINAMATH_CALUDE_approximation_problem_l1279_127978


namespace NUMINAMATH_CALUDE_yans_distance_ratio_l1279_127946

theorem yans_distance_ratio (w : ℝ) (x y : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) :
  (y / w = x / w + (x + y) / (6 * w)) → (x / y = 5 / 7) :=
by sorry

end NUMINAMATH_CALUDE_yans_distance_ratio_l1279_127946


namespace NUMINAMATH_CALUDE_daughter_and_child_weight_l1279_127958

/-- The combined weight of a daughter and her child given specific family weight conditions -/
theorem daughter_and_child_weight (total_weight mother_weight daughter_weight child_weight : ℝ) :
  total_weight = mother_weight + daughter_weight + child_weight →
  child_weight = (1 / 5 : ℝ) * mother_weight →
  daughter_weight = 48 →
  total_weight = 120 →
  daughter_weight + child_weight = 60 :=
by
  sorry

#check daughter_and_child_weight

end NUMINAMATH_CALUDE_daughter_and_child_weight_l1279_127958


namespace NUMINAMATH_CALUDE_fish_population_estimate_l1279_127937

/-- Estimates the fish population on January 1 based on capture-recapture data --/
theorem fish_population_estimate 
  (initial_tagged : ℕ)
  (june_sample : ℕ)
  (june_tagged : ℕ)
  (tagged_left_percent : ℚ)
  (new_juvenile_percent : ℚ) :
  initial_tagged = 100 →
  june_sample = 150 →
  june_tagged = 4 →
  tagged_left_percent = 30 / 100 →
  new_juvenile_percent = 50 / 100 →
  ∃ (estimated_population : ℕ), estimated_population = 1312 := by
  sorry


end NUMINAMATH_CALUDE_fish_population_estimate_l1279_127937


namespace NUMINAMATH_CALUDE_all_fractions_repeat_l1279_127905

theorem all_fractions_repeat (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 20) : 
  ¬ (∃ k : ℕ, n * (5^k * 2^k) = 42 * m) :=
sorry

end NUMINAMATH_CALUDE_all_fractions_repeat_l1279_127905


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1279_127950

/-- The total surface area of a rectangular solid -/
def totalSurfaceArea (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 10 meters, 
    width 9 meters, and depth 6 meters is 408 square meters -/
theorem rectangular_solid_surface_area :
  totalSurfaceArea 10 9 6 = 408 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1279_127950


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l1279_127987

theorem nested_subtraction_simplification (z : ℝ) :
  1 - (2 - (3 - (4 - (5 - z)))) = 3 - z := by sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l1279_127987


namespace NUMINAMATH_CALUDE_unique_valid_n_l1279_127943

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

end NUMINAMATH_CALUDE_unique_valid_n_l1279_127943


namespace NUMINAMATH_CALUDE_selection_methods_l1279_127926

theorem selection_methods (n_boys n_girls n_select : ℕ) : 
  n_boys = 4 → n_girls = 3 → n_select = 4 →
  (Nat.choose (n_boys + n_girls) n_select) - (Nat.choose n_boys n_select) = 34 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_l1279_127926


namespace NUMINAMATH_CALUDE_tan_half_angle_second_quadrant_l1279_127941

/-- If α is an angle in the second quadrant and 3sinα + 4cosα = 0, then tan(α/2) = 2 -/
theorem tan_half_angle_second_quadrant (α : Real) : 
  π/2 < α ∧ α < π → -- α is in the second quadrant
  3 * Real.sin α + 4 * Real.cos α = 0 → -- given equation
  Real.tan (α/2) = 2 := by sorry

end NUMINAMATH_CALUDE_tan_half_angle_second_quadrant_l1279_127941


namespace NUMINAMATH_CALUDE_percentage_increase_l1279_127933

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 100 → final = 150 → (final - initial) / initial * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1279_127933


namespace NUMINAMATH_CALUDE_forgotten_angles_sum_l1279_127901

theorem forgotten_angles_sum (n : ℕ) (measured_sum : ℝ) : 
  n > 2 → 
  measured_sum = 2873 → 
  ∃ (missing_sum : ℝ), 
    missing_sum = ((n - 2) * 180 : ℝ) - measured_sum ∧ 
    missing_sum = 7 := by
  sorry

end NUMINAMATH_CALUDE_forgotten_angles_sum_l1279_127901


namespace NUMINAMATH_CALUDE_no_two_common_tangents_l1279_127919

/-- Two circles in a plane with radii r and 2r -/
structure TwoCircles (r : ℝ) where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ

/-- The number of common tangents between two circles -/
def numCommonTangents (c : TwoCircles r) : ℕ := sorry

/-- Theorem: It's impossible for two circles with radii r and 2r to have exactly 2 common tangents -/
theorem no_two_common_tangents (r : ℝ) (hr : r > 0) :
  ∀ c : TwoCircles r, numCommonTangents c ≠ 2 := by sorry

end NUMINAMATH_CALUDE_no_two_common_tangents_l1279_127919


namespace NUMINAMATH_CALUDE_log_sum_simplification_l1279_127966

theorem log_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_simplification_l1279_127966


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_value_l1279_127991

theorem quadratic_solution_implies_value (a b : ℝ) : 
  (1 : ℝ)^2 + a * 1 + 2 * b = 0 → 2023 - a - 2 * b = 2024 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_value_l1279_127991


namespace NUMINAMATH_CALUDE_fraction_problem_l1279_127942

theorem fraction_problem : ∃ x : ℚ, (65 / 100 * 40 : ℚ) = x * 25 + 6 ∧ x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1279_127942


namespace NUMINAMATH_CALUDE_computer_cost_l1279_127965

theorem computer_cost (initial_amount printer_cost amount_left : ℕ) 
  (h1 : initial_amount = 450)
  (h2 : printer_cost = 40)
  (h3 : amount_left = 10) :
  initial_amount - printer_cost - amount_left = 400 :=
by sorry

end NUMINAMATH_CALUDE_computer_cost_l1279_127965


namespace NUMINAMATH_CALUDE_price_change_percentage_l1279_127910

theorem price_change_percentage (P : ℝ) (x : ℝ) : 
  P * (1 + x/100) * (1 - x/100) = 0.64 * P → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_price_change_percentage_l1279_127910


namespace NUMINAMATH_CALUDE_carly_running_ratio_l1279_127909

/-- Carly's running schedule over four weeks -/
def running_schedule (r : ℚ) : Fin 4 → ℚ
  | 0 => 2                    -- First week
  | 1 => 2 * r + 3            -- Second week
  | 2 => 9/7 * (2 * r + 3)    -- Third week
  | 3 => 4                    -- Fourth week

theorem carly_running_ratio :
  ∃ r : ℚ,
    running_schedule r 2 = 9 ∧
    running_schedule r 3 = running_schedule r 2 - 5 ∧
    running_schedule r 1 / running_schedule r 0 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_carly_running_ratio_l1279_127909


namespace NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l1279_127992

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

end NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l1279_127992


namespace NUMINAMATH_CALUDE_special_matrix_vector_product_l1279_127973

def matrix_vector_op (a b c d e f : ℝ) : ℝ × ℝ :=
  (a * e + b * f, c * e + d * f)

theorem special_matrix_vector_product 
  (α β : ℝ) 
  (h1 : α + β = Real.pi) 
  (h2 : α - β = Real.pi / 2) : 
  matrix_vector_op (Real.sin α) (Real.cos α) (Real.cos α) (Real.sin α) (Real.cos β) (Real.sin β) = (0, 0) := by
  sorry

end NUMINAMATH_CALUDE_special_matrix_vector_product_l1279_127973


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l1279_127957

/-- Represents a two-digit number -/
def two_digit_number := { n : ℕ | 10 ≤ n ∧ n < 100 }

/-- Constructs a six-digit number by repeating a two-digit number three times -/
def repeat_three_times (n : two_digit_number) : ℕ :=
  100000 * n + 1000 * n + n

theorem six_digit_divisibility (n : two_digit_number) :
  (repeat_three_times n) % 10101 = 0 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l1279_127957


namespace NUMINAMATH_CALUDE_fraction_simplification_l1279_127907

theorem fraction_simplification : (3 * 4) / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1279_127907


namespace NUMINAMATH_CALUDE_seventh_selected_number_l1279_127931

def random_sequence : List ℕ := [6572, 0802, 6319, 8702, 4369, 9728, 0198, 3204, 9243, 4935, 8200, 3623, 4869, 6938, 7481]

def is_valid (n : ℕ) : Bool := 1 ≤ n ∧ n ≤ 500

def select_valid_numbers (seq : List ℕ) : List ℕ :=
  seq.filter (λ n => is_valid (n % 1000))

theorem seventh_selected_number :
  (select_valid_numbers random_sequence).nthLe 6 sorry = 320 := by sorry

end NUMINAMATH_CALUDE_seventh_selected_number_l1279_127931


namespace NUMINAMATH_CALUDE_purely_imaginary_square_root_l1279_127925

theorem purely_imaginary_square_root (a : ℝ) : 
  (∃ b : ℝ, (a - Complex.I) ^ 2 = Complex.I * b ∧ b ≠ 0) → (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_square_root_l1279_127925


namespace NUMINAMATH_CALUDE_x_one_value_l1279_127960

theorem x_one_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h_equation : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + (x₃ - x₄)^2 + x₄^2 = 1/3) :
  x₁ = 4/5 := by
sorry

end NUMINAMATH_CALUDE_x_one_value_l1279_127960


namespace NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l1279_127917

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l1279_127917


namespace NUMINAMATH_CALUDE_roden_fish_count_l1279_127976

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := 15

/-- The number of blue fish Roden bought -/
def blue_fish : ℕ := 7

/-- The total number of fish Roden bought -/
def total_fish : ℕ := gold_fish + blue_fish

theorem roden_fish_count : total_fish = 22 := by
  sorry

end NUMINAMATH_CALUDE_roden_fish_count_l1279_127976


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l1279_127967

theorem fraction_product_simplification :
  (6 : ℚ) / 3 * (9 : ℚ) / 6 * (12 : ℚ) / 9 * (15 : ℚ) / 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l1279_127967


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1279_127900

/-- Given a square with perimeter 144 units divided into 4 congruent rectangles by vertical lines,
    the perimeter of one rectangle is 90 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (num_rectangles : ℕ) : 
  square_perimeter = 144 → 
  num_rectangles = 4 → 
  ∃ (rectangle_perimeter : ℝ), rectangle_perimeter = 90 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1279_127900


namespace NUMINAMATH_CALUDE_traci_road_trip_l1279_127945

/-- Proves that the fraction of the remaining distance traveled between the first and second stops is 1/4 -/
theorem traci_road_trip (total_distance : ℝ) (first_stop_fraction : ℝ) (final_leg : ℝ) : 
  total_distance = 600 →
  first_stop_fraction = 1/3 →
  final_leg = 300 →
  (total_distance - first_stop_fraction * total_distance - final_leg) / (total_distance - first_stop_fraction * total_distance) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_traci_road_trip_l1279_127945


namespace NUMINAMATH_CALUDE_chinese_books_probability_l1279_127913

theorem chinese_books_probability (total_books : ℕ) (chinese_books : ℕ) (math_books : ℕ) :
  total_books = chinese_books + math_books →
  chinese_books = 3 →
  math_books = 2 →
  (Nat.choose chinese_books 2 : ℚ) / (Nat.choose total_books 2) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_chinese_books_probability_l1279_127913


namespace NUMINAMATH_CALUDE_tate_education_years_l1279_127970

/-- The total years Tate spent in high school and college -/
def totalEducationYears (normalHighSchoolYears : ℕ) : ℕ :=
  let tateHighSchoolYears := normalHighSchoolYears - 1
  let tertiaryEducationYears := 3 * tateHighSchoolYears
  tateHighSchoolYears + tertiaryEducationYears

/-- Theorem stating that Tate's total education years is 12 -/
theorem tate_education_years :
  totalEducationYears 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_tate_education_years_l1279_127970


namespace NUMINAMATH_CALUDE_number_of_girls_in_class_correct_number_of_girls_l1279_127908

theorem number_of_girls_in_class (num_boys : ℕ) (group_size : ℕ) (num_groups : ℕ) : ℕ :=
  let total_members := group_size * num_groups
  let num_girls := total_members - num_boys
  num_girls

theorem correct_number_of_girls :
  number_of_girls_in_class 9 3 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_class_correct_number_of_girls_l1279_127908


namespace NUMINAMATH_CALUDE_fan_ratio_proof_l1279_127916

/-- Represents the number of fans for each team -/
structure FanCount where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The ratio of Mets fans to Red Sox fans is 4:5 -/
def mets_to_red_sox_ratio (fc : FanCount) : Prop :=
  4 * fc.red_sox = 5 * fc.mets

/-- The total number of fans is 330 -/
def total_fans (fc : FanCount) : Prop :=
  fc.yankees + fc.mets + fc.red_sox = 330

/-- There are 88 Mets fans -/
def mets_fan_count (fc : FanCount) : Prop :=
  fc.mets = 88

/-- The ratio of Yankees fans to Mets fans is 3:2 -/
def yankees_to_mets_ratio (fc : FanCount) : Prop :=
  3 * fc.mets = 2 * fc.yankees

theorem fan_ratio_proof (fc : FanCount)
    (h1 : mets_to_red_sox_ratio fc)
    (h2 : total_fans fc)
    (h3 : mets_fan_count fc) :
  yankees_to_mets_ratio fc := by
  sorry

end NUMINAMATH_CALUDE_fan_ratio_proof_l1279_127916


namespace NUMINAMATH_CALUDE_inverse_proportion_m_value_l1279_127984

def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

theorem inverse_proportion_m_value (m : ℝ) :
  (is_inverse_proportion (λ x => (m - 1) * x^(|m| - 2))) →
  m = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_m_value_l1279_127984


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1279_127961

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

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1279_127961


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l1279_127975

/-- Given vectors a and b in R^2, with b = (-1, 2) and their sum (1, 3), 
    prove that the magnitude of a - 2b is 5. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) : 
  b = (-1, 2) → a + b = (1, 3) → ‖a - 2 • b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l1279_127975


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l1279_127955

def repeating_decimal_234 : ℚ := 234 / 999
def repeating_decimal_567 : ℚ := 567 / 999
def repeating_decimal_891 : ℚ := 891 / 999

theorem repeating_decimal_subtraction :
  repeating_decimal_234 - repeating_decimal_567 - repeating_decimal_891 = -1224 / 999 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l1279_127955


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1279_127903

/-- Represents the ratio of students in grades 10, 11, and 12 -/
def student_ratio : Fin 3 → ℕ
| 0 => 2  -- Grade 10
| 1 => 3  -- Grade 11
| 2 => 5  -- Grade 12
| _ => 0  -- Unreachable case

/-- The total parts in the ratio -/
def total_ratio : ℕ := (student_ratio 0) + (student_ratio 1) + (student_ratio 2)

/-- The number of grade 12 students in the sample -/
def grade_12_sample : ℕ := 150

/-- The total sample size -/
def total_sample : ℕ := 300

theorem stratified_sample_size :
  (student_ratio 2 : ℚ) / total_ratio = grade_12_sample / total_sample :=
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l1279_127903


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1279_127915

theorem complex_modulus_problem (z : ℂ) : 
  z = (1 + 2*I)^2 / (-I + 2) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1279_127915


namespace NUMINAMATH_CALUDE_pet_store_puppies_l1279_127914

theorem pet_store_puppies (sold : ℕ) (puppies_per_cage : ℕ) (num_cages : ℕ) :
  sold = 3 ∧ puppies_per_cage = 5 ∧ num_cages = 3 →
  sold + num_cages * puppies_per_cage = 18 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l1279_127914


namespace NUMINAMATH_CALUDE_octagon_diagonals_eq_twenty_l1279_127912

/-- The number of diagonals in an octagon -/
def octagon_diagonals : ℕ :=
  let n := 8  -- number of vertices in an octagon
  let sides := 8  -- number of sides in an octagon
  (n * (n - 1)) / 2 - sides

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals_eq_twenty : octagon_diagonals = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_eq_twenty_l1279_127912


namespace NUMINAMATH_CALUDE_add_twice_eq_thrice_l1279_127920

theorem add_twice_eq_thrice (a : ℝ) : a + 2 * a = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_add_twice_eq_thrice_l1279_127920


namespace NUMINAMATH_CALUDE_division_equality_may_not_hold_l1279_127968

theorem division_equality_may_not_hold (a b c : ℝ) : 
  a = b → ¬(∀ c, a / c = b / c) :=
by
  sorry

end NUMINAMATH_CALUDE_division_equality_may_not_hold_l1279_127968


namespace NUMINAMATH_CALUDE_dollar_square_sum_l1279_127928

/-- Custom operation ▩ for real numbers -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem stating that (x + y)² ▩ (y² + x²) = 4x²y² -/
theorem dollar_square_sum (x y : ℝ) : dollar ((x + y)^2) (y^2 + x^2) = 4 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_dollar_square_sum_l1279_127928


namespace NUMINAMATH_CALUDE_imaginary_product_real_part_l1279_127963

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem imaginary_product_real_part (z : ℂ) (a b : ℝ) 
  (h1 : is_purely_imaginary z) 
  (h2 : (3 * Complex.I) * z = Complex.mk a b) : 
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_product_real_part_l1279_127963


namespace NUMINAMATH_CALUDE_go_stones_count_l1279_127996

/-- Calculates the total number of go stones given the number of stones per bundle,
    the number of bundles of black stones, and the number of white stones. -/
def total_go_stones (stones_per_bundle : ℕ) (black_bundles : ℕ) (white_stones : ℕ) : ℕ :=
  stones_per_bundle * black_bundles + white_stones

/-- Proves that the total number of go stones is 46 given the specified conditions. -/
theorem go_stones_count : total_go_stones 10 3 16 = 46 := by
  sorry

end NUMINAMATH_CALUDE_go_stones_count_l1279_127996


namespace NUMINAMATH_CALUDE_certain_number_proof_l1279_127947

theorem certain_number_proof : ∃ x : ℝ, (x / 3 = 400 * 1.005) ∧ (x = 1206) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1279_127947


namespace NUMINAMATH_CALUDE_no_equal_functions_l1279_127930

def f₁ (x : ℤ) : ℤ := x * (x - 2007)
def f₂ (x : ℤ) : ℤ := (x - 1) * (x - 2006)
def f₁₀₀₄ (x : ℤ) : ℤ := (x - 1003) * (x - 1004)

theorem no_equal_functions :
  ∀ x : ℤ, 0 ≤ x ∧ x ≤ 2007 →
    (f₁ x ≠ f₂ x) ∧ (f₁ x ≠ f₁₀₀₄ x) ∧ (f₂ x ≠ f₁₀₀₄ x) := by
  sorry

end NUMINAMATH_CALUDE_no_equal_functions_l1279_127930


namespace NUMINAMATH_CALUDE_smallest_w_l1279_127971

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_w (w : ℕ) : w ≥ 676 ↔ 
  (is_divisible (1452 * w) (2^4)) ∧ 
  (is_divisible (1452 * w) (3^3)) ∧ 
  (is_divisible (1452 * w) (13^3)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_w_l1279_127971


namespace NUMINAMATH_CALUDE_triangle_angle_sine_cosine_equivalence_l1279_127981

theorem triangle_angle_sine_cosine_equivalence (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) :
  (Real.sin A > Real.sin B) ↔ (Real.cos A + Real.cos (A + C) < 0) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sine_cosine_equivalence_l1279_127981


namespace NUMINAMATH_CALUDE_intersection_A_B_l1279_127951

def A : Set ℝ := {x | ∃ (α β : ℤ), α ≥ 0 ∧ β ≥ 0 ∧ x = 2^α * 3^β}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}

theorem intersection_A_B : A ∩ B = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1279_127951


namespace NUMINAMATH_CALUDE_only_f1_is_even_l1279_127902

-- Define the functions
def f1 (x : ℝ) : ℝ := x^2 - 3*abs x + 2
def f2 (x : ℝ) : ℝ := x^2
def f3 (x : ℝ) : ℝ := x^3
def f4 (x : ℝ) : ℝ := x - 1

-- Define the domain for f2
def f2_domain : Set ℝ := Set.Ioc (-2) 2

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem only_f1_is_even :
  is_even f1 ∧ ¬(is_even f2) ∧ ¬(is_even f3) ∧ ¬(is_even f4) :=
sorry

end NUMINAMATH_CALUDE_only_f1_is_even_l1279_127902


namespace NUMINAMATH_CALUDE_odd_m_triple_g_65_l1279_127952

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem odd_m_triple_g_65 (m : ℤ) (h_odd : m % 2 = 1) :
  g (g (g m)) = 65 → m = 255 := by sorry

end NUMINAMATH_CALUDE_odd_m_triple_g_65_l1279_127952
