import Mathlib

namespace NUMINAMATH_CALUDE_maaza_liters_l1919_191986

theorem maaza_liters (pepsi sprite cans : ℕ) (h1 : pepsi = 144) (h2 : sprite = 368) (h3 : cans = 281) :
  ∃ (M : ℕ), M + pepsi + sprite = cans * (M + pepsi + sprite) / cans ∧
  ∀ (M' : ℕ), M' + pepsi + sprite = cans * (M' + pepsi + sprite) / cans → M ≤ M' :=
by sorry

end NUMINAMATH_CALUDE_maaza_liters_l1919_191986


namespace NUMINAMATH_CALUDE_people_per_car_l1919_191968

/-- Given 3.0 cars and 189 people going to the zoo, prove that there are 63 people in each car. -/
theorem people_per_car (total_cars : Float) (total_people : Nat) : 
  total_cars = 3.0 → total_people = 189 → (total_people.toFloat / total_cars).round = 63 := by
  sorry

end NUMINAMATH_CALUDE_people_per_car_l1919_191968


namespace NUMINAMATH_CALUDE_inverse_100_mod_101_l1919_191915

theorem inverse_100_mod_101 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 100 ∧ (100 * x) % 101 = 1 :=
by
  use 100
  sorry

end NUMINAMATH_CALUDE_inverse_100_mod_101_l1919_191915


namespace NUMINAMATH_CALUDE_original_number_proof_l1919_191965

theorem original_number_proof (x : ℝ) : x * 1.1 = 550 ↔ x = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1919_191965


namespace NUMINAMATH_CALUDE_recurrence_sequence_uniqueness_l1919_191917

/-- A sequence defined by a recurrence relation -/
def RecurrenceSequence (p q : ℝ) (a₀ a₁ : ℝ) : ℕ → ℝ
| 0 => a₀
| 1 => a₁
| (n + 2) => p * RecurrenceSequence p q a₀ a₁ (n + 1) + q * RecurrenceSequence p q a₀ a₁ n

/-- Theorem: All terms in the sequence are uniquely determined -/
theorem recurrence_sequence_uniqueness (p q : ℝ) (a₀ a₁ : ℝ) :
  ∀ n : ℕ, ∃! x : ℝ, x = RecurrenceSequence p q a₀ a₁ n :=
by sorry

end NUMINAMATH_CALUDE_recurrence_sequence_uniqueness_l1919_191917


namespace NUMINAMATH_CALUDE_unoccupied_area_l1919_191977

/-- The area of the region not occupied by a smaller square inside a larger square -/
theorem unoccupied_area (large_side small_side : ℝ) (h1 : large_side = 10) (h2 : small_side = 4) :
  large_side ^ 2 - small_side ^ 2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_unoccupied_area_l1919_191977


namespace NUMINAMATH_CALUDE_inverse_inequality_conditions_l1919_191924

theorem inverse_inequality_conditions (a b : ℝ) :
  (1 / a < 1 / b) ↔ (b > 0 ∧ 0 > a) ∨ (0 > a ∧ a > b) ∨ (a > b ∧ b > 0) :=
sorry

end NUMINAMATH_CALUDE_inverse_inequality_conditions_l1919_191924


namespace NUMINAMATH_CALUDE_fraction_equality_l1919_191933

theorem fraction_equality (P Q : ℤ) : 
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 4 ∧ x ≠ -4 → 
    P / (x^2 - 5*x) + Q / (x + 4) = (x^2 - 3*x + 8) / (x^3 - 5*x^2 + 4*x)) →
  (Q : ℚ) / (P : ℚ) = 7 / 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1919_191933


namespace NUMINAMATH_CALUDE_peach_distribution_theorem_l1919_191951

/-- Represents the number of peaches each child received -/
structure PeachDistribution where
  anya : Nat
  katya : Nat
  liza : Nat
  dasha : Nat
  kolya : Nat
  petya : Nat
  tolya : Nat
  vasya : Nat

/-- Represents the last names of the children -/
inductive LastName
  | Ivanov
  | Grishin
  | Andreyev
  | Sergeyev

/-- Represents a child with their name and last name -/
structure Child where
  name : String
  lastName : LastName

/-- The theorem stating the correct distribution of peaches and last names -/
theorem peach_distribution_theorem (d : PeachDistribution) 
  (h1 : d.anya = 1)
  (h2 : d.katya = 2)
  (h3 : d.liza = 3)
  (h4 : d.dasha = 4)
  (h5 : d.kolya = d.liza)
  (h6 : d.petya = 2 * d.dasha)
  (h7 : d.tolya = 3 * d.anya)
  (h8 : d.vasya = 4 * d.katya)
  (h9 : d.anya + d.katya + d.liza + d.dasha + d.kolya + d.petya + d.tolya + d.vasya = 32) :
  ∃ (c1 c2 c3 c4 : Child),
    c1 = { name := "Liza", lastName := LastName.Ivanov } ∧
    c2 = { name := "Dasha", lastName := LastName.Grishin } ∧
    c3 = { name := "Anya", lastName := LastName.Andreyev } ∧
    c4 = { name := "Katya", lastName := LastName.Sergeyev } := by
  sorry

end NUMINAMATH_CALUDE_peach_distribution_theorem_l1919_191951


namespace NUMINAMATH_CALUDE_composition_value_l1919_191930

/-- Given two functions g and h, prove that their composition at x = 2 equals 3890 -/
theorem composition_value :
  let g (x : ℝ) := 3 * x^2 + 2
  let h (x : ℝ) := -5 * x^3 + 4
  g (h 2) = 3890 := by
  sorry

end NUMINAMATH_CALUDE_composition_value_l1919_191930


namespace NUMINAMATH_CALUDE_issac_utensils_count_l1919_191907

/-- The total number of writing utensils bought by Issac -/
def total_utensils (num_pens : ℕ) (num_pencils : ℕ) : ℕ :=
  num_pens + num_pencils

/-- Theorem stating the total number of writing utensils Issac bought -/
theorem issac_utensils_count :
  ∀ (num_pens : ℕ) (num_pencils : ℕ),
    num_pens = 16 →
    num_pencils = 5 * num_pens + 12 →
    total_utensils num_pens num_pencils = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_issac_utensils_count_l1919_191907


namespace NUMINAMATH_CALUDE_sum_of_first_12_terms_of_arithmetic_sequence_l1919_191997

/-- Given the sum of the first 4 terms and the sum of the first 8 terms of an arithmetic sequence,
    this theorem proves that the sum of the first 12 terms is 210. -/
theorem sum_of_first_12_terms_of_arithmetic_sequence 
  (S₄ S₈ : ℕ) (h₁ : S₄ = 30) (h₂ : S₈ = 100) : ∃ S₁₂ : ℕ, S₁₂ = 210 :=
by
  sorry


end NUMINAMATH_CALUDE_sum_of_first_12_terms_of_arithmetic_sequence_l1919_191997


namespace NUMINAMATH_CALUDE_total_signup_combinations_l1919_191960

/-- The number of ways for one person to sign up -/
def signup_options : ℕ := 2

/-- The number of people signing up -/
def num_people : ℕ := 3

/-- Theorem: The total number of different ways for three people to sign up, 
    each with two independent choices, is 8 -/
theorem total_signup_combinations : signup_options ^ num_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_signup_combinations_l1919_191960


namespace NUMINAMATH_CALUDE_problem_solution_l1919_191929

def p (a : ℝ) : Prop := ∀ x ≥ 1, x - x^2 ≤ a

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + 1 = 0

theorem problem_solution (a : ℝ) :
  (¬(¬(p a)) → a ≥ 0) ∧
  ((¬(p a ∧ q a) ∧ (p a ∨ q a)) → (a ≤ -2 ∨ (0 ≤ a ∧ a < 2))) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1919_191929


namespace NUMINAMATH_CALUDE_jimmy_sandwiches_l1919_191998

/-- The number of sandwiches Jimmy can make given the number of bread packs,
    slices per pack, and slices needed per sandwich. -/
def sandwiches_made (bread_packs : ℕ) (slices_per_pack : ℕ) (slices_per_sandwich : ℕ) : ℕ :=
  (bread_packs * slices_per_pack) / slices_per_sandwich

/-- Theorem stating that Jimmy made 8 sandwiches under the given conditions. -/
theorem jimmy_sandwiches :
  sandwiches_made 4 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_sandwiches_l1919_191998


namespace NUMINAMATH_CALUDE_complex_square_l1919_191935

theorem complex_square : (2 + Complex.I) ^ 2 = 3 + 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l1919_191935


namespace NUMINAMATH_CALUDE_circle_containment_l1919_191955

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point is inside a circle if its distance from the center is less than the radius --/
def is_inside (p : ℝ × ℝ) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

theorem circle_containment (circles : Fin 6 → Circle) 
  (O : ℝ × ℝ) (h : ∀ i, is_inside O (circles i)) :
  ∃ i j, i ≠ j ∧ is_inside (circles j).center (circles i) := by
  sorry

end NUMINAMATH_CALUDE_circle_containment_l1919_191955


namespace NUMINAMATH_CALUDE_curve_transformation_l1919_191938

/-- Given a 2x2 matrix A and its inverse, prove that if A transforms a curve F to y = 2x, then F is y = -3x -/
theorem curve_transformation (a b : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; 7, 3]
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![b, -2; -7, a]
  A * A_inv = 1 →
  (∀ x y : ℝ, (A.mulVec ![x, y] = ![x', y'] ∧ y' = 2*x') → y = -3*x) :=
by sorry

end NUMINAMATH_CALUDE_curve_transformation_l1919_191938


namespace NUMINAMATH_CALUDE_count_words_theorem_l1919_191925

/-- The set of all available letters -/
def Letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

/-- The set of consonants -/
def Consonants : Finset Char := {'B', 'C', 'D', 'F'}

/-- The set of vowels -/
def Vowels : Finset Char := {'A', 'E'}

/-- The length of words we're considering -/
def WordLength : Nat := 5

/-- Function to count the number of 5-letter words with at least two consonants -/
def count_words_with_at_least_two_consonants : Nat :=
  sorry

/-- Theorem stating that the number of 5-letter words with at least two consonants is 7424 -/
theorem count_words_theorem : count_words_with_at_least_two_consonants = 7424 := by
  sorry

end NUMINAMATH_CALUDE_count_words_theorem_l1919_191925


namespace NUMINAMATH_CALUDE_three_digit_squares_ending_in_self_l1919_191961

theorem three_digit_squares_ending_in_self (A : ℕ) : 
  (100 ≤ A ∧ A < 1000) ∧ (A^2 ≡ A [ZMOD 1000]) ↔ (A = 376 ∨ A = 625) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_squares_ending_in_self_l1919_191961


namespace NUMINAMATH_CALUDE_cars_combined_efficiency_l1919_191911

/-- Calculates the combined fuel efficiency of three cars given their individual efficiencies -/
def combinedFuelEfficiency (e1 e2 e3 : ℚ) : ℚ :=
  3 / (1 / e1 + 1 / e2 + 1 / e3)

/-- Theorem: The combined fuel efficiency of cars with 30, 15, and 20 mpg is 20 mpg -/
theorem cars_combined_efficiency :
  combinedFuelEfficiency 30 15 20 = 20 := by
  sorry

#eval combinedFuelEfficiency 30 15 20

end NUMINAMATH_CALUDE_cars_combined_efficiency_l1919_191911


namespace NUMINAMATH_CALUDE_problem_solution_l1919_191902

theorem problem_solution (x y : ℕ) (h1 : x > y) (h2 : x + x * y = 391) : x + y = 39 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1919_191902


namespace NUMINAMATH_CALUDE_expression_positivity_l1919_191982

theorem expression_positivity (x y z : ℝ) (h : x^2 + y^2 + z^2 ≠ 0) :
  5*x^2 + 5*y^2 + 5*z^2 + 6*x*y - 8*x*z - 8*y*z > 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_positivity_l1919_191982


namespace NUMINAMATH_CALUDE_sin_product_theorem_l1919_191996

theorem sin_product_theorem :
  Real.sin (10 * Real.pi / 180) *
  Real.sin (30 * Real.pi / 180) *
  Real.sin (50 * Real.pi / 180) *
  Real.sin (70 * Real.pi / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_theorem_l1919_191996


namespace NUMINAMATH_CALUDE_circle_center_sum_l1919_191979

/-- Given a circle with equation x² + y² = 4x - 6y + 9, 
    prove that the sum of the coordinates of its center is -1 -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) → (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 9) ∧ h + k = -1) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1919_191979


namespace NUMINAMATH_CALUDE_julio_is_ten_l1919_191992

-- Define the ages as natural numbers
def zipporah_age : ℕ := 7
def dina_age : ℕ := 51 - zipporah_age
def julio_age : ℕ := 54 - dina_age

-- State the theorem
theorem julio_is_ten : julio_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_julio_is_ten_l1919_191992


namespace NUMINAMATH_CALUDE_babysitter_scream_ratio_l1919_191927

-- Define the variables and constants
def current_rate : ℚ := 16
def new_rate : ℚ := 12
def scream_cost : ℚ := 3
def hours : ℚ := 6
def cost_difference : ℚ := 18

-- Define the theorem
theorem babysitter_scream_ratio :
  let current_cost := current_rate * hours
  let new_cost_without_screams := new_rate * hours
  let new_cost_with_screams := new_cost_without_screams + cost_difference
  let scream_total_cost := new_cost_with_screams - new_cost_without_screams
  let num_screams := scream_total_cost / scream_cost
  num_screams / hours = 1 := by
  sorry

end NUMINAMATH_CALUDE_babysitter_scream_ratio_l1919_191927


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l1919_191983

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem similar_triangle_perimeter
  (t1 : Triangle)
  (h1 : t1.isIsosceles)
  (h2 : t1.a = 12 ∧ t1.b = 12 ∧ t1.c = 18)
  (t2 : Triangle)
  (h3 : areSimilar t1 t2)
  (h4 : min t2.a (min t2.b t2.c) = 30) :
  t2.perimeter = 120 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l1919_191983


namespace NUMINAMATH_CALUDE_min_value_theorem_l1919_191993

theorem min_value_theorem (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + 2*y = 1) :
  (1 / (x + 1) + 1 / y) ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1919_191993


namespace NUMINAMATH_CALUDE_isabel_ds_games_left_l1919_191949

/-- Given that Isabel initially had 90 DS games and gave 87 to her friend,
    prove that she has 3 DS games left. -/
theorem isabel_ds_games_left (initial_games : ℕ) (games_given : ℕ) (games_left : ℕ) : 
  initial_games = 90 → games_given = 87 → games_left = initial_games - games_given → games_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_isabel_ds_games_left_l1919_191949


namespace NUMINAMATH_CALUDE_expand_a_expand_b_expand_c_expand_d_expand_e_l1919_191989

-- Define variables
variable (x y m n : ℝ)

-- Theorem for expression (a)
theorem expand_a : (x + 3*y)^2 = x^2 + 6*x*y + 9*y^2 := by sorry

-- Theorem for expression (b)
theorem expand_b : (2*x + 3*y)^2 = 4*x^2 + 12*x*y + 9*y^2 := by sorry

-- Theorem for expression (c)
theorem expand_c : (m^3 + n^5)^2 = m^6 + 2*m^3*n^5 + n^10 := by sorry

-- Theorem for expression (d)
theorem expand_d : (5*x - 3*y)^2 = 25*x^2 - 30*x*y + 9*y^2 := by sorry

-- Theorem for expression (e)
theorem expand_e : (3*m^5 - 4*n^2)^2 = 9*m^10 - 24*m^5*n^2 + 16*n^4 := by sorry

end NUMINAMATH_CALUDE_expand_a_expand_b_expand_c_expand_d_expand_e_l1919_191989


namespace NUMINAMATH_CALUDE_largest_factor_is_large_barrel_capacity_l1919_191909

def total_oil : ℕ := 95
def small_barrel_capacity : ℕ := 5
def small_barrels_used : ℕ := 1

def remaining_oil : ℕ := total_oil - (small_barrel_capacity * small_barrels_used)

def is_valid_large_barrel_capacity (capacity : ℕ) : Prop :=
  capacity > small_barrel_capacity ∧ 
  remaining_oil % capacity = 0 ∧
  capacity ≤ remaining_oil

theorem largest_factor_is_large_barrel_capacity : 
  ∃ (large_barrel_capacity : ℕ), 
    is_valid_large_barrel_capacity large_barrel_capacity ∧
    ∀ (x : ℕ), is_valid_large_barrel_capacity x → x ≤ large_barrel_capacity := by
  sorry

end NUMINAMATH_CALUDE_largest_factor_is_large_barrel_capacity_l1919_191909


namespace NUMINAMATH_CALUDE_equal_expressions_count_l1919_191950

theorem equal_expressions_count (x : ℝ) (h : x > 0) : 
  (∃! (count : ℕ), count = 2 ∧ 
    count = (Bool.toNat (2 * x^x = x^x + x^x) + 
             Bool.toNat (x^(x+1) = x^x + x^x) + 
             Bool.toNat ((x+1)^x = x^x + x^x) + 
             Bool.toNat (x^(2*(x+1)) = x^x + x^x))) :=
by sorry

end NUMINAMATH_CALUDE_equal_expressions_count_l1919_191950


namespace NUMINAMATH_CALUDE_triangle_AOB_properties_l1919_191958

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem triangle_AOB_properties :
  let magnitude_AB := Real.sqrt ((AB.1)^2 + (AB.2)^2)
  let dot_product_AB_OA := AB.1 * OA.1 + AB.2 * OA.2
  let cos_angle_OA_OB := (OA.1 * OB.1 + OA.2 * OB.2) / 
    (Real.sqrt (OA.1^2 + OA.2^2) * Real.sqrt (OB.1^2 + OB.2^2))
  (magnitude_AB = Real.sqrt 5) ∧
  (dot_product_AB_OA = 0) ∧
  (cos_angle_OA_OB = Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_AOB_properties_l1919_191958


namespace NUMINAMATH_CALUDE_triangle_division_result_l1919_191934

-- Define the process of dividing triangles
def divide_triangles (n : ℕ) : ℕ := 3^n

-- Define the side length after n iterations
def side_length (n : ℕ) : ℚ := 1 / 2^n

-- Theorem statement
theorem triangle_division_result :
  let iterations : ℕ := 12
  let final_count : ℕ := divide_triangles iterations
  let final_side_length : ℚ := side_length iterations
  final_count = 531441 ∧ final_side_length = 1 / 2^12 := by
  sorry


end NUMINAMATH_CALUDE_triangle_division_result_l1919_191934


namespace NUMINAMATH_CALUDE_jean_calories_consumption_l1919_191969

/-- Calculates the total calories consumed by Jean while writing her paper. -/
def total_calories (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

/-- Proves that Jean consumes 900 calories while writing her paper. -/
theorem jean_calories_consumption :
  total_calories 12 2 150 = 900 := by
  sorry

#eval total_calories 12 2 150

end NUMINAMATH_CALUDE_jean_calories_consumption_l1919_191969


namespace NUMINAMATH_CALUDE_y_value_proof_l1919_191971

theorem y_value_proof : ∀ y : ℚ, (1/3 - 1/4 = 4/y) → y = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l1919_191971


namespace NUMINAMATH_CALUDE_prob_heart_spade_king_two_draws_l1919_191919

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ := 52)
  (target_cards : ℕ := 28)

/-- Calculates the probability of drawing at least one target card in two draws with replacement -/
def prob_at_least_one (d : Deck) : ℚ :=
  1 - (1 - d.target_cards / d.total_cards) ^ 2

/-- Theorem stating the probability of drawing at least one heart, spade, or king in two draws -/
theorem prob_heart_spade_king_two_draws :
  prob_at_least_one (Deck.mk 52 28) = 133 / 169 := by
  sorry

#eval prob_at_least_one (Deck.mk 52 28)

end NUMINAMATH_CALUDE_prob_heart_spade_king_two_draws_l1919_191919


namespace NUMINAMATH_CALUDE_not_quadratic_radical_l1919_191962

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop := x ≥ 0

-- State the theorem
theorem not_quadratic_radical : ¬ is_quadratic_radical (-4) := by
  sorry

end NUMINAMATH_CALUDE_not_quadratic_radical_l1919_191962


namespace NUMINAMATH_CALUDE_expected_value_of_marbles_l1919_191954

/-- The set of marble numbers -/
def MarbleSet : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The number of marbles to draw -/
def DrawCount : ℕ := 3

/-- The sum of a combination of marbles -/
def CombinationSum (c : Finset ℕ) : ℕ := c.sum id

/-- The expected value of the sum of drawn marbles -/
noncomputable def ExpectedValue : ℚ :=
  (MarbleSet.powerset.filter (λ s => s.card = DrawCount)).sum CombinationSum /
   (MarbleSet.powerset.filter (λ s => s.card = DrawCount)).card

/-- Theorem: The expected value of the sum of three randomly drawn marbles is 10.5 -/
theorem expected_value_of_marbles : ExpectedValue = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_marbles_l1919_191954


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l1919_191947

/-- 
Given an arithmetic sequence {a_n} with a₁ = -8 and a₂ = -6,
if x is added to a₁, a₄, and a₅ to form a geometric sequence,
then x = -1.
-/
theorem arithmetic_to_geometric_sequence (a : ℕ → ℤ) (x : ℤ) : 
  a 1 = -8 →
  a 2 = -6 →
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →
  ((-8 + x) * x = (-2 + x)^2) →
  ((-2 + x)^2 = x * x) →
  x = -1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l1919_191947


namespace NUMINAMATH_CALUDE_sentences_per_paragraph_l1919_191920

/-- Represents the reading speed in sentences per hour -/
def reading_speed : ℕ := 200

/-- Represents the number of pages in the book -/
def pages : ℕ := 50

/-- Represents the number of paragraphs per page -/
def paragraphs_per_page : ℕ := 20

/-- Represents the total time taken to read the book in hours -/
def total_reading_time : ℕ := 50

/-- Theorem stating the number of sentences per paragraph -/
theorem sentences_per_paragraph : 
  (reading_speed * total_reading_time) / (pages * paragraphs_per_page) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sentences_per_paragraph_l1919_191920


namespace NUMINAMATH_CALUDE_losing_position_characterization_l1919_191988

/-- Represents the state of the table-folding game -/
structure GameState where
  n : ℕ
  m : ℕ

/-- Predicate to determine if a game state is a losing position -/
def is_losing_position (state : GameState) : Prop :=
  ∃ k : ℕ, state.m = (state.n + 1) * 2^k - 1

/-- The main theorem stating the characterization of losing positions -/
theorem losing_position_characterization (state : GameState) :
  is_losing_position state ↔ 
  (∀ fold : ℕ, fold > 0 → fold ≤ state.m → 
    ¬is_losing_position ⟨state.n, state.m - fold⟩) ∧
  (∀ fold : ℕ, fold > 0 → fold ≤ state.n → 
    ¬is_losing_position ⟨state.n - fold, state.m⟩) :=
sorry

end NUMINAMATH_CALUDE_losing_position_characterization_l1919_191988


namespace NUMINAMATH_CALUDE_johns_weight_l1919_191995

theorem johns_weight (john mark : ℝ) 
  (h1 : john + mark = 240)
  (h2 : john - mark = john / 3) : 
  john = 144 := by
sorry

end NUMINAMATH_CALUDE_johns_weight_l1919_191995


namespace NUMINAMATH_CALUDE_manuscript_productivity_l1919_191916

/-- Given a manuscript with 60,000 words, written over 120 hours including 20 hours of breaks,
    the average productivity during actual writing time is 600 words per hour. -/
theorem manuscript_productivity (total_words : ℕ) (total_hours : ℕ) (break_hours : ℕ)
    (h1 : total_words = 60000)
    (h2 : total_hours = 120)
    (h3 : break_hours = 20) :
  (total_words : ℝ) / (total_hours - break_hours : ℝ) = 600 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_productivity_l1919_191916


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1919_191910

/-- Given a line segment from (0, 2) to (3, y) with length 10 and y > 0, prove y = 2 + √91 -/
theorem line_segment_endpoint (y : ℝ) (h1 : y > 0) : 
  (((3 - 0)^2 + (y - 2)^2 : ℝ) = 10^2) → y = 2 + Real.sqrt 91 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1919_191910


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1919_191948

theorem quadratic_equation_roots (k : ℝ) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 3 * x₁^2 + k * x₁ = 5 ∧ 3 * x₂^2 + k * x₂ = 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1919_191948


namespace NUMINAMATH_CALUDE_smallest_marble_count_l1919_191984

/-- Represents the number of marbles of each color -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the total number of marbles -/
def total_marbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green

/-- Checks if the given marble count satisfies the equal probability conditions -/
def satisfies_conditions (mc : MarbleCount) : Prop :=
  let r := mc.red
  let w := mc.white
  let b := mc.blue
  let g := mc.green
  (r * (r - 1) * (r - 2) * (r - 3)) / 24 = 
    (w * r * (r - 1) * (r - 2)) / 6 ∧
  (r * (r - 1) * (r - 2) * (r - 3)) / 24 = 
    (w * b * r * (r - 1)) / 2 ∧
  (w * b * r * (r - 1)) / 2 = 
    w * b * g * r

theorem smallest_marble_count : 
  ∃ (mc : MarbleCount), 
    satisfies_conditions mc ∧ 
    total_marbles mc = 21 ∧ 
    (∀ (mc' : MarbleCount), satisfies_conditions mc' → total_marbles mc' ≥ 21) := by
  sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l1919_191984


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_l1919_191942

theorem sum_of_a_and_b_is_one :
  ∀ (a b : ℝ),
  (∃ (x : ℝ), x = a + Real.sqrt b) →
  (a + Real.sqrt b + (a - Real.sqrt b) = -4) →
  ((a + Real.sqrt b) * (a - Real.sqrt b) = 1) →
  a + b = 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_l1919_191942


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1919_191926

theorem rectangle_dimensions (x y : ℕ+) : 
  x * y = 36 ∧ x + y = 13 → (x = 9 ∧ y = 4) ∨ (x = 4 ∧ y = 9) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1919_191926


namespace NUMINAMATH_CALUDE_range_of_a_when_p_and_q_false_l1919_191932

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 2

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∃! x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a x = 0

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (1/2) (3/2) → x^2 + 3*(a+1)*x + 2 ≤ 0

-- Theorem statement
theorem range_of_a_when_p_and_q_false :
  ∀ a : ℝ, ¬(p a ∧ q a) → a > -5/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_p_and_q_false_l1919_191932


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1919_191941

theorem rectangle_dimensions :
  ∀ (x y : ℝ), 
    x > 0 ∧ y > 0 →
    x * y = 1/9 →
    y = 3 * x →
    x = Real.sqrt 3 / 9 ∧ y = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1919_191941


namespace NUMINAMATH_CALUDE_joe_paint_usage_l1919_191923

theorem joe_paint_usage (total_paint : ℝ) (used_paint : ℝ) 
  (h1 : total_paint = 360)
  (h2 : used_paint = 225) : 
  ∃ (first_week_fraction : ℝ),
    first_week_fraction * total_paint + 
    (1 / 2) * (total_paint - first_week_fraction * total_paint) = used_paint ∧
    first_week_fraction = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_joe_paint_usage_l1919_191923


namespace NUMINAMATH_CALUDE_inequality_preservation_l1919_191908

theorem inequality_preservation (a b : ℝ) : a < b → 1 - a > 1 - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1919_191908


namespace NUMINAMATH_CALUDE_susan_vacation_pay_missed_l1919_191906

/-- Calculate the pay missed during Susan's vacation --/
theorem susan_vacation_pay_missed
  (vacation_length : ℕ) -- Length of vacation in weeks
  (work_days_per_week : ℕ) -- Number of work days per week
  (paid_vacation_days : ℕ) -- Number of paid vacation days
  (hourly_rate : ℚ) -- Hourly pay rate
  (hours_per_day : ℕ) -- Number of work hours per day
  (h1 : vacation_length = 2)
  (h2 : work_days_per_week = 5)
  (h3 : paid_vacation_days = 6)
  (h4 : hourly_rate = 15)
  (h5 : hours_per_day = 8) :
  (vacation_length * work_days_per_week - paid_vacation_days) * (hourly_rate * hours_per_day) = 480 :=
by sorry

end NUMINAMATH_CALUDE_susan_vacation_pay_missed_l1919_191906


namespace NUMINAMATH_CALUDE_cube_guessing_game_l1919_191959

/-- The maximum amount Alexei can guarantee himself in the cube guessing game -/
def maxGuaranteedAmount (m : ℕ) (n : ℕ) : ℚ :=
  2^m / (Nat.choose m n)

/-- The problem statement for the cube guessing game -/
theorem cube_guessing_game (n : ℕ) (hn : n ≤ 100) :
  /- Part a: One blue cube -/
  maxGuaranteedAmount 100 1 = 2^100 / 100 ∧
  /- Part b: n blue cubes -/
  maxGuaranteedAmount 100 n = 2^100 / (Nat.choose 100 n) :=
sorry

end NUMINAMATH_CALUDE_cube_guessing_game_l1919_191959


namespace NUMINAMATH_CALUDE_sphere_volume_l1919_191946

theorem sphere_volume (R : ℝ) (h : R = 3) : (4 / 3 : ℝ) * Real.pi * R^3 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l1919_191946


namespace NUMINAMATH_CALUDE_geometric_mean_of_one_and_nine_l1919_191900

theorem geometric_mean_of_one_and_nine :
  ∃ (c : ℝ), c^2 = 1 * 9 ∧ (c = 3 ∨ c = -3) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_one_and_nine_l1919_191900


namespace NUMINAMATH_CALUDE_subtraction_mistake_l1919_191974

/-- Given two two-digit numbers, if the first number is misread by increasing both digits by 3
    and the incorrect subtraction results in 44, then the correct subtraction equals 11. -/
theorem subtraction_mistake (A B C D : ℕ) : 
  (A < 10) → (B < 10) → (C < 10) → (D < 10) →
  ((10 * (A + 3) + (B + 3)) - (10 * C + D) = 44) →
  ((10 * A + B) - (10 * C + D) = 11) := by
sorry

end NUMINAMATH_CALUDE_subtraction_mistake_l1919_191974


namespace NUMINAMATH_CALUDE_sum_after_removal_is_perfect_square_l1919_191953

-- Define the set M
def M : Set Nat := {n | 1 ≤ n ∧ n ≤ 2017}

-- Define the sum of all elements in M
def sum_M : Nat := (2017 * 2018) / 2

-- Define the element to be removed
def removed_element : Nat := 1677

-- Theorem to prove
theorem sum_after_removal_is_perfect_square :
  ∃ k : Nat, sum_M - removed_element = k^2 ∧ removed_element ∈ M :=
sorry

end NUMINAMATH_CALUDE_sum_after_removal_is_perfect_square_l1919_191953


namespace NUMINAMATH_CALUDE_f_property_l1919_191985

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 3

-- State the theorem
theorem f_property (a b : ℝ) : f a b (-2) = 7 → f a b 2 = -13 := by
  sorry

end NUMINAMATH_CALUDE_f_property_l1919_191985


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_translation_l1919_191976

theorem fixed_point_of_exponential_translation (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 + a^(x - 1)
  f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_translation_l1919_191976


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_existence_condition_l1919_191912

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |2*x + a|

-- Part I
theorem solution_set_for_a_equals_one :
  let a := 1
  {x : ℝ | f a x ≥ 5} = Set.Ici 2 ∪ Set.Iic (-4/3) := by sorry

-- Part II
theorem range_of_a_for_existence_condition :
  (∃ x₀ : ℝ, f a x₀ + |x₀ - 2| < 3) ↔ -7 < a ∧ a < -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_existence_condition_l1919_191912


namespace NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_4_and_5_l1919_191975

theorem smallest_three_digit_divisible_by_4_and_5 :
  ∃ n : ℕ, n = 100 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 4 ∣ m ∧ 5 ∣ m → n ≤ m) ∧
  4 ∣ n ∧ 5 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_4_and_5_l1919_191975


namespace NUMINAMATH_CALUDE_exists_double_application_square_l1919_191904

theorem exists_double_application_square :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_double_application_square_l1919_191904


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1919_191937

theorem cubic_equation_solution :
  ∃ x : ℝ, 2 * x^3 + 24 * x = 3 - 12 * x^2 ∧ x = Real.rpow (19/2) (1/3) - 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1919_191937


namespace NUMINAMATH_CALUDE_coefficient_sum_l1919_191981

theorem coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10 + a₁₁*(x+1)^11) →
  a₁ + a₂ + a₁₁ = 781 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_sum_l1919_191981


namespace NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l1919_191921

def f (x : ℝ) := x^2 + 4*x + 3

theorem f_increasing_on_positive_reals :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l1919_191921


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1919_191940

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties 
  (a : ℕ → ℤ) (d : ℤ) (h : arithmetic_sequence a d) :
  (a 5 = -1 ∧ a 8 = 2 → a 1 = -5 ∧ d = 1) ∧
  (a 1 + a 6 = 12 ∧ a 4 = 7 → a 9 = 17) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1919_191940


namespace NUMINAMATH_CALUDE_geraldo_tea_consumption_l1919_191944

/-- Proves that given 20 gallons of tea poured into 80 containers, if Geraldo drinks 3.5 containers, he consumes 7 pints of tea. -/
theorem geraldo_tea_consumption 
  (total_gallons : ℝ) 
  (num_containers : ℝ) 
  (geraldo_containers : ℝ) 
  (gallons_to_pints : ℝ → ℝ) :
  total_gallons = 20 ∧ 
  num_containers = 80 ∧ 
  geraldo_containers = 3.5 ∧ 
  (∀ x, gallons_to_pints x = 8 * x) →
  geraldo_containers * (gallons_to_pints total_gallons / num_containers) = 7 :=
by sorry

end NUMINAMATH_CALUDE_geraldo_tea_consumption_l1919_191944


namespace NUMINAMATH_CALUDE_aizhai_bridge_investment_l1919_191918

/-- Converts a number to scientific notation with a specified number of significant figures -/
def to_scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

/-- Checks if two scientific notations are equal -/
def scientific_notation_eq (a : ℝ × ℤ) (b : ℝ × ℤ) : Prop :=
  sorry

theorem aizhai_bridge_investment :
  let investment := 1650000000
  let sig_figs := 3
  let result := to_scientific_notation investment sig_figs
  scientific_notation_eq result (1.65, 9) :=
sorry

end NUMINAMATH_CALUDE_aizhai_bridge_investment_l1919_191918


namespace NUMINAMATH_CALUDE_hall_length_l1919_191956

/-- Proves that given the conditions of the hall and verandah, the length of the hall is 20 meters -/
theorem hall_length (hall_breadth : ℝ) (verandah_width : ℝ) (flooring_rate : ℝ) (total_cost : ℝ) :
  hall_breadth = 15 →
  verandah_width = 2.5 →
  flooring_rate = 3.5 →
  total_cost = 700 →
  ∃ (hall_length : ℝ),
    hall_length = 20 ∧
    (hall_length + 2 * verandah_width) * (hall_breadth + 2 * verandah_width) -
    hall_length * hall_breadth = total_cost / flooring_rate :=
by sorry

end NUMINAMATH_CALUDE_hall_length_l1919_191956


namespace NUMINAMATH_CALUDE_victor_finished_last_l1919_191972

-- Define the set of runners
inductive Runner : Type
| Lotar : Runner
| Manfred : Runner
| Victor : Runner
| Jan : Runner
| Eddy : Runner

-- Define the relation "finished before"
def finished_before : Runner → Runner → Prop := sorry

-- State the conditions
axiom lotar_before_manfred : finished_before Runner.Lotar Runner.Manfred
axiom victor_after_jan : finished_before Runner.Jan Runner.Victor
axiom manfred_before_jan : finished_before Runner.Manfred Runner.Jan
axiom eddy_before_victor : finished_before Runner.Eddy Runner.Victor

-- Define what it means to finish last
def finished_last (r : Runner) : Prop :=
  ∀ other : Runner, other ≠ r → finished_before other r

-- State the theorem
theorem victor_finished_last :
  finished_last Runner.Victor :=
sorry

end NUMINAMATH_CALUDE_victor_finished_last_l1919_191972


namespace NUMINAMATH_CALUDE_middle_digit_is_two_l1919_191939

theorem middle_digit_is_two (ABCDE : ℕ) : 
  ABCDE ≥ 10000 ∧ ABCDE < 100000 →
  4 * (10 * ABCDE + 4) = 400000 + ABCDE →
  (ABCDE / 100) % 10 = 2 := by
sorry

end NUMINAMATH_CALUDE_middle_digit_is_two_l1919_191939


namespace NUMINAMATH_CALUDE_line_through_points_l1919_191931

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- Given distinct vectors p and q, if m*p + (5/6)*q lies on the line through p and q, then m = 1/6 -/
theorem line_through_points (p q : V) (m : ℝ) 
  (h_distinct : p ≠ q)
  (h_on_line : ∃ t : ℝ, m • p + (5/6) • q = p + t • (q - p)) :
  m = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1919_191931


namespace NUMINAMATH_CALUDE_parabola_symmetry_l1919_191936

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line with inclination angle -/
structure Line where
  angle : ℝ

/-- Function to check if a point is on the parabola -/
def on_parabola (para : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * para.p * pt.x

/-- Function to check if a line passes through a point -/
def passes_through (l : Line) (pt : Point) : Prop :=
  true -- Simplified for this problem

/-- Function to check if two points are symmetric with respect to a line -/
def symmetric_wrt_line (p1 p2 : Point) (l : Line) : Prop :=
  true -- Simplified for this problem

/-- Main theorem -/
theorem parabola_symmetry (para : Parabola) (l : Line) (p q : Point) :
  l.angle = π / 6 →
  passes_through l (Point.mk (para.p / 2) 0) →
  on_parabola para p →
  q = Point.mk 5 0 →
  symmetric_wrt_line p q l →
  para.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l1919_191936


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1919_191980

theorem sqrt_sum_inequality (x y α : ℝ) (h : Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α)) : 
  x + y ≥ 2 * α := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1919_191980


namespace NUMINAMATH_CALUDE_book_selection_probability_l1919_191978

theorem book_selection_probability :
  let n : ℕ := 12  -- Total number of books
  let k : ℕ := 6   -- Number of books each student selects
  let m : ℕ := 3   -- Number of books in common

  -- Probability of selecting exactly m books in common
  (Nat.choose n m * Nat.choose (n - m) (k - m) * Nat.choose (n - k) (k - m) : ℚ) /
  (Nat.choose n k * Nat.choose n k : ℚ) = 100 / 231 := by
sorry

end NUMINAMATH_CALUDE_book_selection_probability_l1919_191978


namespace NUMINAMATH_CALUDE_S_equals_zero_two_neg_two_l1919_191987

def imaginary_unit : ℂ := Complex.I

def S : Set ℂ := {z | ∃ n : ℤ, z = (imaginary_unit ^ n) + (imaginary_unit ^ (-n))}

theorem S_equals_zero_two_neg_two : S = {0, 2, -2} := by sorry

end NUMINAMATH_CALUDE_S_equals_zero_two_neg_two_l1919_191987


namespace NUMINAMATH_CALUDE_line_perp_to_plane_l1919_191945

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp : Line → Line → Prop)
variable (perpToPlane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- Define the intersection operation for lines
variable (intersect : Line → Line → Set Point)

-- Theorem statement
theorem line_perp_to_plane 
  (a b c : Line) (α : Plane) (A : Point) :
  perp c a → 
  perp c b → 
  subset a α → 
  subset b α → 
  intersect a b = {A} → 
  perpToPlane c α :=
sorry

end NUMINAMATH_CALUDE_line_perp_to_plane_l1919_191945


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_l1919_191967

/-- A trinomial ax^2 + bx + c is a perfect square if there exists a real number r
    such that ax^2 + bx + c = (√a * x + r)^2 for all x. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ r : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (Real.sqrt a * x + r)^2

theorem perfect_square_trinomial_m (m : ℝ) :
  IsPerfectSquareTrinomial 1 (-m) 16 → m = 8 ∨ m = -8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_l1919_191967


namespace NUMINAMATH_CALUDE_birthday_dinner_cost_l1919_191928

theorem birthday_dinner_cost (num_people : ℕ) (meal_cost drink_cost dessert_cost : ℚ) :
  num_people = 5 →
  meal_cost = 12 →
  drink_cost = 3 →
  dessert_cost = 5 →
  (num_people : ℚ) * (meal_cost + drink_cost + dessert_cost) = 100 := by
  sorry

end NUMINAMATH_CALUDE_birthday_dinner_cost_l1919_191928


namespace NUMINAMATH_CALUDE_linear_independence_exp_trig_l1919_191973

theorem linear_independence_exp_trig (α β : ℝ) (h : β ≠ 0) :
  ∀ (α₁ α₂ : ℝ), (∀ x : ℝ, α₁ * Real.exp (α * x) * Real.sin (β * x) + 
                           α₂ * Real.exp (α * x) * Real.cos (β * x) = 0) →
                 α₁ = 0 ∧ α₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_independence_exp_trig_l1919_191973


namespace NUMINAMATH_CALUDE_counterexample_existence_l1919_191943

theorem counterexample_existence : ∃ (n : ℕ), ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n + 2)) ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_existence_l1919_191943


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1919_191952

theorem quadratic_equation_roots (x : ℝ) : 
  (∃! r : ℝ, x^2 - 2*x + 1 = 0) ↔ (x^2 - 2*x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1919_191952


namespace NUMINAMATH_CALUDE_ducks_in_lake_l1919_191913

theorem ducks_in_lake (initial_ducks joining_ducks : ℕ) 
  (h1 : initial_ducks = 13)
  (h2 : joining_ducks = 20) : 
  initial_ducks + joining_ducks = 33 := by
sorry

end NUMINAMATH_CALUDE_ducks_in_lake_l1919_191913


namespace NUMINAMATH_CALUDE_pocket_money_mode_and_median_l1919_191966

def pocket_money : List ℕ := [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5, 6]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem pocket_money_mode_and_median :
  mode pocket_money = 2 ∧ median pocket_money = 3 := by sorry

end NUMINAMATH_CALUDE_pocket_money_mode_and_median_l1919_191966


namespace NUMINAMATH_CALUDE_unique_number_problem_l1919_191922

theorem unique_number_problem : ∃! (x : ℝ), x > 0 ∧ (((x^2 / 3)^3) / 9) = x :=
by sorry

end NUMINAMATH_CALUDE_unique_number_problem_l1919_191922


namespace NUMINAMATH_CALUDE_crow_probability_l1919_191999

/-- Represents the number of crows of each color on each tree -/
structure CrowCounts where
  birchWhite : ℕ
  birchBlack : ℕ
  oakWhite : ℕ
  oakBlack : ℕ

/-- The probability of the number of white crows on the birch returning to its initial count -/
def probReturnToInitial (c : CrowCounts) : ℚ :=
  (c.birchBlack * (c.oakBlack + 1) + c.birchWhite * (c.oakWhite + 1)) / (50 * 51)

/-- The probability of the number of white crows on the birch changing -/
def probChange (c : CrowCounts) : ℚ :=
  (c.birchBlack * c.oakWhite + c.birchWhite * c.oakBlack) / (50 * 51)

theorem crow_probability (c : CrowCounts) 
  (h1 : c.birchWhite + c.birchBlack = 50)
  (h2 : c.oakWhite + c.oakBlack = 50)
  (h3 : c.birchWhite > 0)
  (h4 : c.birchBlack ≥ c.birchWhite)
  (h5 : c.oakBlack ≥ c.oakWhite - 1) :
  probReturnToInitial c > probChange c := by
  sorry

end NUMINAMATH_CALUDE_crow_probability_l1919_191999


namespace NUMINAMATH_CALUDE_expression_value_l1919_191914

theorem expression_value : ∀ x y : ℝ, x = 2 ∧ y = 3 → x^3 + y^2 * (x^2 * y) = 116 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1919_191914


namespace NUMINAMATH_CALUDE_money_distribution_inconsistency_l1919_191994

/-- Represents the money distribution problem with aunts and children --/
theorem money_distribution_inconsistency 
  (jade_money : ℕ) 
  (julia_money : ℕ) 
  (jack_money : ℕ) 
  (john_money : ℕ) 
  (jane_money : ℕ) 
  (total_after : ℕ) 
  (aunt_mary_gift : ℕ) 
  (aunt_susan_gift : ℕ) 
  (h1 : jade_money = 38)
  (h2 : julia_money = jade_money / 2)
  (h3 : jack_money = 12)
  (h4 : john_money = 15)
  (h5 : jane_money = 20)
  (h6 : total_after = 225)
  (h7 : aunt_mary_gift = 65)
  (h8 : aunt_susan_gift = 70) : 
  ¬(∃ (aunt_lucy_gift : ℕ) (individual_gift : ℕ),
    jade_money + julia_money + jack_money + john_money + jane_money + 
    aunt_mary_gift + aunt_susan_gift + aunt_lucy_gift = total_after ∧
    aunt_mary_gift + aunt_susan_gift + aunt_lucy_gift = 5 * individual_gift) :=
sorry


end NUMINAMATH_CALUDE_money_distribution_inconsistency_l1919_191994


namespace NUMINAMATH_CALUDE_hunter_saw_twelve_ants_l1919_191903

/-- The number of ants Hunter saw in the playground -/
def ants_seen (spiders ladybugs_initial ladybugs_left total_insects : ℕ) : ℕ :=
  total_insects - spiders - ladybugs_left

/-- Theorem stating that Hunter saw 12 ants given the problem conditions -/
theorem hunter_saw_twelve_ants :
  let spiders : ℕ := 3
  let ladybugs_initial : ℕ := 8
  let ladybugs_left : ℕ := ladybugs_initial - 2
  let total_insects : ℕ := 21
  ants_seen spiders ladybugs_initial ladybugs_left total_insects = 12 := by
  sorry

end NUMINAMATH_CALUDE_hunter_saw_twelve_ants_l1919_191903


namespace NUMINAMATH_CALUDE_ship_length_l1919_191905

theorem ship_length (emily_step : ℝ) (ship_step : ℝ) :
  let emily_forward := 150
  let emily_backward := 70
  let wind_factor := 0.9
  let ship_length := 150 * emily_step - 150 * ship_step
  emily_backward * emily_step = ship_length - emily_backward * ship_step * wind_factor
  →
  ship_length = 19950 / 213 * emily_step :=
by sorry

end NUMINAMATH_CALUDE_ship_length_l1919_191905


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l1919_191990

/-- Given a circle with original radius r₀ > 0, prove that when the radius is
    increased by 50%, the ratio of the new circumference to the new area
    is equal to 4 / (3r₀). -/
theorem circle_ratio_after_increase (r₀ : ℝ) (h : r₀ > 0) :
  let new_radius := 1.5 * r₀
  let new_circumference := 2 * Real.pi * new_radius
  let new_area := Real.pi * new_radius ^ 2
  new_circumference / new_area = 4 / (3 * r₀) := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l1919_191990


namespace NUMINAMATH_CALUDE_amusement_park_tickets_l1919_191970

/-- Proves that the number of children in the group is 15 given the specified conditions -/
theorem amusement_park_tickets (total_cost adult_price child_price adult_child_difference : ℕ) 
  (h1 : total_cost = 720)
  (h2 : adult_price = 15)
  (h3 : child_price = 8)
  (h4 : adult_child_difference = 25) :
  ∃ (num_children : ℕ), 
    (num_children + adult_child_difference) * adult_price + num_children * child_price = total_cost ∧ 
    num_children = 15 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_tickets_l1919_191970


namespace NUMINAMATH_CALUDE_parkway_elementary_soccer_l1919_191963

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (boys_soccer_percentage : ℚ) :
  total_students = 420 →
  boys = 296 →
  soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  (total_students - boys) - (soccer_players - (boys_soccer_percentage * soccer_players).floor) = 89 := by
  sorry

end NUMINAMATH_CALUDE_parkway_elementary_soccer_l1919_191963


namespace NUMINAMATH_CALUDE_n_in_interval_l1919_191957

def is_repeating_decimal (q : ℚ) (period : ℕ) : Prop :=
  ∃ (k : ℕ), q * (10^period - 1) = k ∧ k < 10^period

theorem n_in_interval (n : ℕ) :
  n < 500 →
  is_repeating_decimal (1 / n) 4 →
  is_repeating_decimal (1 / (n + 4)) 2 →
  n ∈ Set.Icc 1 125 :=
sorry

end NUMINAMATH_CALUDE_n_in_interval_l1919_191957


namespace NUMINAMATH_CALUDE_absolute_value_identity_l1919_191964

theorem absolute_value_identity (x : ℝ) (h : x = 2023) : 
  |‖x‖ - x| - ‖x‖ - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_identity_l1919_191964


namespace NUMINAMATH_CALUDE_greatest_value_cubic_inequality_l1919_191991

theorem greatest_value_cubic_inequality :
  let f : ℝ → ℝ := λ b => -b^3 + b^2 + 7*b - 10
  ∃ (max_b : ℝ), max_b = 4 + Real.sqrt 6 ∧
    (∀ b : ℝ, f b ≥ 0 → b ≤ max_b) ∧
    f max_b ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_cubic_inequality_l1919_191991


namespace NUMINAMATH_CALUDE_difference_of_expressions_l1919_191901

theorem difference_of_expressions : 
  ((0.85 * 250)^2 / 2.3) - ((3/5 * 175) / 2.3) = 19587.5 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_expressions_l1919_191901
