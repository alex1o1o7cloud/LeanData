import Mathlib

namespace NUMINAMATH_CALUDE_cheese_balls_per_serving_l3526_352696

/-- Given the information about cheese balls in barrels, calculate the number of cheese balls per serving -/
theorem cheese_balls_per_serving 
  (barrel_24oz : ℕ) 
  (barrel_35oz : ℕ) 
  (servings_24oz : ℕ) 
  (cheese_balls_35oz : ℕ) 
  (h1 : barrel_24oz = 24) 
  (h2 : barrel_35oz = 35) 
  (h3 : servings_24oz = 60) 
  (h4 : cheese_balls_35oz = 1050) : 
  (cheese_balls_35oz / barrel_35oz * barrel_24oz) / servings_24oz = 12 := by
  sorry


end NUMINAMATH_CALUDE_cheese_balls_per_serving_l3526_352696


namespace NUMINAMATH_CALUDE_altitude_sum_of_triangle_l3526_352626

/-- The sum of altitudes of a triangle formed by the line 15x + 3y = 45 and the coordinate axes --/
theorem altitude_sum_of_triangle (x y : ℝ) : 
  (15 * x + 3 * y = 45) →  -- Line equation
  ∃ (a b c : ℝ), -- Altitudes
    (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) ∧ -- Altitudes are non-negative
    (a + b + c = (18 * Real.sqrt 26 + 15) / Real.sqrt 26) ∧ -- Sum of altitudes
    (∃ (x₁ y₁ : ℝ), 15 * x₁ + 3 * y₁ = 45 ∧ x₁ ≥ 0 ∧ y₁ ≥ 0) -- Triangle exists in the first quadrant
    :=
by sorry

end NUMINAMATH_CALUDE_altitude_sum_of_triangle_l3526_352626


namespace NUMINAMATH_CALUDE_season_games_l3526_352666

/-- The number of baseball games in a month -/
def games_per_month : ℕ := 7

/-- The number of months in a season -/
def months_in_season : ℕ := 2

/-- The total number of games in a season -/
def total_games : ℕ := games_per_month * months_in_season

theorem season_games : total_games = 14 := by sorry

end NUMINAMATH_CALUDE_season_games_l3526_352666


namespace NUMINAMATH_CALUDE_factorial_square_root_product_l3526_352643

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_square_root_product : (Real.sqrt (factorial 5 * factorial 4))^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_product_l3526_352643


namespace NUMINAMATH_CALUDE_min_seating_arrangement_l3526_352639

/-- Given a circular table with 60 chairs, this theorem proves that the smallest number of people
    that can be seated such that any additional person must sit next to someone is 20. -/
theorem min_seating_arrangement (n : ℕ) : n = 20 ↔ (
  n ≤ 60 ∧ 
  (∀ m : ℕ, m < n → ∃ (arrangement : Fin 60 → Bool), 
    (∃ i : Fin 60, ¬arrangement i) ∧ 
    (∀ i : Fin 60, arrangement i → 
      (arrangement (i + 1) ∨ arrangement (i + 59)))) ∧
  (∀ m : ℕ, m > n → ¬∃ (arrangement : Fin 60 → Bool),
    (∀ i : Fin 60, arrangement i → 
      (arrangement (i + 1) ∨ arrangement (i + 59)))))
  := by sorry


end NUMINAMATH_CALUDE_min_seating_arrangement_l3526_352639


namespace NUMINAMATH_CALUDE_zeros_of_g_l3526_352687

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x + b
def g (a b x : ℝ) : ℝ := b * x^2 - a * x

-- State the theorem
theorem zeros_of_g (a b : ℝ) (h : f a b 2 = 0) :
  (g a b 0 = 0) ∧ (g a b (-1/2) = 0) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_g_l3526_352687


namespace NUMINAMATH_CALUDE_system_solution_l3526_352693

theorem system_solution :
  let S : Set (ℝ × ℝ × ℝ) := {(x, y, z) | 
    x + y - z = 4 ∧
    x^2 - y^2 + z^2 = -4 ∧
    x * y * z = 6}
  S = {(2, 3, 1), (-1, 3, -2)} := by sorry

end NUMINAMATH_CALUDE_system_solution_l3526_352693


namespace NUMINAMATH_CALUDE_parabola_translation_l3526_352667

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola vertically by a given amount -/
def translateVertical (p : Parabola) (dy : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + dy }

/-- Translates a parabola horizontally by a given amount -/
def translateHorizontal (p : Parabola) (dx : ℝ) : Parabola :=
  { a := p.a, b := p.b - 2 * p.a * dx, c := p.c + p.a * dx^2 - p.b * dx }

theorem parabola_translation (p : Parabola) :
  p.a = 1 ∧ p.b = -2 ∧ p.c = 4 →
  let p' := translateHorizontal (translateVertical p 3) 1
  p'.a = 1 ∧ p'.b = -4 ∧ p'.c = 10 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l3526_352667


namespace NUMINAMATH_CALUDE_inequality_theorem_l3526_352690

theorem inequality_theorem (p q r s t u : ℝ) 
  (h1 : p^2 < s^2) (h2 : q^2 < t^2) (h3 : r^2 < u^2) :
  p^2 * q^2 + q^2 * r^2 + r^2 * p^2 < s^2 * t^2 + t^2 * u^2 + u^2 * s^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3526_352690


namespace NUMINAMATH_CALUDE_single_tool_users_count_l3526_352650

/-- The number of attendants who used a pencil -/
def pencil_users : ℕ := 25

/-- The number of attendants who used a pen -/
def pen_users : ℕ := 15

/-- The number of attendants who used both pencil and pen -/
def both_users : ℕ := 10

/-- The number of attendants who used only one type of writing tool -/
def single_tool_users : ℕ := (pencil_users - both_users) + (pen_users - both_users)

theorem single_tool_users_count : single_tool_users = 20 := by
  sorry

end NUMINAMATH_CALUDE_single_tool_users_count_l3526_352650


namespace NUMINAMATH_CALUDE_circle_radius_l3526_352627

theorem circle_radius (x y : ℝ) : 
  (∃ r, r > 0 ∧ ∀ x y, x^2 + y^2 - 4*x + 2*y + 2 = 0 ↔ (x - 2)^2 + (y + 1)^2 = r^2) →
  (∃ r, r > 0 ∧ ∀ x y, x^2 + y^2 - 4*x + 2*y + 2 = 0 ↔ (x - 2)^2 + (y + 1)^2 = r^2 ∧ r = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l3526_352627


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l3526_352609

def p (x : ℝ) : Prop := |x - 4| > 2

def q (x : ℝ) : Prop := x > 1

def not_p (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 6

theorem not_p_sufficient_not_necessary_for_q :
  (∀ x, not_p x → q x) ∧ ¬(∀ x, q x → not_p x) := by sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l3526_352609


namespace NUMINAMATH_CALUDE_square_side_length_average_l3526_352655

theorem square_side_length_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l3526_352655


namespace NUMINAMATH_CALUDE_total_students_l3526_352694

theorem total_students (general : ℕ) (biology : ℕ) (math : ℕ) : 
  general = 30 →
  biology = 2 * general →
  math = (3 * (general + biology)) / 5 →
  general + biology + math = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_total_students_l3526_352694


namespace NUMINAMATH_CALUDE_dandelion_seed_percentage_l3526_352611

def num_sunflowers : ℕ := 6
def num_dandelions : ℕ := 8
def seeds_per_sunflower : ℕ := 9
def seeds_per_dandelion : ℕ := 12

def total_seeds : ℕ := num_sunflowers * seeds_per_sunflower + num_dandelions * seeds_per_dandelion
def dandelion_seeds : ℕ := num_dandelions * seeds_per_dandelion

theorem dandelion_seed_percentage :
  (dandelion_seeds : ℚ) / total_seeds * 100 = 64 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_seed_percentage_l3526_352611


namespace NUMINAMATH_CALUDE_sum_interior_angles_polygon_l3526_352620

theorem sum_interior_angles_polygon (n : ℕ) (h : n ≥ 3) :
  (360 / 30 : ℕ) = n → (n - 2) * 180 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_polygon_l3526_352620


namespace NUMINAMATH_CALUDE_circle_diameter_l3526_352654

theorem circle_diameter (AE EB ED : ℝ) (h1 : AE = 2) (h2 : EB = 6) (h3 : ED = 3) :
  let AB := AE + EB
  let CE := (AE * EB) / ED
  let AM := (AB) / 2
  let OM := (AE + EB) / 2
  let OA := Real.sqrt (AM^2 + OM^2)
  let diameter := 2 * OA
  diameter = Real.sqrt 65 := by sorry

end NUMINAMATH_CALUDE_circle_diameter_l3526_352654


namespace NUMINAMATH_CALUDE_probability_green_is_9_31_l3526_352635

/-- Represents the number of jelly beans of each color in the bag -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the total number of jelly beans in the bag -/
def totalJellyBeans (bag : JellyBeanBag) : ℕ :=
  bag.red + bag.green + bag.yellow + bag.blue

/-- Calculates the probability of selecting a green jelly bean -/
def probabilityGreen (bag : JellyBeanBag) : ℚ :=
  bag.green / totalJellyBeans bag

/-- The specific bag of jelly beans described in the problem -/
def specificBag : JellyBeanBag :=
  { red := 10, green := 9, yellow := 5, blue := 7 }

/-- Theorem stating that the probability of selecting a green jelly bean
    from the specific bag is 9/31 -/
theorem probability_green_is_9_31 :
  probabilityGreen specificBag = 9 / 31 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_is_9_31_l3526_352635


namespace NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_l3526_352603

theorem x_positive_necessary_not_sufficient :
  (∃ x : ℝ, x > 0 ∧ (x - 2) * (x - 4) ≥ 0) ∧
  (∀ x : ℝ, (x - 2) * (x - 4) < 0 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_l3526_352603


namespace NUMINAMATH_CALUDE_binary_arithmetic_l3526_352607

/-- Addition and subtraction of binary numbers --/
theorem binary_arithmetic : 
  let a := 0b1101
  let b := 0b10
  let c := 0b101
  let d := 0b11
  let result := 0b1011
  (a + b + c) - d = result :=
by sorry

end NUMINAMATH_CALUDE_binary_arithmetic_l3526_352607


namespace NUMINAMATH_CALUDE_f_range_theorem_l3526_352600

def f (x m : ℝ) : ℝ := |x + 1| + |x - m|

theorem f_range_theorem :
  (∀ m : ℝ, (∀ x : ℝ, f x m ≥ 3) ↔ m ∈ Set.Ici 2 ∪ Set.Iic (-4)) ∧
  (∀ m : ℝ, (∃ x : ℝ, f m m - 2*m ≥ x^2 - x) ↔ m ∈ Set.Iic (5/4)) := by
  sorry

end NUMINAMATH_CALUDE_f_range_theorem_l3526_352600


namespace NUMINAMATH_CALUDE_correct_match_l3526_352661

/-- Represents a philosophical statement --/
structure PhilosophicalStatement :=
  (text : String)
  (interpretation : String)

/-- Checks if a statement represents seizing opportunity for qualitative change --/
def representsQualitativeChange (statement : PhilosophicalStatement) : Prop :=
  statement.interpretation = "Decisively seize the opportunity to promote qualitative change"

/-- Checks if a statement represents forward development --/
def representsForwardDevelopment (statement : PhilosophicalStatement) : Prop :=
  statement.interpretation = "The future is bright"

/-- The four given statements --/
def statement1 : PhilosophicalStatement :=
  { text := "As cold comes and heat goes, the four seasons change"
  , interpretation := "Things are developing" }

def statement2 : PhilosophicalStatement :=
  { text := "Thousands of flowers arranged, just waiting for the first thunder"
  , interpretation := "Decisively seize the opportunity to promote qualitative change" }

def statement3 : PhilosophicalStatement :=
  { text := "Despite the intention to plant flowers, they don't bloom; unintentionally planting willows, they grow into shade"
  , interpretation := "The unity of contradictions" }

def statement4 : PhilosophicalStatement :=
  { text := "There will be times when the strong winds break the waves, and we will sail across the sea with clouds"
  , interpretation := "The future is bright" }

/-- Theorem stating that statements 2 and 4 correctly match the required interpretations --/
theorem correct_match :
  representsQualitativeChange statement2 ∧
  representsForwardDevelopment statement4 :=
by sorry

end NUMINAMATH_CALUDE_correct_match_l3526_352661


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3526_352683

theorem inequality_and_equality_condition (x : ℝ) (hx : x ≠ 0) : 
  max 0 (Real.log (abs x)) ≥ 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (abs x) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (abs (x^2 - 1)) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ∧
  (max 0 (Real.log (abs x)) = 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (abs x) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (abs (x^2 - 1)) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ↔ 
  (x = (Real.sqrt 5 + 1) / 2 ∨ x = (Real.sqrt 5 - 1) / 2 ∨ 
   x = -(Real.sqrt 5 + 1) / 2 ∨ x = -(Real.sqrt 5 - 1) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3526_352683


namespace NUMINAMATH_CALUDE_boxes_opened_is_twelve_l3526_352638

/-- Calculates the number of boxes opened given the number of samples per box,
    the number of customers who tried a sample, and the number of samples left over. -/
def boxes_opened (samples_per_box : ℕ) (customers : ℕ) (samples_left : ℕ) : ℕ :=
  (customers + samples_left) / samples_per_box

/-- Proves that given the conditions, the number of boxes opened is 12. -/
theorem boxes_opened_is_twelve :
  boxes_opened 20 235 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_boxes_opened_is_twelve_l3526_352638


namespace NUMINAMATH_CALUDE_not_power_of_integer_l3526_352684

theorem not_power_of_integer (m : ℕ) : ¬ ∃ (n k : ℕ), m * (m + 1) = n^k := by
  sorry

end NUMINAMATH_CALUDE_not_power_of_integer_l3526_352684


namespace NUMINAMATH_CALUDE_john_total_needed_l3526_352685

/-- The amount of money John has, in dollars. -/
def john_has : ℚ := 0.75

/-- The additional amount John needs, in dollars. -/
def john_needs_more : ℚ := 1.75

/-- The total amount John needs is the sum of what he has and what he needs more. -/
theorem john_total_needed : john_has + john_needs_more = 2.50 := by
  sorry

end NUMINAMATH_CALUDE_john_total_needed_l3526_352685


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l3526_352625

/-- The area of a rectangular garden with length three times its width and width of 13 meters is 507 square meters. -/
theorem rectangular_garden_area : ∀ (width length area : ℝ),
  width = 13 →
  length = 3 * width →
  area = length * width →
  area = 507 := by sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l3526_352625


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3526_352662

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3526_352662


namespace NUMINAMATH_CALUDE_train_speed_l3526_352691

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 160) (h2 : time = 6) :
  length / time = 26.67 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3526_352691


namespace NUMINAMATH_CALUDE_variance_of_linear_transform_l3526_352632

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- A linear transformation of a random variable -/
structure LinearTransform (X : BinomialRV) where
  a : ℝ
  b : ℝ
  Y : ℝ := a * X.n + b

theorem variance_of_linear_transform (X : BinomialRV) (Y : LinearTransform X) :
  X.n = 5 ∧ X.p = 1/4 ∧ Y.a = 4 ∧ Y.b = -3 →
  Y.a^2 * variance X = 15 :=
sorry

end NUMINAMATH_CALUDE_variance_of_linear_transform_l3526_352632


namespace NUMINAMATH_CALUDE_cos_BAE_value_l3526_352699

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the point E on BC
def E (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the lengths of the sides
def AB (triangle : Triangle) : ℝ := 4
def AC (triangle : Triangle) : ℝ := 8
def BC (triangle : Triangle) : ℝ := 10

-- Define that AE bisects angle BAC
def AE_bisects_BAC (triangle : Triangle) : Prop := sorry

-- Define the cosine of angle BAE
def cos_BAE (triangle : Triangle) : ℝ := sorry

-- Theorem statement
theorem cos_BAE_value (triangle : Triangle) 
  (h1 : AB triangle = 4) 
  (h2 : AC triangle = 8) 
  (h3 : BC triangle = 10) 
  (h4 : AE_bisects_BAC triangle) : 
  cos_BAE triangle = Real.sqrt (11/32) := by
  sorry

end NUMINAMATH_CALUDE_cos_BAE_value_l3526_352699


namespace NUMINAMATH_CALUDE_mom_bought_51_shirts_l3526_352659

/-- The number of t-shirts in a package -/
def shirts_per_package : ℕ := 3

/-- The number of packages if t-shirts were purchased in packages -/
def num_packages : ℕ := 17

/-- The total number of t-shirts Mom bought -/
def total_shirts : ℕ := shirts_per_package * num_packages

theorem mom_bought_51_shirts : total_shirts = 51 := by
  sorry

end NUMINAMATH_CALUDE_mom_bought_51_shirts_l3526_352659


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3526_352608

theorem complex_equation_solution (a : ℝ) : 
  (a^2 - a : ℂ) + (3*a - 1 : ℂ)*Complex.I = 2 + 5*Complex.I → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3526_352608


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l3526_352653

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.3 * P) 
  (hN : N = 0.6 * (2 * P)) : 
  M / N = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l3526_352653


namespace NUMINAMATH_CALUDE_average_weight_increase_l3526_352604

/-- Theorem: Increase in average weight when replacing a person in a group -/
theorem average_weight_increase
  (n : ℕ)                           -- number of people in the group
  (initial_weight : ℝ)              -- weight of the person being replaced
  (new_weight : ℝ)                  -- weight of the new person
  (h1 : n = 8)                      -- there are 8 people in the group
  (h2 : initial_weight = 55)        -- the initial person weighs 55 kg
  (h3 : new_weight = 75)            -- the new person weighs 75 kg
  : (new_weight - initial_weight) / n = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3526_352604


namespace NUMINAMATH_CALUDE_sesame_mass_scientific_notation_l3526_352674

theorem sesame_mass_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.00000201 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.01 ∧ n = -6 :=
sorry

end NUMINAMATH_CALUDE_sesame_mass_scientific_notation_l3526_352674


namespace NUMINAMATH_CALUDE_orange_count_correct_l3526_352616

/-- The number of oranges in the box after adding and removing specified quantities -/
def final_oranges (initial added removed : ℝ) : ℝ :=
  initial + added - removed

/-- Theorem stating that the final number of oranges in the box is correct -/
theorem orange_count_correct (initial added removed : ℝ) :
  final_oranges initial added removed = initial + added - removed := by
  sorry

end NUMINAMATH_CALUDE_orange_count_correct_l3526_352616


namespace NUMINAMATH_CALUDE_scientific_notation_16907_l3526_352612

theorem scientific_notation_16907 :
  16907 = 1.6907 * (10 : ℝ)^4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_16907_l3526_352612


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l3526_352679

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m + 3) * x^(m^2 - 7) + m*x - 2 = a*x^2 + b*x + c) ∧ 
  (m + 3 ≠ 0) → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l3526_352679


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l3526_352656

/-- The number of eggs in a full container -/
def full_container : ℕ := 15

/-- The number of eggs in an underfilled container -/
def underfilled_container : ℕ := 14

/-- The number of underfilled containers -/
def num_underfilled : ℕ := 3

/-- The minimum number of eggs initially bought -/
def min_initial_eggs : ℕ := 151

theorem smallest_number_of_eggs (n : ℕ) (h : n > min_initial_eggs) : 
  (∃ (c : ℕ), n = c * full_container - num_underfilled * (full_container - underfilled_container)) →
  162 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l3526_352656


namespace NUMINAMATH_CALUDE_grocer_bananas_theorem_l3526_352641

/-- Represents the number of pounds of bananas purchased by the grocer -/
def bananas_purchased : ℝ := 96

/-- Represents the purchase price in dollars per 3 pounds of bananas -/
def purchase_price : ℝ := 0.50

/-- Represents the selling price in dollars per 4 pounds of bananas -/
def selling_price : ℝ := 1.00

/-- Represents the total profit in dollars -/
def total_profit : ℝ := 8.00

/-- Theorem stating that the number of pounds of bananas purchased by the grocer is 96 -/
theorem grocer_bananas_theorem :
  bananas_purchased = 96 ∧
  (selling_price / 4 - purchase_price / 3) * bananas_purchased = total_profit :=
sorry

end NUMINAMATH_CALUDE_grocer_bananas_theorem_l3526_352641


namespace NUMINAMATH_CALUDE_cylinder_volume_from_sheet_l3526_352652

/-- The volume of a cylinder formed by a rectangular sheet as its lateral surface -/
theorem cylinder_volume_from_sheet (length width : ℝ) (h : length = 12 ∧ width = 8) :
  ∃ (volume : ℝ), (volume = 192 / Real.pi ∨ volume = 288 / Real.pi) ∧
  ∃ (radius height : ℝ), 
    (2 * Real.pi * radius = width ∧ height = length ∧ volume = Real.pi * radius^2 * height) ∨
    (2 * Real.pi * radius = length ∧ height = width ∧ volume = Real.pi * radius^2 * height) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_sheet_l3526_352652


namespace NUMINAMATH_CALUDE_power_function_property_l3526_352670

/-- Given a power function f(x) = x^α where α ∈ ℝ, 
    if f(2) = √2, then f(4) = 2 -/
theorem power_function_property (α : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, f x = x ^ α) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l3526_352670


namespace NUMINAMATH_CALUDE_two_numbers_puzzle_l3526_352692

def is_two_digit_same_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ (n / 10 = n % 10)

def is_three_digit_same_digits (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100 = (n / 10) % 10) ∧ (n / 100 = n % 10)

theorem two_numbers_puzzle :
  ∀ a b : ℕ,
    a > 0 ∧ b > 0 →
    is_two_digit_same_digits (a + b) →
    is_three_digit_same_digits (a * b) →
    ((a = 37 ∧ b = 18) ∨ (a = 18 ∧ b = 37) ∨ (a = 74 ∧ b = 3) ∨ (a = 3 ∧ b = 74)) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_puzzle_l3526_352692


namespace NUMINAMATH_CALUDE_smallest_integer_in_special_set_l3526_352644

theorem smallest_integer_in_special_set : ∀ n : ℤ,
  (n + 6 > 2 * ((7 * n + 21) / 7)) →
  (∀ m : ℤ, m < n → m + 6 ≤ 2 * ((7 * m + 21) / 7)) →
  n = -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_special_set_l3526_352644


namespace NUMINAMATH_CALUDE_election_vote_count_l3526_352672

theorem election_vote_count : ∃ (total_votes : ℕ), 
  (total_votes > 0) ∧ 
  (∃ (candidate_votes rival_votes : ℕ),
    (candidate_votes = (total_votes * 3) / 10) ∧
    (rival_votes = candidate_votes + 4000) ∧
    (candidate_votes + rival_votes = total_votes) ∧
    (total_votes = 10000)) := by
  sorry

end NUMINAMATH_CALUDE_election_vote_count_l3526_352672


namespace NUMINAMATH_CALUDE_range_of_m_l3526_352642

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m = 0 ∧ x₂^2 - 2*x₂ + m = 0

def q (m : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (m + 2)*x₁ - 1 < (m + 2)*x₂ - 1

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (m ≤ -2 ∨ m ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3526_352642


namespace NUMINAMATH_CALUDE_product_of_solutions_l3526_352680

-- Define the equation
def equation (x : ℝ) : Prop :=
  (2 * x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 5)

-- Theorem statement
theorem product_of_solutions : 
  ∃ (x₁ x₂ : ℝ), equation x₁ ∧ equation x₂ ∧ x₁ * x₂ = 3 :=
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3526_352680


namespace NUMINAMATH_CALUDE_sum_of_segments_constant_l3526_352689

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- Checks if a point is inside a triangle -/
def isInterior (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Calculates the length of a segment from a vertex to the intersection
    of a parallel line through a point with the opposite side -/
def segmentLength (t : Triangle) (p : Point) (v : Point) : ℝ :=
  sorry

/-- Main theorem -/
theorem sum_of_segments_constant (t : Triangle) (p : Point) :
  isEquilateral t → isInterior p t →
  segmentLength t p t.A + segmentLength t p t.B + segmentLength t p t.C =
  distance t.A t.B :=
by sorry

end NUMINAMATH_CALUDE_sum_of_segments_constant_l3526_352689


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3526_352628

def f (x : ℝ) : ℝ := x^4 + 9*x^3 + 18*x^2 + 2023*x - 2021

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x^12 + 9*x^11 + 18*x^10 + 2023*x^9 - 2021*x^8 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_unique_positive_solution_l3526_352628


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l3526_352630

theorem rectangle_area_difference (l w : ℕ) : 
  (l + w = 25) →  -- Perimeter condition: 2l + 2w = 50 simplified
  (∃ (l' w' : ℕ), l' + w' = 25 ∧ l' * w' = 156) ∧  -- Existence of max area
  (∃ (l'' w'' : ℕ), l'' + w'' = 25 ∧ l'' * w'' = 24) ∧  -- Existence of min area
  (∀ (l''' w''' : ℕ), l''' + w''' = 25 → l''' * w''' ≤ 156) ∧  -- Max area condition
  (∀ (l'''' w'''' : ℕ), l'''' + w'''' = 25 → l'''' * w'''' ≥ 24) →  -- Min area condition
  156 - 24 = 132  -- Difference between max and min areas
  := by sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l3526_352630


namespace NUMINAMATH_CALUDE_gasoline_tank_capacity_l3526_352636

theorem gasoline_tank_capacity : ∃ x : ℚ, 
  (3/4 : ℚ) * x - (1/3 : ℚ) * x = 18 ∧ x = 43.2 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_tank_capacity_l3526_352636


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_is_120_l3526_352657

/-- Calculates the maximum number of people who can ride a Ferris wheel simultaneously -/
def max_ferris_wheel_capacity (total_seats : ℕ) (people_per_seat : ℕ) (broken_seats : ℕ) : ℕ :=
  (total_seats - broken_seats) * people_per_seat

/-- Proves that the maximum capacity of the Ferris wheel under given conditions is 120 people -/
theorem ferris_wheel_capacity_is_120 :
  max_ferris_wheel_capacity 18 15 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_is_120_l3526_352657


namespace NUMINAMATH_CALUDE_train_journey_duration_l3526_352617

-- Define the time type
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

-- Define the function to calculate time difference
def timeDifference (t1 t2 : Time) : Time :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  let diffMinutes := totalMinutes2 - totalMinutes1
  { hours := diffMinutes / 60, minutes := diffMinutes % 60 }

-- Theorem statement
theorem train_journey_duration :
  let departureTime := { hours := 9, minutes := 20 : Time }
  let arrivalTime := { hours := 11, minutes := 30 : Time }
  timeDifference departureTime arrivalTime = { hours := 2, minutes := 10 : Time } := by
  sorry


end NUMINAMATH_CALUDE_train_journey_duration_l3526_352617


namespace NUMINAMATH_CALUDE_expression_simplification_l3526_352686

theorem expression_simplification (x : ℝ) : 
  2 * x - 3 * (2 - x) + 4 * (2 + x) - 5 * (1 - 3 * x) = 24 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3526_352686


namespace NUMINAMATH_CALUDE_carlas_order_cost_l3526_352640

/-- Calculates the final cost of Carla's order at McDonald's --/
theorem carlas_order_cost (base_cost : ℝ) (coupon_discount : ℝ) (senior_discount_rate : ℝ) (swap_charge : ℝ)
  (h1 : base_cost = 7.5)
  (h2 : coupon_discount = 2.5)
  (h3 : senior_discount_rate = 0.2)
  (h4 : swap_charge = 1.0) :
  base_cost - coupon_discount - (base_cost - coupon_discount) * senior_discount_rate + swap_charge = 5 :=
by sorry


end NUMINAMATH_CALUDE_carlas_order_cost_l3526_352640


namespace NUMINAMATH_CALUDE_sale_to_cost_ratio_l3526_352671

/-- Given an article with cost price, sale price, and profit, proves that if the ratio of profit to cost price is 2, then the ratio of sale price to cost price is 3. -/
theorem sale_to_cost_ratio (cost_price sale_price profit : ℝ) 
  (h1 : profit / cost_price = 2) 
  (h2 : profit = sale_price - cost_price) :
  sale_price / cost_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_sale_to_cost_ratio_l3526_352671


namespace NUMINAMATH_CALUDE_solution_set_equality_max_value_g_l3526_352673

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := f x ≥ 1

-- Define the solution set
def solution_set : Set ℝ := {x | x ≥ 1}

-- Define the function g
def g (x : ℝ) : ℝ := f x - x^2 + x

-- Theorem 1: The solution set of f(x) ≥ 1 is {x | x ≥ 1}
theorem solution_set_equality : 
  {x : ℝ | inequality_condition x} = solution_set := by sorry

-- Theorem 2: The maximum value of g(x) is 5/4
theorem max_value_g : 
  ∃ (x : ℝ), g x = 5/4 ∧ ∀ (y : ℝ), g y ≤ 5/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_max_value_g_l3526_352673


namespace NUMINAMATH_CALUDE_quentavious_nickels_l3526_352682

/-- Represents the exchange of nickels for gum pieces -/
def exchange_nickels_for_gum (initial_nickels : ℕ) (gum_pieces : ℕ) (remaining_nickels : ℕ) : Prop :=
  initial_nickels = (gum_pieces / 2) + remaining_nickels

/-- Proves that given the conditions, Quentavious must have started with 5 nickels -/
theorem quentavious_nickels : 
  ∀ (initial_nickels : ℕ),
    exchange_nickels_for_gum initial_nickels 6 2 →
    initial_nickels = 5 := by
  sorry


end NUMINAMATH_CALUDE_quentavious_nickels_l3526_352682


namespace NUMINAMATH_CALUDE_square_side_length_l3526_352621

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 49) (h2 : side^2 = area) : side = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3526_352621


namespace NUMINAMATH_CALUDE_maia_daily_work_requests_l3526_352622

/-- The number of client requests Maia receives daily -/
def daily_requests : ℕ := 6

/-- The number of days Maia works -/
def work_days : ℕ := 5

/-- The number of client requests remaining after the work period -/
def remaining_requests : ℕ := 10

/-- The number of client requests Maia works on each day -/
def daily_work_requests : ℕ := (daily_requests * work_days - remaining_requests) / work_days

theorem maia_daily_work_requests :
  daily_work_requests = 4 := by sorry

end NUMINAMATH_CALUDE_maia_daily_work_requests_l3526_352622


namespace NUMINAMATH_CALUDE_happy_tail_dog_count_l3526_352637

theorem happy_tail_dog_count :
  let jump : ℕ := 65
  let fetch : ℕ := 40
  let bark : ℕ := 45
  let jump_fetch : ℕ := 25
  let fetch_bark : ℕ := 20
  let jump_bark : ℕ := 23
  let all_three : ℕ := 15
  let none : ℕ := 12
  
  -- Dogs that can jump and fetch, but not bark
  let jump_fetch_only : ℕ := jump_fetch - all_three
  -- Dogs that can fetch and bark, but not jump
  let fetch_bark_only : ℕ := fetch_bark - all_three
  -- Dogs that can jump and bark, but not fetch
  let jump_bark_only : ℕ := jump_bark - all_three
  
  -- Dogs that can only jump
  let jump_only : ℕ := jump - (jump_fetch_only + jump_bark_only + all_three)
  -- Dogs that can only fetch
  let fetch_only : ℕ := fetch - (jump_fetch_only + fetch_bark_only + all_three)
  -- Dogs that can only bark
  let bark_only : ℕ := bark - (jump_bark_only + fetch_bark_only + all_three)
  
  -- Total number of dogs
  jump_only + fetch_only + bark_only + jump_fetch_only + fetch_bark_only + jump_bark_only + all_three + none = 109 :=
by
  sorry

end NUMINAMATH_CALUDE_happy_tail_dog_count_l3526_352637


namespace NUMINAMATH_CALUDE_rug_overlap_problem_l3526_352664

theorem rug_overlap_problem (total_rug_area : ℝ) (covered_floor_area : ℝ) (two_layer_area : ℝ)
  (h1 : total_rug_area = 200)
  (h2 : covered_floor_area = 140)
  (h3 : two_layer_area = 24) :
  ∃ (three_layer_area : ℝ),
    three_layer_area = 18 ∧
    total_rug_area - covered_floor_area = two_layer_area + 2 * three_layer_area :=
by sorry

end NUMINAMATH_CALUDE_rug_overlap_problem_l3526_352664


namespace NUMINAMATH_CALUDE_waiter_problem_l3526_352602

/-- The number of customers who left the waiter's section -/
def customers_left : ℕ := 14

/-- The number of people at each remaining table -/
def people_per_table : ℕ := 4

/-- The number of tables in the waiter's section -/
def number_of_tables : ℕ := 2

/-- The initial number of customers in the waiter's section -/
def initial_customers : ℕ := 22

theorem waiter_problem :
  initial_customers = customers_left + (number_of_tables * people_per_table) :=
sorry

end NUMINAMATH_CALUDE_waiter_problem_l3526_352602


namespace NUMINAMATH_CALUDE_greatest_common_divisor_620_180_under_100_l3526_352663

theorem greatest_common_divisor_620_180_under_100 :
  ∃ (d : ℕ), d = Nat.gcd 620 180 ∧ d < 100 ∧ d ∣ 620 ∧ d ∣ 180 ∧
  ∀ (x : ℕ), x < 100 → x ∣ 620 → x ∣ 180 → x ≤ d :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_620_180_under_100_l3526_352663


namespace NUMINAMATH_CALUDE_combined_age_in_eight_years_l3526_352668

/-- Given the current age and the relation between your age and your brother's age 5 years ago,
    calculate the combined age of you and your brother in 8 years. -/
theorem combined_age_in_eight_years
  (your_current_age : ℕ)
  (h1 : your_current_age = 13)
  (h2 : your_current_age - 5 = (your_current_age + 3) - 5) :
  your_current_age + 8 + (your_current_age + 3) + 8 = 50 := by
  sorry

end NUMINAMATH_CALUDE_combined_age_in_eight_years_l3526_352668


namespace NUMINAMATH_CALUDE_smallest_natural_number_satisfying_conditions_l3526_352606

theorem smallest_natural_number_satisfying_conditions : 
  ∃ (n : ℕ), n = 37 ∧ 
  (∃ (k : ℕ), n + 13 = 5 * k) ∧ 
  (∃ (m : ℕ), n - 13 = 6 * m) ∧
  (∀ (x : ℕ), x < n → ¬((∃ (k : ℕ), x + 13 = 5 * k) ∧ (∃ (m : ℕ), x - 13 = 6 * m))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_natural_number_satisfying_conditions_l3526_352606


namespace NUMINAMATH_CALUDE_unique_prime_solution_l3526_352669

theorem unique_prime_solution : ∃! (p q r : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p + q^2 = r^4 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l3526_352669


namespace NUMINAMATH_CALUDE_mutual_fund_yield_range_theorem_l3526_352646

/-- Represents the range of annual yields for mutual funds -/
structure YieldRange where
  last_year : ℝ
  improvement_rate : ℝ

/-- Calculates the new range of annual yields after improvement -/
def new_range (yr : YieldRange) : ℝ :=
  yr.last_year * (1 + yr.improvement_rate)

theorem mutual_fund_yield_range_theorem (yr : YieldRange) 
  (h1 : yr.last_year = 10000)
  (h2 : yr.improvement_rate = 0.15) : 
  new_range yr = 11500 := by
  sorry

#check mutual_fund_yield_range_theorem

end NUMINAMATH_CALUDE_mutual_fund_yield_range_theorem_l3526_352646


namespace NUMINAMATH_CALUDE_calculation_proof_l3526_352601

theorem calculation_proof :
  (- (1 : ℤ)^4 + 16 / (-2 : ℤ)^3 * |-3 - 1| = -3) ∧
  (∀ a b : ℝ, -2 * (a^2 * b - 1/4 * a * b^2 + 1/2 * a^3) - (-2 * a^2 * b + 3 * a * b^2) = -5/2 * a * b^2 - a^3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3526_352601


namespace NUMINAMATH_CALUDE_job_completion_time_specific_job_completion_time_l3526_352615

/-- 
If one person completes a job in a days and another person completes the same job in b days,
then together they will complete the job in (a * b) / (a + b) days.
-/
theorem job_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let combined_time := (a * b) / (a + b)
  combined_time = (a⁻¹ + b⁻¹)⁻¹ :=
by sorry

/--
If one person completes a job in 8 days and another person completes the same job in 24 days,
then together they will complete the job in 6 days.
-/
theorem specific_job_completion_time :
  let a := 8
  let b := 24
  let combined_time := (a * b) / (a + b)
  combined_time = 6 :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_specific_job_completion_time_l3526_352615


namespace NUMINAMATH_CALUDE_purely_imaginary_quotient_implies_a_l3526_352624

def z₁ (a : ℝ) : ℂ := a + 2 * Complex.I
def z₂ : ℂ := 3 - 4 * Complex.I

theorem purely_imaginary_quotient_implies_a (a : ℝ) :
  (z₁ a / z₂).re = 0 → (z₁ a / z₂).im ≠ 0 → a = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_quotient_implies_a_l3526_352624


namespace NUMINAMATH_CALUDE_smallest_cookie_count_l3526_352681

theorem smallest_cookie_count (b : ℕ) : 
  b > 0 ∧
  b % 5 = 4 ∧
  b % 6 = 3 ∧
  b % 8 = 5 ∧
  b % 9 = 7 ∧
  (∀ c : ℕ, c > 0 ∧ c % 5 = 4 ∧ c % 6 = 3 ∧ c % 8 = 5 ∧ c % 9 = 7 → b ≤ c) →
  b = 909 := by
sorry

end NUMINAMATH_CALUDE_smallest_cookie_count_l3526_352681


namespace NUMINAMATH_CALUDE_average_ducks_is_35_l3526_352619

/-- The average number of ducks bought by three students -/
def averageDucks (adelaide ephraim kolton : ℕ) : ℚ :=
  (adelaide + ephraim + kolton : ℚ) / 3

/-- Theorem: The average number of ducks bought is 35 -/
theorem average_ducks_is_35 :
  let adelaide := 30
  let ephraim := adelaide / 2
  let kolton := ephraim + 45
  averageDucks adelaide ephraim kolton = 35 := by
sorry

end NUMINAMATH_CALUDE_average_ducks_is_35_l3526_352619


namespace NUMINAMATH_CALUDE_optimal_advertising_strategy_l3526_352675

/-- Sales revenue function -/
def R (x₁ x₂ : ℝ) : ℝ := -2 * x₁^2 - x₂^2 + 13 * x₁ + 11 * x₂ - 28

/-- Profit function -/
def profit (x₁ x₂ : ℝ) : ℝ := R x₁ x₂ - x₁ - x₂

theorem optimal_advertising_strategy :
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 5 ∧ 
    ∀ (y₁ y₂ : ℝ), y₁ + y₂ = 5 → profit x₁ x₂ ≥ profit y₁ y₂) ∧
  profit 2 3 = 9 ∧
  (∀ (y₁ y₂ : ℝ), profit 3 5 ≥ profit y₁ y₂) ∧
  profit 3 5 = 15 := by sorry

end NUMINAMATH_CALUDE_optimal_advertising_strategy_l3526_352675


namespace NUMINAMATH_CALUDE_cosine_sum_inequality_l3526_352676

theorem cosine_sum_inequality (x y z : ℝ) (h : x + y + z = 0) :
  |Real.cos x| + |Real.cos y| + |Real.cos z| ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_inequality_l3526_352676


namespace NUMINAMATH_CALUDE_stratified_sampling_second_grade_l3526_352665

theorem stratified_sampling_second_grade 
  (total_sample : ℕ) 
  (ratio_first : ℕ) 
  (ratio_second : ℕ) 
  (ratio_third : ℕ) :
  total_sample = 50 →
  ratio_first = 3 →
  ratio_second = 3 →
  ratio_third = 4 →
  (ratio_second : ℚ) / (ratio_first + ratio_second + ratio_third : ℚ) * total_sample = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_second_grade_l3526_352665


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3526_352645

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point :
  ∀ (given_line : Line),
  given_line.a = 1 ∧ given_line.b = -2 ∧ given_line.c = -2 →
  ∃ (parallel_line : Line),
    parallel parallel_line given_line ∧
    point_on_line 1 0 parallel_line ∧
    parallel_line.a = 1 ∧ parallel_line.b = -2 ∧ parallel_line.c = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3526_352645


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3526_352631

def polynomial (x : ℝ) : ℝ := 8*x^4 - 10*x^3 + 7*x^2 - 5*x - 30

def divisor (x : ℝ) : ℝ := 2*x - 4

theorem polynomial_division_remainder :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 36 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3526_352631


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l3526_352629

theorem initial_mixture_volume 
  (initial_x_percentage : Real) 
  (initial_y_percentage : Real)
  (added_x_volume : Real)
  (final_x_percentage : Real) :
  initial_x_percentage = 0.20 →
  initial_y_percentage = 0.80 →
  added_x_volume = 20 →
  final_x_percentage = 0.36 →
  ∃ (initial_volume : Real),
    initial_volume = 80 ∧
    (initial_x_percentage * initial_volume + added_x_volume) / (initial_volume + added_x_volume) = final_x_percentage :=
by sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l3526_352629


namespace NUMINAMATH_CALUDE_intersection_of_3n_and_2m_plus_1_l3526_352614

theorem intersection_of_3n_and_2m_plus_1 :
  {x : ℤ | ∃ n : ℤ, x = 3 * n} ∩ {x : ℤ | ∃ m : ℤ, x = 2 * m + 1} =
  {x : ℤ | ∃ k : ℤ, x = 12 * k + 1 ∨ x = 12 * k + 5} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_3n_and_2m_plus_1_l3526_352614


namespace NUMINAMATH_CALUDE_multiple_of_smaller_number_l3526_352658

theorem multiple_of_smaller_number (S L k : ℤ) : 
  S = 14 → 
  L = k * S - 3 → 
  S + L = 39 → 
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_smaller_number_l3526_352658


namespace NUMINAMATH_CALUDE_product_draw_probabilities_l3526_352660

/-- Represents the probability space for drawing products -/
structure ProductDraw where
  total : Nat
  defective : Nat
  nonDefective : Nat
  hTotal : total = defective + nonDefective

/-- The probability of drawing a defective product on the first draw -/
def probFirstDefective (pd : ProductDraw) : Rat :=
  pd.defective / pd.total

/-- The probability of drawing defective products on both draws -/
def probBothDefective (pd : ProductDraw) : Rat :=
  (pd.defective / pd.total) * ((pd.defective - 1) / (pd.total - 1))

/-- The probability of drawing a defective product on the second draw, given the first was defective -/
def probSecondDefectiveGivenFirst (pd : ProductDraw) : Rat :=
  (pd.defective - 1) / (pd.total - 1)

theorem product_draw_probabilities (pd : ProductDraw) 
  (h1 : pd.total = 20) 
  (h2 : pd.defective = 5) 
  (h3 : pd.nonDefective = 15) : 
  probFirstDefective pd = 1/4 ∧ 
  probBothDefective pd = 1/19 ∧ 
  probSecondDefectiveGivenFirst pd = 4/19 := by
  sorry

end NUMINAMATH_CALUDE_product_draw_probabilities_l3526_352660


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3526_352648

theorem sum_of_coefficients (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (4*x - 2)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 32 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3526_352648


namespace NUMINAMATH_CALUDE_matrix_power_identity_l3526_352623

variable {n : ℕ}

/-- Prove that for n×n complex matrices A, B, and C, if A^2 = B^2 = C^2 and B^3 = ABC + 2I, then A^6 = I. -/
theorem matrix_power_identity 
  (A B C : Matrix (Fin n) (Fin n) ℂ) 
  (h1 : A ^ 2 = B ^ 2)
  (h2 : B ^ 2 = C ^ 2)
  (h3 : B ^ 3 = A * B * C + 2 • 1) : 
  A ^ 6 = 1 := by
sorry

end NUMINAMATH_CALUDE_matrix_power_identity_l3526_352623


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_six_in_range_l3526_352677

theorem unique_square_divisible_by_six_in_range : ∃! x : ℕ, 
  (∃ n : ℕ, x = n^2) ∧ 
  (x % 6 = 0) ∧ 
  (50 ≤ x) ∧ 
  (x ≤ 150) ∧ 
  x = 144 := by
sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_six_in_range_l3526_352677


namespace NUMINAMATH_CALUDE_constant_d_value_l3526_352633

variables (a d : ℝ)

theorem constant_d_value (h : ∀ x : ℝ, (x + 3) * (x + a) = x^2 + d*x + 12) : d = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_d_value_l3526_352633


namespace NUMINAMATH_CALUDE_sum_of_ages_l3526_352634

def father_age : ℕ := 48
def son_age : ℕ := 27

theorem sum_of_ages : father_age + son_age = 75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3526_352634


namespace NUMINAMATH_CALUDE_mean_temperature_l3526_352651

def temperatures : List ℚ := [-8, -5, -5, -6, 0, 4]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : ℚ) = -10/3 := by
sorry

end NUMINAMATH_CALUDE_mean_temperature_l3526_352651


namespace NUMINAMATH_CALUDE_max_students_distribution_l3526_352698

theorem max_students_distribution (pens pencils erasers notebooks rulers : ℕ) 
  (h1 : pens = 3528) 
  (h2 : pencils = 3920) 
  (h3 : erasers = 3150) 
  (h4 : notebooks = 5880) 
  (h5 : rulers = 4410) : 
  Nat.gcd pens (Nat.gcd pencils (Nat.gcd erasers (Nat.gcd notebooks rulers))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3526_352698


namespace NUMINAMATH_CALUDE_pilot_miles_on_tuesday_l3526_352610

theorem pilot_miles_on_tuesday :
  ∀ x : ℕ,
  (3 * (x + 1475) = 7827) →
  x = 1134 := by
sorry

end NUMINAMATH_CALUDE_pilot_miles_on_tuesday_l3526_352610


namespace NUMINAMATH_CALUDE_bar_chart_best_for_rainfall_l3526_352613

-- Define the characteristics of the data
structure RainfallData where
  area : String
  seasons : Fin 4 → Float
  isRainfall : Bool

-- Define the types of charts
inductive ChartType
  | Bar
  | Line
  | Pie

-- Define a function to determine the best chart type
def bestChartType (data : RainfallData) : ChartType :=
  ChartType.Bar

-- Theorem stating that bar chart is the best choice for rainfall data
theorem bar_chart_best_for_rainfall (data : RainfallData) :
  data.isRainfall = true → bestChartType data = ChartType.Bar :=
by
  sorry

#check bar_chart_best_for_rainfall

end NUMINAMATH_CALUDE_bar_chart_best_for_rainfall_l3526_352613


namespace NUMINAMATH_CALUDE_mark_change_factor_l3526_352688

/-- Given a class of students, prove that if their marks are changed by a factor
    that doubles the average, then this factor must be 2. -/
theorem mark_change_factor
  (n : ℕ)                    -- number of students
  (initial_avg : ℝ)          -- initial average mark
  (final_avg : ℝ)            -- final average mark
  (h_n : n = 30)             -- there are 30 students
  (h_initial : initial_avg = 45)  -- initial average is 45
  (h_final : final_avg = 90)      -- final average is 90
  : (final_avg / initial_avg : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_mark_change_factor_l3526_352688


namespace NUMINAMATH_CALUDE_parabola_properties_l3526_352647

/-- Represents a parabola of the form y = ax^2 - 2ax + 3 -/
structure Parabola where
  a : ℝ
  h : a ≠ 0

/-- The axis of symmetry of the parabola -/
def axisOfSymmetry (p : Parabola) : ℝ := 1

/-- The shifted parabola's vertex is on the x-axis -/
def vertexOnXAxis (p : Parabola) : Prop :=
  p.a = 3/4 ∨ p.a = -3/2

theorem parabola_properties (p : Parabola) :
  (axisOfSymmetry p = 1) ∧
  (vertexOnXAxis p ↔ (p.a = 3/4 ∨ p.a = -3/2)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3526_352647


namespace NUMINAMATH_CALUDE_intersection_empty_implies_t_geq_one_l3526_352605

theorem intersection_empty_implies_t_geq_one (t : ℝ) : 
  let A : Set ℝ := {-1, 0, 1}
  let B : Set ℝ := {x | x > t}
  A ∩ B = ∅ → t ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_t_geq_one_l3526_352605


namespace NUMINAMATH_CALUDE_polygon_sides_l3526_352618

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 4 * 360) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3526_352618


namespace NUMINAMATH_CALUDE_faye_coloring_books_l3526_352695

/-- Calculates the total number of coloring books Faye has after giving some away and buying more. -/
def total_coloring_books (initial : ℕ) (given_away : ℕ) (bought : ℕ) : ℕ :=
  initial - given_away + bought

/-- Proves that Faye ends up with 79 coloring books given the initial conditions. -/
theorem faye_coloring_books : total_coloring_books 34 3 48 = 79 := by
  sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l3526_352695


namespace NUMINAMATH_CALUDE_equation_two_distinct_roots_l3526_352649

theorem equation_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^4 - 2*a*x^2 - x + a^2 - a = 0 ∧ 
    y^4 - 2*a*y^2 - y + a^2 - a = 0 ∧
    (∀ z : ℝ, z^4 - 2*a*z^2 - z + a^2 - a = 0 → (z = x ∨ z = y))) →
  a > -1/4 ∧ a < 3/4 :=
sorry

end NUMINAMATH_CALUDE_equation_two_distinct_roots_l3526_352649


namespace NUMINAMATH_CALUDE_quadratic_function_a_value_l3526_352678

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-value of a quadratic function at a given x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

/-- The x-coordinate of the vertex of a quadratic function -/
def QuadraticFunction.vertexX (f : QuadraticFunction) : ℚ := -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
def QuadraticFunction.vertexY (f : QuadraticFunction) : ℚ :=
  f.eval (f.vertexX)

theorem quadratic_function_a_value (f : QuadraticFunction) :
  f.vertexX = 2 ∧ f.vertexY = 5 ∧ f.eval 1 = 2 ∧ f.eval 3 = 2 → f.a = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_a_value_l3526_352678


namespace NUMINAMATH_CALUDE_unique_tournaments_eq_fib_l3526_352697

/-- Represents a sequence of scores in descending order -/
def ScoreSequence (n : ℕ) := { a : Fin n → ℕ // ∀ i j, i ≤ j → a i ≥ a j }

/-- Represents a tournament outcome -/
structure Tournament (n : ℕ) where
  scores : ScoreSequence n
  team_scores : Fin n → ℕ

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The number of unique tournament outcomes for n teams -/
def uniqueTournaments (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of unique tournament outcomes is the (n+1)th Fibonacci number -/
theorem unique_tournaments_eq_fib (n : ℕ) : uniqueTournaments n = fib (n + 1) := by sorry

end NUMINAMATH_CALUDE_unique_tournaments_eq_fib_l3526_352697
