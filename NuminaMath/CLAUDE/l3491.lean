import Mathlib

namespace NUMINAMATH_CALUDE_five_letter_words_count_l3491_349189

def word_count : ℕ := 26

theorem five_letter_words_count :
  (Finset.sum (Finset.range 4) (λ k => Nat.choose 5 k)) = word_count := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_count_l3491_349189


namespace NUMINAMATH_CALUDE_square_of_sum_leq_sum_of_squares_l3491_349113

theorem square_of_sum_leq_sum_of_squares (a b : ℝ) :
  ((a + b) / 2) ^ 2 ≤ (a^2 + b^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_leq_sum_of_squares_l3491_349113


namespace NUMINAMATH_CALUDE_spade_or_king_probability_l3491_349129

/-- The probability of drawing a spade or a king from a standard deck of cards -/
theorem spade_or_king_probability (total_cards : ℕ) (spades : ℕ) (kings : ℕ) (overlap : ℕ) :
  total_cards = 52 →
  spades = 13 →
  kings = 4 →
  overlap = 1 →
  (spades + kings - overlap : ℚ) / total_cards = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_spade_or_king_probability_l3491_349129


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l3491_349174

theorem inequality_not_always_true (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ¬ (∀ a b c, c < b ∧ b < a ∧ a * c < 0 → c * b < a * b) := by
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l3491_349174


namespace NUMINAMATH_CALUDE_triangle_problem_l3491_349105

noncomputable section

def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2 - 1

theorem triangle_problem (A B C a b c : ℝ) :
  c = Real.sqrt 3 →
  f C = 0 →
  Real.sin B = 2 * Real.sin A →
  0 < C →
  C < π →
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →
  C = π/3 ∧ a = 1 ∧ b = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_problem_l3491_349105


namespace NUMINAMATH_CALUDE_sqrt_x_minus_two_defined_l3491_349128

theorem sqrt_x_minus_two_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_two_defined_l3491_349128


namespace NUMINAMATH_CALUDE_unique_solution_system_l3491_349135

/-- The system of equations:
    3^y * 81 = 9^(x^2)
    lg y = lg x - lg 0.5
    has only one positive real solution (x, y) = (2, 4) -/
theorem unique_solution_system (x y : ℝ) 
  (h1 : (3 : ℝ)^y * 81 = 9^(x^2))
  (h2 : Real.log y / Real.log 10 = Real.log x / Real.log 10 - Real.log 0.5 / Real.log 10)
  (h3 : x > 0)
  (h4 : y > 0) : 
  x = 2 ∧ y = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3491_349135


namespace NUMINAMATH_CALUDE_inequality_proof_l3491_349163

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  (a / (b + c) + b / (a + c) + c / (a + b)) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3491_349163


namespace NUMINAMATH_CALUDE_base_8_to_16_digit_count_l3491_349120

theorem base_8_to_16_digit_count :
  ∀ n : ℕ,
  (1000 ≤ n ∧ n ≤ 7777) →  -- 4 digits in base 8
  (512 ≤ n ∧ n ≤ 4095) →   -- Equivalent range in decimal
  (0x200 ≤ n ∧ n ≤ 0xFFF)  -- 3 digits in base 16
  := by sorry

end NUMINAMATH_CALUDE_base_8_to_16_digit_count_l3491_349120


namespace NUMINAMATH_CALUDE_monday_production_tuesday_production_wednesday_production_thursday_pricing_l3491_349156

/-- Represents the recipe and conditions for making Zippies -/
structure ZippieRecipe where
  gluze_per_batch : ℚ
  blurpos_per_batch : ℚ
  zippies_per_batch : ℕ
  gluze_price : ℚ
  zippie_price : ℚ
  zippie_profit : ℚ

/-- The standard Zippie recipe -/
def standard_recipe : ZippieRecipe :=
  { gluze_per_batch := 4
  , blurpos_per_batch := 3
  , zippies_per_batch := 60
  , gluze_price := 1.8
  , zippie_price := 0.5
  , zippie_profit := 0.3 }

/-- Theorem for Monday's production -/
theorem monday_production (recipe : ZippieRecipe) (gluze_used : ℚ) :
  recipe = standard_recipe →
  gluze_used = 28 →
  gluze_used / recipe.gluze_per_batch * recipe.blurpos_per_batch = 21 :=
sorry

/-- Theorem for Tuesday's production -/
theorem tuesday_production (recipe : ZippieRecipe) (ingredient_used : ℚ) :
  recipe = standard_recipe →
  ingredient_used = 48 →
  (ingredient_used / recipe.gluze_per_batch * recipe.blurpos_per_batch = 36 ∨
   ingredient_used / recipe.blurpos_per_batch * recipe.gluze_per_batch = 64) :=
sorry

/-- Theorem for Wednesday's production -/
theorem wednesday_production (recipe : ZippieRecipe) (gluze_available blurpos_available : ℚ) :
  recipe = standard_recipe →
  gluze_available = 64 →
  blurpos_available = 42 →
  min (gluze_available / recipe.gluze_per_batch) (blurpos_available / recipe.blurpos_per_batch) * recipe.zippies_per_batch = 840 :=
sorry

/-- Theorem for Thursday's pricing -/
theorem thursday_pricing (recipe : ZippieRecipe) :
  recipe = standard_recipe →
  (recipe.zippie_price - recipe.zippie_profit) * recipe.zippies_per_batch - recipe.gluze_price * recipe.gluze_per_batch = 1.6 * recipe.blurpos_per_batch :=
sorry

end NUMINAMATH_CALUDE_monday_production_tuesday_production_wednesday_production_thursday_pricing_l3491_349156


namespace NUMINAMATH_CALUDE_word_problems_count_l3491_349130

theorem word_problems_count (total_questions : ℕ) (addition_subtraction_problems : ℕ) 
  (h1 : total_questions = 45)
  (h2 : addition_subtraction_problems = 28) :
  total_questions - addition_subtraction_problems = 17 := by
  sorry

end NUMINAMATH_CALUDE_word_problems_count_l3491_349130


namespace NUMINAMATH_CALUDE_medal_distribution_proof_l3491_349117

/-- Represents the number of runners --/
def total_runners : ℕ := 10

/-- Represents the number of British runners --/
def british_runners : ℕ := 4

/-- Represents the number of medals --/
def medals : ℕ := 3

/-- Calculates the number of ways to award medals with at least one British runner winning --/
def ways_to_award_medals : ℕ := sorry

theorem medal_distribution_proof :
  ways_to_award_medals = 492 :=
by sorry

end NUMINAMATH_CALUDE_medal_distribution_proof_l3491_349117


namespace NUMINAMATH_CALUDE_max_a_part_1_range_a_part_2_l3491_349187

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Part I
theorem max_a_part_1 : 
  (∃ a_max : ℝ, ∀ a : ℝ, (∀ x : ℝ, g x ≤ 5 → f a x ≤ 6) → a ≤ a_max) ∧
  (∀ x : ℝ, g x ≤ 5 → f 1 x ≤ 6) :=
sorry

-- Part II
theorem range_a_part_2 : 
  {a : ℝ | ∀ x : ℝ, f a x + g x ≥ 3} = {a : ℝ | a ≥ 2} :=
sorry

end NUMINAMATH_CALUDE_max_a_part_1_range_a_part_2_l3491_349187


namespace NUMINAMATH_CALUDE_businessmen_drink_neither_l3491_349159

theorem businessmen_drink_neither (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ)
  (h_total : total = 30)
  (h_coffee : coffee = 15)
  (h_tea : tea = 13)
  (h_both : both = 7) :
  total - ((coffee + tea) - both) = 9 :=
by sorry

end NUMINAMATH_CALUDE_businessmen_drink_neither_l3491_349159


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3491_349192

/-- The decimal representation of 0.6̄03 as a rational number -/
def repeating_decimal : ℚ := 0.6 + (3 : ℚ) / 100 / (1 - 1/100)

/-- Theorem stating that 0.6̄03 is equal to 104/165 -/
theorem repeating_decimal_as_fraction : repeating_decimal = 104 / 165 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3491_349192


namespace NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l3491_349176

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem fiftieth_term_of_sequence : arithmetic_sequence 2 4 50 = 198 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l3491_349176


namespace NUMINAMATH_CALUDE_fib_divisibility_l3491_349151

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- State the theorem
theorem fib_divisibility (m n : ℕ) (h : m > 0) (h' : n > 0) : 
  m ∣ n → (fib (m - 1)) ∣ (fib (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_fib_divisibility_l3491_349151


namespace NUMINAMATH_CALUDE_equation_rewrite_product_l3491_349103

theorem equation_rewrite_product (a b x y : ℝ) (m' n' p' : ℤ) :
  (a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 2)) →
  ((a^m'*x - a^n') * (a^p'*y - a^3) = a^5*b^5) →
  m' * n' * p' = 48 := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_product_l3491_349103


namespace NUMINAMATH_CALUDE_absolute_value_of_specific_integers_l3491_349116

theorem absolute_value_of_specific_integers :
  ∃ (a b c : ℤ),
    (∀ x : ℤ, x < 0 → x ≤ a) ∧
    (∀ x : ℤ, |x| ≥ |b|) ∧
    (∀ x : ℤ, x > 0 → c ≤ x) ∧
    |a + b - c| = 2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_of_specific_integers_l3491_349116


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3491_349146

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z := (10 * i) / (3 + i)
  Complex.im z = 3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3491_349146


namespace NUMINAMATH_CALUDE_complicated_expression_equality_l3491_349102

theorem complicated_expression_equality : 
  Real.sqrt (11 * 13) * (1/3) + 2 * (Real.sqrt 17 / 3) - 4 * (Real.sqrt 7 / 5) = 
  (5 * Real.sqrt 143 + 10 * Real.sqrt 17 - 12 * Real.sqrt 7) / 15 := by
sorry

end NUMINAMATH_CALUDE_complicated_expression_equality_l3491_349102


namespace NUMINAMATH_CALUDE_total_whales_is_178_l3491_349139

/-- Represents the number of whales observed during Ishmael's monitoring trips -/
def total_whales (first_trip_male : ℕ) : ℕ :=
  let first_trip_female := 2 * first_trip_male
  let first_trip_total := first_trip_male + first_trip_female
  let second_trip_baby := 8
  let second_trip_parents := 2 * second_trip_baby
  let second_trip_total := second_trip_baby + second_trip_parents
  let third_trip_male := first_trip_male / 2
  let third_trip_female := first_trip_female
  let third_trip_total := third_trip_male + third_trip_female
  first_trip_total + second_trip_total + third_trip_total

/-- Theorem stating that the total number of whales observed is 178 -/
theorem total_whales_is_178 : total_whales 28 = 178 := by
  sorry

end NUMINAMATH_CALUDE_total_whales_is_178_l3491_349139


namespace NUMINAMATH_CALUDE_four_heads_in_five_tosses_l3491_349191

/-- The probability of getting exactly k successes in n trials with probability p of success in each trial. -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem: The probability of getting exactly 4 heads in 5 tosses of a fair coin is 0.15625. -/
theorem four_heads_in_five_tosses :
  binomialProbability 5 4 (1/2) = 0.15625 := by
sorry

end NUMINAMATH_CALUDE_four_heads_in_five_tosses_l3491_349191


namespace NUMINAMATH_CALUDE_integer_points_in_triangle_DEF_l3491_349152

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  leg1 : ℕ
  leg2 : ℕ

/-- Counts the number of integer coordinate points in and on a right triangle -/
def count_integer_points (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle with legs 15 and 20 -/
def triangle_DEF : RightTriangle :=
  { leg1 := 15, leg2 := 20 }

/-- Theorem stating that the number of integer coordinate points in triangle_DEF is 181 -/
theorem integer_points_in_triangle_DEF : 
  count_integer_points triangle_DEF = 181 := by
  sorry

end NUMINAMATH_CALUDE_integer_points_in_triangle_DEF_l3491_349152


namespace NUMINAMATH_CALUDE_journey_distance_l3491_349162

theorem journey_distance (total_distance : ℝ) (bike_speed walking_speed : ℝ) 
  (h1 : bike_speed = 12)
  (h2 : walking_speed = 4)
  (h3 : (3/4 * total_distance) / bike_speed + (1/4 * total_distance) / walking_speed = 1) :
  1/4 * total_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l3491_349162


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3491_349172

-- Define an isosceles triangle with side lengths 3 and 7
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 3 ∧ b = 7 ∧ c = 7) ∨ (a = 7 ∧ b = 3 ∧ c = 7) ∨ (a = 7 ∧ b = 7 ∧ c = 3)

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → Perimeter a b c = 17 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3491_349172


namespace NUMINAMATH_CALUDE_sum_of_first_5n_integers_l3491_349101

theorem sum_of_first_5n_integers (n : ℕ) : 
  (4*n*(4*n+1))/2 = (n*(n+1))/2 + 210 → (5*n*(5*n+1))/2 = 465 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_5n_integers_l3491_349101


namespace NUMINAMATH_CALUDE_max_value_of_largest_integer_l3491_349123

theorem max_value_of_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℝ) / 5 = 45 →
  (max a (max b (max c (max d e)))) - (min a (min b (min c (min d e)))) = 10 →
  max a (max b (max c (max d e))) ≤ 215 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_largest_integer_l3491_349123


namespace NUMINAMATH_CALUDE_third_shot_probability_at_least_one_hit_probability_l3491_349122

/-- A marksman shoots four times independently with a probability of hitting the target of 0.9 each time. -/
structure Marksman where
  shots : Fin 4 → ℝ
  prob_hit : ∀ i, shots i = 0.9
  independent : ∀ i j, i ≠ j → shots i = shots j

/-- The probability of hitting the target on the third shot is 0.9. -/
theorem third_shot_probability (m : Marksman) : m.shots 2 = 0.9 := by sorry

/-- The probability of hitting the target at least once is 1 - 0.1^4. -/
theorem at_least_one_hit_probability (m : Marksman) : 
  1 - (1 - m.shots 0) * (1 - m.shots 1) * (1 - m.shots 2) * (1 - m.shots 3) = 1 - 0.1^4 := by sorry

end NUMINAMATH_CALUDE_third_shot_probability_at_least_one_hit_probability_l3491_349122


namespace NUMINAMATH_CALUDE_original_price_calculation_l3491_349145

/-- Given a 6% rebate followed by a 10% sales tax, if the final price is Rs. 6876.1,
    then the original price was Rs. 6650. -/
theorem original_price_calculation (original_price : ℝ) : 
  (original_price * (1 - 0.06) * (1 + 0.10) = 6876.1) → 
  (original_price = 6650) := by sorry

end NUMINAMATH_CALUDE_original_price_calculation_l3491_349145


namespace NUMINAMATH_CALUDE_negation_equivalence_l3491_349183

theorem negation_equivalence (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3491_349183


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l3491_349124

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mwf : ℕ  -- Hours worked on Monday, Wednesday, Friday
  hours_tth : ℕ  -- Hours worked on Tuesday, Thursday
  days_mwf : ℕ   -- Number of days worked with hours_mwf
  days_tth : ℕ   -- Number of days worked with hours_tth
  weekly_earnings : ℕ

/-- Calculate Sheila's hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := schedule.hours_mwf * schedule.days_mwf + schedule.hours_tth * schedule.days_tth
  schedule.weekly_earnings / total_hours

/-- Theorem: Sheila's hourly wage is $11 --/
theorem sheila_hourly_wage :
  let schedule : WorkSchedule := {
    hours_mwf := 8,
    hours_tth := 6,
    days_mwf := 3,
    days_tth := 2,
    weekly_earnings := 396
  }
  hourly_wage schedule = 11 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l3491_349124


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3491_349178

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 8*x + c

-- State the theorem
theorem quadratic_inequality_solution (c : ℝ) :
  (c > 0) → (∃ x, f c x < 0) ↔ (c > 0 ∧ c < 16) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3491_349178


namespace NUMINAMATH_CALUDE_test_question_count_l3491_349185

/-- Given a test with two-point and four-point questions, prove the total number of questions. -/
theorem test_question_count (two_point_count four_point_count : ℕ) 
  (h1 : two_point_count = 30)
  (h2 : four_point_count = 10) :
  two_point_count + four_point_count = 40 := by
  sorry

#check test_question_count

end NUMINAMATH_CALUDE_test_question_count_l3491_349185


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_value_prove_mean_equality_implies_y_value_l3491_349119

theorem mean_equality_implies_y_value : ℝ → Prop :=
  fun y =>
    (6 + 9 + 18) / 3 = (12 + y) / 2 →
    y = 10

-- The proof is omitted
theorem prove_mean_equality_implies_y_value :
  ∃ y : ℝ, mean_equality_implies_y_value y :=
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_value_prove_mean_equality_implies_y_value_l3491_349119


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l3491_349118

/-- The area of a square with perimeter 24 is 36 -/
theorem square_area_from_perimeter : 
  ∀ s : Real, 
  (s > 0) → 
  (4 * s = 24) → 
  (s * s = 36) :=
by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l3491_349118


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3491_349167

noncomputable def f (x : ℝ) : ℝ := -2/3 * x^3 + 3/2 * x^2 - x

theorem f_increasing_on_interval :
  ∀ x ∈ Set.Icc (1/2 : ℝ) 1, 
    (∀ y ∈ Set.Icc (1/2 : ℝ) 1, x ≤ y → f x ≤ f y) :=
by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3491_349167


namespace NUMINAMATH_CALUDE_parallelepiped_sphere_properties_l3491_349142

/-- Represents a parallelepiped ABCDA₁B₁C₁D₁ with a sphere Ω touching its edges -/
structure Parallelepiped where
  -- Edge length of A₁A
  edge_length : ℝ
  -- Volume of the parallelepiped
  volume : ℝ
  -- Radius of the sphere Ω
  sphere_radius : ℝ
  -- A₁A is perpendicular to ABCD
  edge_perpendicular : edge_length > 0
  -- Sphere Ω touches BB₁, B₁C₁, C₁C, CB, C₁D₁, and AD
  sphere_touches_edges : True
  -- Ω touches C₁D₁ at K where C₁K = 9 and KD₁ = 4
  sphere_touch_point : edge_length > 13

/-- The theorem stating the properties of the parallelepiped and sphere -/
theorem parallelepiped_sphere_properties : 
  ∃ (p : Parallelepiped), 
    p.edge_length = 18 ∧ 
    p.volume = 3888 ∧ 
    p.sphere_radius = 3 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_sphere_properties_l3491_349142


namespace NUMINAMATH_CALUDE_basil_leaves_count_l3491_349168

theorem basil_leaves_count (basil_pots rosemary_pots thyme_pots : ℕ)
  (rosemary_leaves_per_pot thyme_leaves_per_pot : ℕ)
  (total_leaves : ℕ) :
  basil_pots = 3 →
  rosemary_pots = 9 →
  thyme_pots = 6 →
  rosemary_leaves_per_pot = 18 →
  thyme_leaves_per_pot = 30 →
  total_leaves = 354 →
  ∃ (basil_leaves_per_pot : ℕ),
    basil_leaves_per_pot * basil_pots +
    rosemary_leaves_per_pot * rosemary_pots +
    thyme_leaves_per_pot * thyme_pots = total_leaves ∧
    basil_leaves_per_pot = 4 :=
by sorry

end NUMINAMATH_CALUDE_basil_leaves_count_l3491_349168


namespace NUMINAMATH_CALUDE_exists_valid_chain_l3491_349132

/-- A chain between two integers is a finite sequence of positive integers
    where the product of any two consecutive elements is divisible by their sum. -/
def IsValidChain (chain : List Nat) : Prop :=
  chain.length ≥ 2 ∧
  ∀ i, i + 1 < chain.length →
    (chain[i]! * chain[i+1]!) % (chain[i]! + chain[i+1]!) = 0

/-- For any two integers greater than 2, there exists a valid chain between them. -/
theorem exists_valid_chain (m n : Nat) (hm : m > 2) (hn : n > 2) :
  ∃ (chain : List Nat), chain.head! = m ∧ chain.getLast! = n ∧ IsValidChain chain :=
sorry

end NUMINAMATH_CALUDE_exists_valid_chain_l3491_349132


namespace NUMINAMATH_CALUDE_gcd_lcm_product_30_45_l3491_349143

theorem gcd_lcm_product_30_45 : Nat.gcd 30 45 * Nat.lcm 30 45 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_30_45_l3491_349143


namespace NUMINAMATH_CALUDE_carls_gift_bags_l3491_349181

/-- Represents the gift bag distribution problem at Carl's open house. -/
theorem carls_gift_bags (total_visitors : ℕ) (extravagant_bags : ℕ) (additional_bags : ℕ) :
  total_visitors = 90 →
  extravagant_bags = 10 →
  additional_bags = 60 →
  total_visitors - (extravagant_bags + additional_bags) = 30 := by
  sorry

#check carls_gift_bags

end NUMINAMATH_CALUDE_carls_gift_bags_l3491_349181


namespace NUMINAMATH_CALUDE_tyler_cake_eggs_l3491_349125

/-- Represents the number of eggs needed for a cake --/
def eggs_for_cake (people : ℕ) : ℕ := 2 * (people / 4)

/-- Represents the number of additional eggs needed --/
def additional_eggs_needed (recipe_eggs : ℕ) (available_eggs : ℕ) : ℕ :=
  max (recipe_eggs - available_eggs) 0

theorem tyler_cake_eggs : 
  additional_eggs_needed (eggs_for_cake 8) 3 = 1 := by sorry

end NUMINAMATH_CALUDE_tyler_cake_eggs_l3491_349125


namespace NUMINAMATH_CALUDE_m_range_proof_l3491_349140

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0

def q (m : ℝ) : Prop := m ∈ Set.Icc (-1 : ℝ) 5

-- Define the range of m
def m_range : Set ℝ := Set.Ioi (-1 : ℝ) ∪ Set.Ioc 1 5

-- Theorem statement
theorem m_range_proof :
  (∀ m : ℝ, (p m ∧ q m → False) ∧ (p m ∨ q m)) →
  (∀ m : ℝ, m ∈ m_range ↔ (p m ∨ q m) ∧ ¬(p m ∧ q m)) :=
by sorry

end NUMINAMATH_CALUDE_m_range_proof_l3491_349140


namespace NUMINAMATH_CALUDE_ticket_problem_l3491_349188

theorem ticket_problem (T : ℚ) : 
  (1/2 : ℚ) * T + (1/4 : ℚ) * ((1/2 : ℚ) * T) = 3600 → T = 5760 := by
  sorry

#check ticket_problem

end NUMINAMATH_CALUDE_ticket_problem_l3491_349188


namespace NUMINAMATH_CALUDE_min_value_theorem_l3491_349112

theorem min_value_theorem (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 10) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (w : ℝ), w = x^2 + y^2 + z^2 + x^2*y → w ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3491_349112


namespace NUMINAMATH_CALUDE_similar_triangles_l3491_349173

/-- Given five complex numbers representing points in a plane, if three triangles formed by these points are directly similar, then a fourth triangle is also directly similar to them. -/
theorem similar_triangles (a b c u v : ℂ) 
  (h : (v - a) / (u - a) = (u - v) / (b - v) ∧ (u - v) / (b - v) = (c - u) / (v - u)) : 
  (v - a) / (u - a) = (c - a) / (b - a) := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_l3491_349173


namespace NUMINAMATH_CALUDE_inequality_xyz_l3491_349175

theorem inequality_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 + y^4 + z^2 ≥ x*y*z*Real.sqrt 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_xyz_l3491_349175


namespace NUMINAMATH_CALUDE_exam_score_problem_l3491_349198

/-- Given an exam with 150 questions, where correct answers score 5 marks,
    wrong answers lose 2 marks, and the total score is 370,
    prove that the number of correctly answered questions is 95. -/
theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ)
  (h_total : total_questions = 150)
  (h_correct : correct_score = 5)
  (h_wrong : wrong_score = -2)
  (h_score : total_score = 370) :
  ∃ (correct_answers : ℕ),
    correct_answers = 95 ∧
    correct_answers ≤ total_questions ∧
    (correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score) :=
by sorry


end NUMINAMATH_CALUDE_exam_score_problem_l3491_349198


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3491_349164

theorem sphere_surface_area (r : ℝ) (h : r = 4) : 4 * Real.pi * r^2 = 64 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3491_349164


namespace NUMINAMATH_CALUDE_waiter_customers_l3491_349148

/-- The total number of customers a waiter has after new arrivals -/
def total_customers (initial : ℕ) (new_arrivals : ℕ) : ℕ :=
  initial + new_arrivals

/-- Theorem stating that with 3 initial customers and 5 new arrivals, the total is 8 -/
theorem waiter_customers : total_customers 3 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l3491_349148


namespace NUMINAMATH_CALUDE_line_through_points_with_xintercept_l3491_349160

/-- A line in a 2D plane -/
structure Line where
  slope : ℚ
  yIntercept : ℚ

/-- Create a line from two points -/
def Line.fromPoints (x1 y1 x2 y2 : ℚ) : Line :=
  let slope := (y2 - y1) / (x2 - x1)
  let yIntercept := y1 - slope * x1
  { slope := slope, yIntercept := yIntercept }

/-- Get the x-coordinate for a given y-coordinate on a line -/
def Line.xCoordinate (l : Line) (y : ℚ) : ℚ :=
  (y - l.yIntercept) / l.slope

theorem line_through_points_with_xintercept
  (line : Line)
  (h1 : line = Line.fromPoints 4 0 10 3)
  (h2 : line.xCoordinate (-6) = -8) : True :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_with_xintercept_l3491_349160


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3491_349194

def f (a x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3491_349194


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3491_349136

/-- The quadratic equation (m-1)x^2 - 2x + 1 = 0 has real roots if and only if m ≤ 2 and m ≠ 1 -/
theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 2 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3491_349136


namespace NUMINAMATH_CALUDE_expression_value_l3491_349150

theorem expression_value :
  let x : ℤ := 3
  let y : ℤ := 2
  let z : ℤ := 4
  let w : ℤ := -1
  x^2 * y - 2 * x * y + 3 * x * z - (x + y) * (y + z) * (z + w) = -48 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3491_349150


namespace NUMINAMATH_CALUDE_smallest_period_is_40_l3491_349161

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The smallest positive period of functions satisfying the condition -/
theorem smallest_period_is_40 :
  ∀ f : ℝ → ℝ, SatisfiesCondition f →
    (∃ p : ℝ, p > 0 ∧ IsPeriod f p ∧
      ∀ q : ℝ, q > 0 → IsPeriod f q → p ≤ q) →
    (∃ p : ℝ, p > 0 ∧ IsPeriod f p ∧
      ∀ q : ℝ, q > 0 → IsPeriod f q → p ≤ q) ∧ p = 40 :=
by sorry

end NUMINAMATH_CALUDE_smallest_period_is_40_l3491_349161


namespace NUMINAMATH_CALUDE_mixed_fraction_division_l3491_349158

theorem mixed_fraction_division :
  (7 + 1/3) / (2 + 1/2) = 44/15 := by sorry

end NUMINAMATH_CALUDE_mixed_fraction_division_l3491_349158


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l3491_349149

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l3491_349149


namespace NUMINAMATH_CALUDE_grade_students_count_l3491_349170

theorem grade_students_count : ∃ n : ℕ, 
  400 < n ∧ n < 500 ∧
  n % 3 = 2 ∧
  n % 5 = 3 ∧
  n % 7 = 2 ∧
  n = 443 :=
by sorry

end NUMINAMATH_CALUDE_grade_students_count_l3491_349170


namespace NUMINAMATH_CALUDE_rotation_composition_is_translation_l3491_349155

-- Define a plane figure
def PlaneFigure : Type := sorry

-- Define a point in the plane
def Point : Type := sorry

-- Define a rotation transformation
def rotate (center : Point) (angle : ℝ) (figure : PlaneFigure) : PlaneFigure := sorry

-- Define a translation transformation
def translate (displacement : Point) (figure : PlaneFigure) : PlaneFigure := sorry

-- Define composition of transformations
def compose (t1 t2 : PlaneFigure → PlaneFigure) : PlaneFigure → PlaneFigure := sorry

theorem rotation_composition_is_translation 
  (F : PlaneFigure) (O O₁ : Point) (α : ℝ) :
  ∃ d : Point, compose (rotate O α) (rotate O₁ (-α)) F = translate d F :=
sorry

end NUMINAMATH_CALUDE_rotation_composition_is_translation_l3491_349155


namespace NUMINAMATH_CALUDE_total_albums_l3491_349109

/-- The number of albums each person has -/
structure Albums where
  adele : ℕ
  bridget : ℕ
  katrina : ℕ
  miriam : ℕ

/-- The conditions of the problem -/
def album_conditions (a : Albums) : Prop :=
  a.adele = 30 ∧
  a.bridget = a.adele - 15 ∧
  a.katrina = 6 * a.bridget ∧
  a.miriam = 5 * a.katrina

/-- The theorem to be proved -/
theorem total_albums (a : Albums) (h : album_conditions a) : 
  a.adele + a.bridget + a.katrina + a.miriam = 585 := by
  sorry


end NUMINAMATH_CALUDE_total_albums_l3491_349109


namespace NUMINAMATH_CALUDE_school_students_count_l3491_349127

theorem school_students_count :
  ∀ (total_students boys girls : ℕ),
  total_students = boys + girls →
  boys = 80 →
  girls = (80 * total_students) / 100 →
  total_students = 400 := by
sorry

end NUMINAMATH_CALUDE_school_students_count_l3491_349127


namespace NUMINAMATH_CALUDE_hike_vans_count_l3491_349186

/-- Calculates the number of vans required for a hike --/
def calculate_vans (total_people : ℕ) (cars : ℕ) (taxis : ℕ) 
  (people_per_car : ℕ) (people_per_taxi : ℕ) (people_per_van : ℕ) : ℕ :=
  let people_in_cars_and_taxis := cars * people_per_car + taxis * people_per_taxi
  let people_in_vans := total_people - people_in_cars_and_taxis
  (people_in_vans + people_per_van - 1) / people_per_van

theorem hike_vans_count : 
  calculate_vans 58 3 6 4 6 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hike_vans_count_l3491_349186


namespace NUMINAMATH_CALUDE_martha_total_time_l3491_349134

def router_time : ℕ := 10

def on_hold_time : ℕ := 6 * router_time

def yelling_time : ℕ := on_hold_time / 2

def total_time : ℕ := router_time + on_hold_time + yelling_time

theorem martha_total_time : total_time = 100 := by
  sorry

end NUMINAMATH_CALUDE_martha_total_time_l3491_349134


namespace NUMINAMATH_CALUDE_cloth_cost_price_l3491_349147

theorem cloth_cost_price
  (total_length : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 60)
  (h2 : selling_price = 8400)
  (h3 : profit_per_meter = 12) :
  (selling_price - total_length * profit_per_meter) / total_length = 128 :=
by sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l3491_349147


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3491_349177

def i : ℂ := Complex.I

theorem complex_number_in_first_quadrant :
  let z : ℂ := i * (1 - i) * i
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3491_349177


namespace NUMINAMATH_CALUDE_rals_age_l3491_349179

theorem rals_age (suri_age suri_age_in_3_years ral_age : ℕ) :
  suri_age_in_3_years = suri_age + 3 →
  suri_age_in_3_years = 16 →
  ral_age = 2 * suri_age →
  ral_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_rals_age_l3491_349179


namespace NUMINAMATH_CALUDE_item_price_ratio_l3491_349193

theorem item_price_ratio (x y c : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_item_price_ratio_l3491_349193


namespace NUMINAMATH_CALUDE_no_triangle_condition_l3491_349165

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) : Prop := 4 * x + 3 * y + 5 = 0
def line3 (m x y : ℝ) : Prop := m * x - y - 1 = 0

-- Define when three lines form a triangle
def form_triangle (l1 l2 l3 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    l1 x1 y1 ∧ l2 x2 y2 ∧ l3 x3 y3 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧ (x2 ≠ x3 ∨ y2 ≠ y3) ∧ (x3 ≠ x1 ∨ y3 ≠ y1)

-- Theorem statement
theorem no_triangle_condition (m : ℝ) :
  ¬(form_triangle line1 line2 (line3 m)) ↔ m ∈ ({-4/3, 2/3, 4/3} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_no_triangle_condition_l3491_349165


namespace NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l3491_349180

/-- Triangle with positive integer side lengths --/
structure Triangle :=
  (side1 : ℕ+) (side2 : ℕ+) (side3 : ℕ+)

/-- Isosceles triangle with two equal sides --/
def IsoscelesTriangle (t : Triangle) : Prop :=
  t.side1 = t.side2

/-- Point J is the intersection of angle bisectors of ∠Q and ∠R --/
def HasIntersectionJ (t : Triangle) : Prop :=
  ∃ j : ℝ × ℝ, true  -- We don't need to specify the exact conditions for J

/-- Length of QJ is 10 --/
def QJLength (t : Triangle) : Prop :=
  ∃ qj : ℝ, qj = 10

/-- Perimeter of a triangle --/
def Perimeter (t : Triangle) : ℕ :=
  t.side1.val + t.side2.val + t.side3.val

/-- The main theorem --/
theorem smallest_perimeter_isosceles_triangle :
  ∀ t : Triangle,
    IsoscelesTriangle t →
    HasIntersectionJ t →
    QJLength t →
    (∀ t' : Triangle,
      IsoscelesTriangle t' →
      HasIntersectionJ t' →
      QJLength t' →
      Perimeter t ≤ Perimeter t') →
    Perimeter t = 120 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l3491_349180


namespace NUMINAMATH_CALUDE_sculpture_cost_theorem_l3491_349157

/-- Calculates the total cost of John's custom sculpture --/
def calculate_sculpture_cost (base_price : ℝ) (standard_discount : ℝ) (marble_increase : ℝ) 
  (glass_increase : ℝ) (shipping_cost : ℝ) (tax_rate : ℝ) : ℝ :=
  let discounted_price := base_price * (1 - standard_discount)
  let marble_price := discounted_price * (1 + marble_increase)
  let glass_price := marble_price * (1 + glass_increase)
  let pre_tax_price := glass_price
  let tax := pre_tax_price * tax_rate
  pre_tax_price + tax + shipping_cost

/-- The total cost of John's sculpture is $1058.18 --/
theorem sculpture_cost_theorem : 
  calculate_sculpture_cost 450 0.15 0.70 0.35 75 0.12 = 1058.18 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_theorem_l3491_349157


namespace NUMINAMATH_CALUDE_percentage_not_sold_is_25_percent_l3491_349110

-- Define the initial stock and daily sales
def initial_stock : ℕ := 600
def monday_sales : ℕ := 25
def tuesday_sales : ℕ := 70
def wednesday_sales : ℕ := 100
def thursday_sales : ℕ := 110
def friday_sales : ℕ := 145

-- Define the total sales
def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

-- Define the number of bags not sold
def bags_not_sold : ℕ := initial_stock - total_sales

-- Theorem to prove
theorem percentage_not_sold_is_25_percent :
  (bags_not_sold : ℚ) / (initial_stock : ℚ) * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_percentage_not_sold_is_25_percent_l3491_349110


namespace NUMINAMATH_CALUDE_bread_cost_l3491_349115

/-- The cost of the loaf of bread given the conditions of Ted's sandwich-making scenario --/
theorem bread_cost (sandwich_meat_cost meat_packs cheese_cost cheese_packs : ℕ → ℚ)
  (meat_coupon cheese_coupon : ℚ) (sandwich_price : ℚ) (sandwich_count : ℕ) :
  let total_meat_cost := meat_packs 2 * sandwich_meat_cost 1 - meat_coupon
  let total_cheese_cost := cheese_packs 2 * cheese_cost 1 - cheese_coupon
  let total_ingredient_cost := total_meat_cost + total_cheese_cost
  let total_revenue := sandwich_count * sandwich_price
  total_revenue - total_ingredient_cost = 4 :=
by sorry

#check bread_cost (λ _ => 5) (λ _ => 2) (λ _ => 4) (λ _ => 2) 1 1 2 10

end NUMINAMATH_CALUDE_bread_cost_l3491_349115


namespace NUMINAMATH_CALUDE_set_intersection_example_l3491_349133

theorem set_intersection_example : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 4}
  A ∩ B = {2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l3491_349133


namespace NUMINAMATH_CALUDE_homework_problem_distribution_l3491_349197

theorem homework_problem_distribution (total : ℕ) (true_false : ℕ) : 
  total = 45 → true_false = 6 → ∃ (multiple_choice free_response : ℕ),
    multiple_choice = 2 * free_response ∧
    free_response > true_false ∧
    multiple_choice + free_response + true_false = total ∧
    free_response - true_false = 7 :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_distribution_l3491_349197


namespace NUMINAMATH_CALUDE_sqrt_3_binary_representation_l3491_349141

open Real

theorem sqrt_3_binary_representation (n : ℕ+) :
  ¬ (2^(n.val + 1) ∣ ⌊2^(2 * n.val) * Real.sqrt 3⌋) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_binary_representation_l3491_349141


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l3491_349171

-- Define the perimeter of square A
def perimeterA : ℝ := 36

-- Define the relationship between areas of square A and B
def areaRelation (areaA areaB : ℝ) : Prop := areaB = areaA / 3

-- State the theorem
theorem square_perimeter_relation (sideA sideB : ℝ) 
  (h1 : sideA * 4 = perimeterA)
  (h2 : areaRelation (sideA * sideA) (sideB * sideB)) :
  4 * sideB = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l3491_349171


namespace NUMINAMATH_CALUDE_gcf_420_144_l3491_349131

theorem gcf_420_144 : Nat.gcd 420 144 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_420_144_l3491_349131


namespace NUMINAMATH_CALUDE_least_possible_average_speed_l3491_349154

/-- Represents a palindromic number -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The drive duration in hours -/
def driveDuration : ℕ := 5

/-- The speed limit in miles per hour -/
def speedLimit : ℕ := 65

/-- The initial odometer reading -/
def initialReading : ℕ := 123321

/-- Theorem: The least possible average speed is 20 miles per hour -/
theorem least_possible_average_speed :
  ∃ (finalReading : ℕ),
    isPalindrome initialReading ∧
    isPalindrome finalReading ∧
    finalReading > initialReading ∧
    finalReading - initialReading ≤ driveDuration * speedLimit ∧
    (finalReading - initialReading) / driveDuration = 20 ∧
    ∀ (otherFinalReading : ℕ),
      isPalindrome otherFinalReading →
      otherFinalReading > initialReading →
      otherFinalReading - initialReading ≤ driveDuration * speedLimit →
      (otherFinalReading - initialReading) / driveDuration ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_least_possible_average_speed_l3491_349154


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3491_349121

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x < 1 → x^2 - 4*x + 3 > 0) ∧ 
  (∃ x : ℝ, x^2 - 4*x + 3 > 0 ∧ x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3491_349121


namespace NUMINAMATH_CALUDE_recurrence_sequence_property_l3491_349184

/-- A sequence of integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℤ) : Prop :=
  (∀ n, a n ≠ -1) ∧
  (∀ n, a (n + 2) = (a n + 2006) / (a (n + 1) + 1))

/-- The theorem stating the properties of the recurrence sequence -/
theorem recurrence_sequence_property (a : ℕ → ℤ) (h : RecurrenceSequence a) :
  ∃ x y : ℤ, x * y = 2006 ∧ (∀ n, a n = x ∨ a n = y) ∧ (∀ n, a n = a (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_property_l3491_349184


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_true_implication_l3491_349107

theorem sufficient_condition_implies_true_implication (p q : Prop) :
  (p → q) → (p → q) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_true_implication_l3491_349107


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3491_349108

theorem trigonometric_identity : 
  (Real.cos (10 * π / 180)) / (Real.tan (20 * π / 180)) + 
  Real.sqrt 3 * (Real.sin (10 * π / 180)) * (Real.tan (70 * π / 180)) - 
  2 * (Real.cos (40 * π / 180)) = 2 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3491_349108


namespace NUMINAMATH_CALUDE_salt_solution_percentage_l3491_349126

theorem salt_solution_percentage (S : ℝ) : 
  S ≥ 0 ∧ S ≤ 100 →  -- Ensure S is a valid percentage
  (3/4 * S + 1/4 * 28 = 16) →  -- Equation representing the mixing of solutions
  S = 12 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_percentage_l3491_349126


namespace NUMINAMATH_CALUDE_consecutive_sum_transformation_l3491_349104

theorem consecutive_sum_transformation (S : ℤ) : 
  ∃ (a : ℤ), 
    (a + (a + 1) = S) → 
    (3 * (a + 5) + 3 * ((a + 1) + 5) = 3 * S + 30) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_transformation_l3491_349104


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l3491_349166

/-- Given a two-digit number n, returns the number obtained by switching its digits -/
def switch_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Checks if a number satisfies the condition: switching its digits and multiplying by 3 results in 3n -/
def satisfies_condition (n : ℕ) : Prop :=
  3 * switch_digits n = 3 * n

theorem smallest_satisfying_number :
  ∃ (n : ℕ),
    10 ≤ n ∧ n < 100 ∧
    satisfies_condition n ∧
    (∀ m : ℕ, 10 ≤ m ∧ m < n → ¬satisfies_condition m) ∧
    n = 11 := by
  sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l3491_349166


namespace NUMINAMATH_CALUDE_line_equation_equivalence_slope_intercept_parameters_l3491_349138

/-- Given a line equation in vector form, prove its slope-intercept form and parameters -/
theorem line_equation_equivalence :
  ∀ (x y : ℝ),
  (2 : ℝ) * (x - 4) + (-1 : ℝ) * (y + 3) = 0 ↔ y = 2 * x - 11 :=
by sorry

/-- Prove the slope and y-intercept of the line -/
theorem slope_intercept_parameters :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (2 : ℝ) * (x - 4) + (-1 : ℝ) * (y + 3) = 0 ↔ y = m * x + b) ∧ m = 2 ∧ b = -11 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_slope_intercept_parameters_l3491_349138


namespace NUMINAMATH_CALUDE_z_value_l3491_349199

theorem z_value (x y z : ℚ) 
  (eq1 : 3 * x^2 + 2 * x * y * z - y^3 + 11 = z)
  (eq2 : x = 2)
  (eq3 : y = 3) : 
  z = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_z_value_l3491_349199


namespace NUMINAMATH_CALUDE_beam_width_calculation_beam_width_250_pounds_l3491_349137

/-- The maximum load a beam can support is directly proportional to its width -/
def load_proportional_to_width (load width : ℝ) : Prop :=
  ∃ k : ℝ, load = k * width

/-- Theorem: Given the proportionality between load and width, and a reference beam,
    calculate the width of a beam supporting a specific load -/
theorem beam_width_calculation
  (reference_width reference_load target_load : ℝ)
  (h_positive : reference_width > 0 ∧ reference_load > 0 ∧ target_load > 0)
  (h_prop : load_proportional_to_width reference_load reference_width)
  (h_prop_target : load_proportional_to_width target_load (target_load * reference_width / reference_load)) :
  load_proportional_to_width target_load ((target_load * reference_width) / reference_load) :=
by sorry

/-- The width of a beam supporting 250 pounds, given a reference beam of 3.5 inches
    supporting 583.3333 pounds, is 1.5 inches -/
theorem beam_width_250_pounds :
  (250 : ℝ) * (3.5 : ℝ) / (583.3333 : ℝ) = (1.5 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_beam_width_calculation_beam_width_250_pounds_l3491_349137


namespace NUMINAMATH_CALUDE_average_first_16_even_numbers_l3491_349190

theorem average_first_16_even_numbers :
  let first_even : ℕ := 2
  let last_even : ℕ := 32
  let count : ℕ := 16
  (first_even + last_even) / 2 = 17 :=
by sorry

end NUMINAMATH_CALUDE_average_first_16_even_numbers_l3491_349190


namespace NUMINAMATH_CALUDE_willie_stickers_l3491_349111

theorem willie_stickers (initial : ℝ) (received : ℝ) (total : ℝ) :
  initial = 278.5 →
  received = 43.8 →
  total = initial + received →
  total = 322.3 := by
sorry

end NUMINAMATH_CALUDE_willie_stickers_l3491_349111


namespace NUMINAMATH_CALUDE_sum_always_positive_l3491_349106

variable {f : ℝ → ℝ}
variable {a : ℕ → ℝ}

def MonotonicIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_always_positive
  (h_incr : MonotonicIncreasing f)
  (h_odd : OddFunction f)
  (h_arith : ArithmeticSequence a)
  (h_a3_pos : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_always_positive_l3491_349106


namespace NUMINAMATH_CALUDE_no_function_satisfying_condition_l3491_349114

open Real

-- Define the type for positive real numbers
def PositiveReal := {x : ℝ // x > 0}

-- State the theorem
theorem no_function_satisfying_condition :
  ¬ ∃ (f : PositiveReal → PositiveReal),
    ∀ (x y : PositiveReal),
      (f (⟨x.val + y.val, sorry⟩)).val ^ 2 ≥ (f x).val ^ 2 * (1 + y.val * (f x).val) :=
by sorry

end NUMINAMATH_CALUDE_no_function_satisfying_condition_l3491_349114


namespace NUMINAMATH_CALUDE_inequality_proof_l3491_349144

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xyz : x * y * z ≥ 1) :
  (x^4 + y) * (y^4 + z) * (z^4 + x) ≥ (x + y^2) * (y + z^2) * (z + x^2) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3491_349144


namespace NUMINAMATH_CALUDE_third_dimension_of_smaller_box_l3491_349100

/-- The length of the third dimension of the smaller box -/
def h : ℕ := sorry

/-- The volume of the larger box -/
def large_box_volume : ℕ := 12 * 14 * 16

/-- The volume of a single smaller box -/
def small_box_volume (h : ℕ) : ℕ := 3 * 7 * h

/-- The number of smaller boxes that fit into the larger box -/
def num_boxes : ℕ := 64

theorem third_dimension_of_smaller_box :
  (num_boxes * small_box_volume h ≤ large_box_volume) → h = 2 := by
  sorry

end NUMINAMATH_CALUDE_third_dimension_of_smaller_box_l3491_349100


namespace NUMINAMATH_CALUDE_stock_worth_calculation_l3491_349169

/-- Proves that the total worth of the stock is 20000 given the specified conditions --/
theorem stock_worth_calculation (stock_worth : ℝ) : 
  (0.2 * stock_worth * 1.1 + 0.8 * stock_worth * 0.95 = stock_worth - 400) → 
  stock_worth = 20000 := by
  sorry

end NUMINAMATH_CALUDE_stock_worth_calculation_l3491_349169


namespace NUMINAMATH_CALUDE_water_park_admission_charge_l3491_349182

/-- Calculates the total admission charge for an adult and accompanying children in a water park. -/
def totalAdmissionCharge (adultCharge childCharge : ℚ) (numChildren : ℕ) : ℚ :=
  adultCharge + childCharge * numChildren

/-- Proves that the total admission charge for an adult and 3 children is $3.25 -/
theorem water_park_admission_charge :
  let adultCharge : ℚ := 1
  let childCharge : ℚ := 3/4
  let numChildren : ℕ := 3
  totalAdmissionCharge adultCharge childCharge numChildren = 13/4 := by
sorry

#eval totalAdmissionCharge 1 (3/4) 3

end NUMINAMATH_CALUDE_water_park_admission_charge_l3491_349182


namespace NUMINAMATH_CALUDE_second_concert_attendance_l3491_349153

theorem second_concert_attendance 
  (first_concert : ℕ) 
  (additional_attendees : ℕ) 
  (h1 : first_concert = 65899) 
  (h2 : additional_attendees = 119) : 
  first_concert + additional_attendees = 66018 := by
  sorry

end NUMINAMATH_CALUDE_second_concert_attendance_l3491_349153


namespace NUMINAMATH_CALUDE_roots_product_equality_l3491_349195

theorem roots_product_equality (p q : ℝ) (α β γ δ : ℝ) 
  (h1 : α^2 + p*α - 2 = 0)
  (h2 : β^2 + p*β - 2 = 0)
  (h3 : γ^2 + q*γ - 2 = 0)
  (h4 : δ^2 + q*δ - 2 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -(p^2 - q^2) := by
  sorry

end NUMINAMATH_CALUDE_roots_product_equality_l3491_349195


namespace NUMINAMATH_CALUDE_purchase_price_l3491_349196

/-- The total price of a purchase of shirts and a tie -/
def total_price (shirt_price : ℝ) (tie_price : ℝ) (discount : ℝ) : ℝ :=
  2 * shirt_price + tie_price + shirt_price * (1 - discount)

/-- The proposition that the total price is 3500 rubles -/
theorem purchase_price :
  ∃ (shirt_price tie_price : ℝ),
    2 * shirt_price + tie_price = 2600 ∧
    total_price shirt_price tie_price 0.25 = 3500 ∧
    shirt_price = 1200 := by
  sorry

end NUMINAMATH_CALUDE_purchase_price_l3491_349196
