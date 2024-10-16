import Mathlib

namespace NUMINAMATH_CALUDE_triangle_property_l2498_249848

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C,
    prove that if sin A(a^2 + b^2 - c^2) = ab(2sin B - sin C),
    then A = π/3 and 3/2 < sin B + sin C ≤ √3 -/
theorem triangle_property (a b c A B C : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_triangle : A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_condition : Real.sin A * (a^2 + b^2 - c^2) = a * b * (2 * Real.sin B - Real.sin C)) :
  A = π/3 ∧ 3/2 < Real.sin B + Real.sin C ∧ Real.sin B + Real.sin C ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l2498_249848


namespace NUMINAMATH_CALUDE_sequence_limit_zero_l2498_249872

/-- Given an infinite sequence {a_n} where the limit of (a_{n+1} - a_n/2) as n approaches infinity is 0,
    prove that the limit of a_n as n approaches infinity is 0. -/
theorem sequence_limit_zero
  (a : ℕ → ℝ)
  (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |a (n + 1) - a n / 2| < ε) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n| < ε :=
sorry

end NUMINAMATH_CALUDE_sequence_limit_zero_l2498_249872


namespace NUMINAMATH_CALUDE_min_minutes_for_plan_b_cheaper_l2498_249837

def plan_a_cost (minutes : ℕ) : ℚ := 15 + (12 / 100) * minutes
def plan_b_cost (minutes : ℕ) : ℚ := 30 + (6 / 100) * minutes

theorem min_minutes_for_plan_b_cheaper : 
  ∀ m : ℕ, m ≥ 251 → plan_b_cost m < plan_a_cost m ∧
  ∀ n : ℕ, n < 251 → plan_a_cost n ≤ plan_b_cost n :=
by sorry

end NUMINAMATH_CALUDE_min_minutes_for_plan_b_cheaper_l2498_249837


namespace NUMINAMATH_CALUDE_mom_shirt_purchase_l2498_249807

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom needs to buy -/
def packages_to_buy : ℚ := 11.83333333

/-- The total number of t-shirts Mom wants to buy -/
def total_shirts : ℕ := 71

theorem mom_shirt_purchase :
  ⌊(packages_to_buy * shirts_per_package : ℚ)⌋ = total_shirts := by
  sorry

end NUMINAMATH_CALUDE_mom_shirt_purchase_l2498_249807


namespace NUMINAMATH_CALUDE_coefficient_x5y4_in_expansion_x_plus_y_9_l2498_249843

theorem coefficient_x5y4_in_expansion_x_plus_y_9 :
  (Finset.range 10).sum (λ k => Nat.choose 9 k * X^k * Y^(9 - k)) =
  126 * X^5 * Y^4 + (Finset.range 10).sum (λ k => if k ≠ 5 then Nat.choose 9 k * X^k * Y^(9 - k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x5y4_in_expansion_x_plus_y_9_l2498_249843


namespace NUMINAMATH_CALUDE_factor_y6_minus_64_l2498_249808

theorem factor_y6_minus_64 (y : ℝ) : 
  y^6 - 64 = (y - 2) * (y + 2) * (y^2 + 2*y + 4) * (y^2 - 2*y + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_y6_minus_64_l2498_249808


namespace NUMINAMATH_CALUDE_equation_equivalence_l2498_249813

theorem equation_equivalence (x y : ℝ) :
  (2*x - 3*y)^2 = 4*x^2 + 9*y^2 ↔ x*y = 0 :=
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2498_249813


namespace NUMINAMATH_CALUDE_function_composition_equality_l2498_249804

theorem function_composition_equality 
  (m n p q : ℝ) 
  (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = m * x + n) 
  (hg : ∀ x, g x = p * x + q) : 
  (∀ x, f (g x) = g (f x)) ↔ m + q = n + p := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l2498_249804


namespace NUMINAMATH_CALUDE_always_positive_l2498_249864

theorem always_positive (x : ℝ) : x^2 + |x| + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_l2498_249864


namespace NUMINAMATH_CALUDE_go_relay_match_sequences_l2498_249881

/-- Represents the number of players in each team -/
def team_size : ℕ := 7

/-- Represents the maximum number of matches possible -/
def max_matches : ℕ := 2 * team_size - 1

/-- Represents the number of matches the winning team must win -/
def required_wins : ℕ := team_size

/-- The number of possible match sequences in a Go relay match -/
def match_sequences : ℕ := 2 * (Nat.choose max_matches required_wins)

theorem go_relay_match_sequences :
  match_sequences = 3432 :=
sorry

end NUMINAMATH_CALUDE_go_relay_match_sequences_l2498_249881


namespace NUMINAMATH_CALUDE_right_triangle_ratio_square_l2498_249845

theorem right_triangle_ratio_square (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) 
  (h4 : c^2 = a^2 + b^2) (h5 : a / b = b / c) : (a / b)^2 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_square_l2498_249845


namespace NUMINAMATH_CALUDE_factorial_14_mod_17_l2498_249886

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem factorial_14_mod_17 : 
  factorial 14 % 17 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_factorial_14_mod_17_l2498_249886


namespace NUMINAMATH_CALUDE_robert_interest_l2498_249849

/-- Calculates the total interest earned in a year given an inheritance amount,
    two interest rates, and the amount invested at the higher rate. -/
def total_interest (inheritance : ℝ) (rate1 : ℝ) (rate2 : ℝ) (amount_at_rate2 : ℝ) : ℝ :=
  let amount_at_rate1 := inheritance - amount_at_rate2
  let interest1 := amount_at_rate1 * rate1
  let interest2 := amount_at_rate2 * rate2
  interest1 + interest2

/-- Theorem stating that given Robert's inheritance and investment conditions,
    the total interest earned in a year is $227. -/
theorem robert_interest :
  total_interest 4000 0.05 0.065 1800 = 227 := by
  sorry

end NUMINAMATH_CALUDE_robert_interest_l2498_249849


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2498_249863

theorem inequality_system_solution (x : ℝ) : 
  (x - 2 > 1 ∧ -2 * x ≤ 4) ↔ x > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2498_249863


namespace NUMINAMATH_CALUDE_cos_neg_570_deg_l2498_249878

-- Define the cosine function for degrees
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

-- State the theorem
theorem cos_neg_570_deg : cos_deg (-570) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_570_deg_l2498_249878


namespace NUMINAMATH_CALUDE_grocery_store_lite_soda_l2498_249861

/-- Given a grocery store with soda bottles, proves that the number of lite soda bottles is 60 -/
theorem grocery_store_lite_soda (regular : ℕ) (diet : ℕ) (lite : ℕ) 
  (h1 : regular = 81)
  (h2 : diet = 60)
  (h3 : diet = lite) : 
  lite = 60 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_lite_soda_l2498_249861


namespace NUMINAMATH_CALUDE_rhombus_in_quadrilateral_l2498_249812

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Represents a rhombus -/
structure Rhombus :=
  (X Y Z V : Point)

/-- Checks if two line segments are parallel -/
def are_parallel (P1 P2 Q1 Q2 : Point) : Prop :=
  (P2.x - P1.x) * (Q2.y - Q1.y) = (P2.y - P1.y) * (Q2.x - Q1.x)

/-- Checks if a point is inside a quadrilateral -/
def is_inside (P : Point) (Q : Quadrilateral) : Prop :=
  sorry -- Definition of a point being inside a quadrilateral

/-- Main theorem: There exists a rhombus within a given quadrilateral
    such that its sides are parallel to the quadrilateral's diagonals -/
theorem rhombus_in_quadrilateral (ABCD : Quadrilateral) :
  ∃ (XYZV : Rhombus),
    (is_inside XYZV.X ABCD) ∧ (is_inside XYZV.Y ABCD) ∧
    (is_inside XYZV.Z ABCD) ∧ (is_inside XYZV.V ABCD) ∧
    (are_parallel XYZV.X XYZV.Y ABCD.A ABCD.C) ∧
    (are_parallel XYZV.X XYZV.Z ABCD.B ABCD.D) ∧
    (are_parallel XYZV.Y XYZV.Z ABCD.A ABCD.C) ∧
    (are_parallel XYZV.V XYZV.Y ABCD.B ABCD.D) :=
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_rhombus_in_quadrilateral_l2498_249812


namespace NUMINAMATH_CALUDE_smallest_b_quadratic_inequality_l2498_249821

theorem smallest_b_quadratic_inequality :
  let f : ℝ → ℝ := fun b => -3 * b^2 + 13 * b - 10
  ∃ b_min : ℝ, b_min = -2/3 ∧
    (∀ b : ℝ, f b ≥ 0 → b ≥ b_min) ∧
    f b_min ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_quadratic_inequality_l2498_249821


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l2498_249811

theorem complex_modulus_equality : Complex.abs (1/3 - 3*I) = Real.sqrt 82 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l2498_249811


namespace NUMINAMATH_CALUDE_airline_seats_per_row_l2498_249817

/-- Proves that the number of seats in each row is 7 for an airline company with given conditions. -/
theorem airline_seats_per_row :
  let num_airplanes : ℕ := 5
  let rows_per_airplane : ℕ := 20
  let flights_per_airplane_per_day : ℕ := 2
  let total_passengers_per_day : ℕ := 1400
  let seats_per_row : ℕ := total_passengers_per_day / (num_airplanes * flights_per_airplane_per_day * rows_per_airplane)
  seats_per_row = 7 := by sorry

end NUMINAMATH_CALUDE_airline_seats_per_row_l2498_249817


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2498_249868

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  x + y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 4/y₀ = 1 ∧ x₀ + y₀ = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2498_249868


namespace NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l2498_249818

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l2498_249818


namespace NUMINAMATH_CALUDE_expression_equals_two_l2498_249846

theorem expression_equals_two :
  (Real.sqrt 3) ^ 0 + 2 ^ (-1 : ℤ) + Real.sqrt 2 * Real.cos (45 * π / 180) - |-(1/2)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l2498_249846


namespace NUMINAMATH_CALUDE_induction_base_case_not_always_one_l2498_249858

/-- In mathematical induction, the base case is not always n = 1. -/
theorem induction_base_case_not_always_one : ∃ (P : ℕ → Prop) (n₀ : ℕ), 
  n₀ ≠ 1 ∧ (∀ n ≥ n₀, P n → P (n + 1)) → (∀ n ≥ n₀, P n) :=
sorry

end NUMINAMATH_CALUDE_induction_base_case_not_always_one_l2498_249858


namespace NUMINAMATH_CALUDE_range_of_x_when_m_is_two_range_of_m_given_inequality_l2498_249895

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |m * x + 1| + |2 * x - 3|

-- Theorem 1
theorem range_of_x_when_m_is_two :
  ∀ x : ℝ, f 2 x = 4 ↔ -1/2 ≤ x ∧ x ≤ 3/2 := by sorry

-- Theorem 2
theorem range_of_m_given_inequality :
  (∀ a : ℝ, a > 0 → f m 1 ≤ (2 * a^2 + 8) / a) → -8 ≤ m ∧ m ≤ 6 := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_m_is_two_range_of_m_given_inequality_l2498_249895


namespace NUMINAMATH_CALUDE_negative_inequality_l2498_249816

theorem negative_inequality (a b : ℝ) (h : a > b) : -2 - a < -2 - b := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l2498_249816


namespace NUMINAMATH_CALUDE_hearts_clubs_equal_prob_l2498_249882

/-- Represents the suits in a standard deck of playing cards -/
inductive Suit
| Hearts
| Diamonds
| Clubs
| Spades

/-- Represents the ranks in a standard deck of playing cards -/
inductive Rank
| Ace
| Two
| Three
| Four
| Five
| Six
| Seven
| Eight
| Nine
| Ten
| Jack
| Queen
| King

/-- Represents a playing card -/
structure Card where
  suit : Suit
  rank : Rank

/-- Represents a standard deck of playing cards -/
def Deck : Type := List Card

/-- The number of cards of each suit in a standard deck -/
def cardsPerSuit : Nat := 13

/-- The total number of cards in a standard deck (excluding Jokers) -/
def totalCards : Nat := 52

/-- The probability of drawing a specific suit from a standard deck -/
def probSuit (s : Suit) : Rat := cardsPerSuit / totalCards

theorem hearts_clubs_equal_prob :
  probSuit Suit.Hearts = probSuit Suit.Clubs := by
  sorry

end NUMINAMATH_CALUDE_hearts_clubs_equal_prob_l2498_249882


namespace NUMINAMATH_CALUDE_benjamins_speed_l2498_249835

/-- Given a distance of 800 kilometers and a time of 10 hours, prove that the speed is 80 kilometers per hour. -/
theorem benjamins_speed (distance : ℝ) (time : ℝ) (h1 : distance = 800) (h2 : time = 10) :
  distance / time = 80 := by
  sorry

end NUMINAMATH_CALUDE_benjamins_speed_l2498_249835


namespace NUMINAMATH_CALUDE_sequence_general_term_l2498_249854

theorem sequence_general_term (a : ℕ → ℤ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 3) :
  ∀ n : ℕ, a n = 3 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2498_249854


namespace NUMINAMATH_CALUDE_exam_score_standard_deviations_l2498_249893

/-- Given an exam with mean score and standard deviation, prove the number of standard deviations above the mean for a specific score -/
theorem exam_score_standard_deviations (mean sd : ℝ) (x : ℝ) 
  (h1 : mean - 2 * sd = 58)
  (h2 : mean = 74)
  (h3 : mean + x * sd = 98) :
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_exam_score_standard_deviations_l2498_249893


namespace NUMINAMATH_CALUDE_ages_sum_l2498_249836

/-- Represents the ages of three people A, B, and C --/
structure Ages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The conditions of the problem --/
def problem_conditions (ages : Ages) : Prop :=
  ages.b = 30 ∧ 
  ∃ x : ℕ, x > 0 ∧ 
    ages.a - 10 = x ∧
    ages.b - 10 = 2 * x ∧
    ages.c - 10 = 3 * x

/-- The theorem to prove --/
theorem ages_sum (ages : Ages) : 
  problem_conditions ages → ages.a + ages.b + ages.c = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l2498_249836


namespace NUMINAMATH_CALUDE_min_magnitude_in_A_l2498_249822

def a : Fin 3 → ℝ := ![1, 2, 3]
def b : Fin 3 → ℝ := ![1, -1, 1]

def A : Set (Fin 3 → ℝ) :=
  {x | ∃ k : ℤ, x = fun i => a i + k * b i}

theorem min_magnitude_in_A :
  ∃ x ∈ A, ∀ y ∈ A, ‖x‖ ≤ ‖y‖ ∧ ‖x‖ = Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_min_magnitude_in_A_l2498_249822


namespace NUMINAMATH_CALUDE_animal_arrangement_count_l2498_249892

/-- The number of ways to arrange animals in cages -/
def arrange_animals (num_chickens num_dogs num_cats num_rabbits : ℕ) : ℕ :=
  Nat.factorial 4 * 
  Nat.factorial num_chickens * 
  Nat.factorial num_dogs * 
  Nat.factorial num_cats * 
  Nat.factorial num_rabbits

/-- Theorem stating the number of arrangements for the given animals -/
theorem animal_arrangement_count :
  arrange_animals 3 3 4 2 = 41472 := by
  sorry

end NUMINAMATH_CALUDE_animal_arrangement_count_l2498_249892


namespace NUMINAMATH_CALUDE_camp_children_count_l2498_249869

/-- Represents the number of children currently in the camp -/
def current_children : ℕ := sorry

/-- Represents the fraction of boys in the camp -/
def boy_fraction : ℚ := 9/10

/-- Represents the fraction of girls in the camp -/
def girl_fraction : ℚ := 1 - boy_fraction

/-- Represents the desired fraction of girls after adding more boys -/
def desired_girl_fraction : ℚ := 1/20

/-- Represents the number of additional boys to be added -/
def additional_boys : ℕ := 60

/-- Theorem stating that the current number of children in the camp is 60 -/
theorem camp_children_count : current_children = 60 := by
  sorry

end NUMINAMATH_CALUDE_camp_children_count_l2498_249869


namespace NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_min_product_exists_l2498_249870

theorem min_product_of_reciprocal_sum (x y : ℕ+) : 
  (1 : ℚ) / x + (1 : ℚ) / (2 * y) = (1 : ℚ) / 7 → x * y ≥ 98 := by
  sorry

theorem min_product_exists : 
  ∃ (x y : ℕ+), (1 : ℚ) / x + (1 : ℚ) / (2 * y) = (1 : ℚ) / 7 ∧ x * y = 98 := by
  sorry

end NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_min_product_exists_l2498_249870


namespace NUMINAMATH_CALUDE_elinas_garden_area_l2498_249810

/-- The area of Elina's rectangular garden --/
def garden_area (length width : ℝ) : ℝ := length * width

/-- The perimeter of Elina's rectangular garden --/
def garden_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

theorem elinas_garden_area :
  ∃ (length width : ℝ),
    length > 0 ∧
    width > 0 ∧
    length * 30 = 1500 ∧
    garden_perimeter length width * 12 = 1500 ∧
    garden_area length width = 625 := by
  sorry

end NUMINAMATH_CALUDE_elinas_garden_area_l2498_249810


namespace NUMINAMATH_CALUDE_iced_tea_price_l2498_249899

/-- The cost of a beverage order --/
def order_cost (cappuccino_price : ℚ) (latte_price : ℚ) (espresso_price : ℚ) (iced_tea_price : ℚ) : ℚ :=
  3 * cappuccino_price + 2 * iced_tea_price + 2 * latte_price + 2 * espresso_price

theorem iced_tea_price (cappuccino_price latte_price espresso_price : ℚ)
  (h1 : cappuccino_price = 2)
  (h2 : latte_price = 3/2)
  (h3 : espresso_price = 1)
  (h4 : ∃ (x : ℚ), order_cost cappuccino_price latte_price espresso_price x = 17) :
  ∃ (x : ℚ), x = 3 ∧ order_cost cappuccino_price latte_price espresso_price x = 17 :=
sorry

end NUMINAMATH_CALUDE_iced_tea_price_l2498_249899


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2498_249889

theorem right_triangle_side_length (A B C : Real) (tanA : Real) (AC : Real) :
  tanA = 3 / 5 →
  AC = 10 →
  A^2 + B^2 = C^2 →
  A / C = tanA →
  B = 2 * Real.sqrt 34 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2498_249889


namespace NUMINAMATH_CALUDE_day_90_of_year_N_minus_1_is_thursday_l2498_249839

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a specific day in a year -/
structure DayInYear where
  year : ℤ
  dayNumber : ℕ

/-- Returns the day of the week for a given day in a year -/
def dayOfWeek (d : DayInYear) : DayOfWeek :=
  sorry

/-- Checks if a given year is a leap year -/
def isLeapYear (year : ℤ) : Bool :=
  sorry

theorem day_90_of_year_N_minus_1_is_thursday
  (N : ℤ)
  (h1 : dayOfWeek ⟨N, 150⟩ = DayOfWeek.Sunday)
  (h2 : dayOfWeek ⟨N + 2, 220⟩ = DayOfWeek.Sunday) :
  dayOfWeek ⟨N - 1, 90⟩ = DayOfWeek.Thursday :=
by sorry

end NUMINAMATH_CALUDE_day_90_of_year_N_minus_1_is_thursday_l2498_249839


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l2498_249853

theorem framed_painting_ratio :
  ∀ (x : ℝ),
  x > 0 →
  (20 + 2*x) * (30 + 6*x) = 1800 →
  (20 + 2*x) / (30 + 6*x) = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l2498_249853


namespace NUMINAMATH_CALUDE_initial_seashells_count_l2498_249819

/-- The number of seashells Tim found initially -/
def initial_seashells : ℕ := sorry

/-- The number of starfish Tim found -/
def starfish : ℕ := 110

/-- The number of seashells Tim gave to Sara -/
def seashells_given : ℕ := 172

/-- The number of seashells Tim has now -/
def current_seashells : ℕ := 507

/-- Theorem stating that the initial number of seashells is equal to 
    the current number of seashells plus the number of seashells given away -/
theorem initial_seashells_count : 
  initial_seashells = current_seashells + seashells_given := by sorry

end NUMINAMATH_CALUDE_initial_seashells_count_l2498_249819


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l2498_249801

/-- Circle1 is defined by the equation x^2 - 6x + y^2 + 10y + 9 = 0 -/
def Circle1 (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 10*y + 9 = 0

/-- Circle2 is defined by the equation x^2 + 4x + y^2 - 8y + 4 = 0 -/
def Circle2 (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 8*y + 4 = 0

/-- The shortest distance between Circle1 and Circle2 is √106 - 9 -/
theorem shortest_distance_between_circles :
  ∃ (d : ℝ), d = Real.sqrt 106 - 9 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    Circle1 x₁ y₁ → Circle2 x₂ y₂ →
    d ≤ Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l2498_249801


namespace NUMINAMATH_CALUDE_cone_lateral_area_l2498_249833

/-- The lateral area of a cone with base radius 6 cm and height 8 cm is 60π cm². -/
theorem cone_lateral_area : 
  let r : ℝ := 6  -- base radius in cm
  let h : ℝ := 8  -- height in cm
  let l : ℝ := (r^2 + h^2).sqrt  -- slant height
  let lateral_area : ℝ := π * r * l  -- formula for lateral area
  lateral_area = 60 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l2498_249833


namespace NUMINAMATH_CALUDE_cats_after_sale_l2498_249857

/-- The number of cats remaining after a sale at a pet store -/
theorem cats_after_sale (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 15 → house = 49 → sold = 19 → siamese + house - sold = 45 := by
  sorry

end NUMINAMATH_CALUDE_cats_after_sale_l2498_249857


namespace NUMINAMATH_CALUDE_is_center_of_hyperbola_l2498_249875

/-- The equation of a hyperbola in the form ((4y-8)^2 / 7^2) - ((2x+6)^2 / 9^2) = 1 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * y - 8)^2 / 7^2 - (2 * x + 6)^2 / 9^2 = 1

/-- The center of a hyperbola -/
def hyperbola_center : ℝ × ℝ := (-3, 2)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem is_center_of_hyperbola :
  ∀ x y : ℝ, hyperbola_equation x y ↔ hyperbola_equation (x - hyperbola_center.1) (y - hyperbola_center.2) :=
by sorry

end NUMINAMATH_CALUDE_is_center_of_hyperbola_l2498_249875


namespace NUMINAMATH_CALUDE_hyperbola_intersection_l2498_249880

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and left focus F₁ at (-c, 0),
    the intersection points of the line x = -c with the hyperbola are (-c, ±b²/a) -/
theorem hyperbola_intersection (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}
  let vertical_line := {p : ℝ × ℝ | p.1 = -c}
  let intersection := hyperbola ∩ vertical_line
  intersection = {(-c, b^2/a), (-c, -b^2/a)} := by sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_l2498_249880


namespace NUMINAMATH_CALUDE_opposite_of_three_l2498_249844

/-- The opposite of a real number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℝ) : ℝ := -x

/-- The opposite of 3 is -3. -/
theorem opposite_of_three : opposite 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l2498_249844


namespace NUMINAMATH_CALUDE_trip_length_l2498_249825

theorem trip_length : 
  ∀ (x : ℚ), 
  (1/4 : ℚ) * x + 36 + (1/3 : ℚ) * x = x → 
  x = 432/5 := by
sorry

end NUMINAMATH_CALUDE_trip_length_l2498_249825


namespace NUMINAMATH_CALUDE_rectangle_area_18_pairs_l2498_249874

def rectangle_area_18 : Set (ℕ × ℕ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 18}

theorem rectangle_area_18_pairs : 
  rectangle_area_18 = {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_18_pairs_l2498_249874


namespace NUMINAMATH_CALUDE_exists_k_good_iff_k_ge_two_l2498_249866

/-- A function is k-good if the GCD of f(m) + n and f(n) + m is at most k for all m ≠ n -/
def IsKGood (k : ℕ) (f : ℕ+ → ℕ+) : Prop :=
  ∀ (m n : ℕ+), m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

/-- There exists a k-good function if and only if k ≥ 2 -/
theorem exists_k_good_iff_k_ge_two :
  ∀ k : ℕ, (∃ f : ℕ+ → ℕ+, IsKGood k f) ↔ k ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_exists_k_good_iff_k_ge_two_l2498_249866


namespace NUMINAMATH_CALUDE_intersection_of_H_and_G_l2498_249888

def H : Set ℕ := {2, 3, 4}
def G : Set ℕ := {1, 3}

theorem intersection_of_H_and_G : H ∩ G = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_H_and_G_l2498_249888


namespace NUMINAMATH_CALUDE_sum_of_integers_with_given_difference_and_product_l2498_249890

theorem sum_of_integers_with_given_difference_and_product :
  ∀ x y : ℕ+, 
    (x : ℝ) - (y : ℝ) = 10 →
    (x : ℝ) * (y : ℝ) = 56 →
    (x : ℝ) + (y : ℝ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_with_given_difference_and_product_l2498_249890


namespace NUMINAMATH_CALUDE_calculation_proof_l2498_249876

theorem calculation_proof : 3 * 15 + 3 * 16 + 3 * 19 + 11 = 161 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2498_249876


namespace NUMINAMATH_CALUDE_language_class_selection_probability_l2498_249841

/-- The probability of selecting two students from different language classes -/
theorem language_class_selection_probability
  (total_students : ℕ)
  (french_students : ℕ)
  (spanish_students : ℕ)
  (no_language_students : ℕ)
  (h1 : total_students = 30)
  (h2 : french_students = 22)
  (h3 : spanish_students = 24)
  (h4 : no_language_students = 2)
  (h5 : french_students + spanish_students - (total_students - no_language_students) + no_language_students = total_students) :
  let both_classes := french_students + spanish_students - (total_students - no_language_students)
  let only_french := french_students - both_classes
  let only_spanish := spanish_students - both_classes
  let total_combinations := total_students.choose 2
  let undesirable_outcomes := (only_french.choose 2) + (only_spanish.choose 2)
  (1 : ℚ) - (undesirable_outcomes : ℚ) / (total_combinations : ℚ) = 14 / 15 :=
by sorry

end NUMINAMATH_CALUDE_language_class_selection_probability_l2498_249841


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2498_249879

theorem quadratic_inequality (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2498_249879


namespace NUMINAMATH_CALUDE_S_is_infinite_l2498_249832

/-- The largest prime divisor of a positive integer -/
def largest_prime_divisor (n : ℕ+) : ℕ+ :=
  sorry

/-- The set of positive integers n where the largest prime divisor of n^4 + n^2 + 1
    equals the largest prime divisor of (n+1)^4 + (n+1)^2 + 1 -/
def S : Set ℕ+ :=
  {n | largest_prime_divisor (n^4 + n^2 + 1) = largest_prime_divisor ((n+1)^4 + (n+1)^2 + 1)}

/-- The main theorem: S is an infinite set -/
theorem S_is_infinite : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_S_is_infinite_l2498_249832


namespace NUMINAMATH_CALUDE_shaded_square_area_ratio_l2498_249840

/-- The ratio of the area of a square with vertices at (2,3), (4,3), (4,5), and (2,5) 
    to the area of a 6x6 square is 1/9. -/
theorem shaded_square_area_ratio : 
  let grid_size : ℕ := 6
  let vertex1 : (ℕ × ℕ) := (2, 3)
  let vertex2 : (ℕ × ℕ) := (4, 3)
  let vertex3 : (ℕ × ℕ) := (4, 5)
  let vertex4 : (ℕ × ℕ) := (2, 5)
  let shaded_square_side : ℕ := vertex2.1 - vertex1.1
  let shaded_square_area : ℕ := shaded_square_side * shaded_square_side
  let grid_area : ℕ := grid_size * grid_size
  (shaded_square_area : ℚ) / grid_area = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_shaded_square_area_ratio_l2498_249840


namespace NUMINAMATH_CALUDE_cos_sin_inequalities_l2498_249855

theorem cos_sin_inequalities (x : ℝ) (h : x > 0) : 
  (Real.cos x > 1 - x^2 / 2) ∧ (Real.sin x > x - x^3 / 6) := by sorry

end NUMINAMATH_CALUDE_cos_sin_inequalities_l2498_249855


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2498_249891

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 2
  f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2498_249891


namespace NUMINAMATH_CALUDE_cookie_rows_per_tray_l2498_249830

/-- Given the total number of cookies, cookies per row, and number of baking trays,
    calculate the number of rows of cookies on each baking tray. -/
def rows_per_tray (total_cookies : ℕ) (cookies_per_row : ℕ) (num_trays : ℕ) : ℕ :=
  (total_cookies / cookies_per_row) / num_trays

/-- Theorem stating that with 120 total cookies, 6 cookies per row, and 4 baking trays,
    there are 5 rows of cookies on each baking tray. -/
theorem cookie_rows_per_tray :
  rows_per_tray 120 6 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cookie_rows_per_tray_l2498_249830


namespace NUMINAMATH_CALUDE_smallest_sports_team_size_l2498_249809

theorem smallest_sports_team_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 3 = 1 ∧ 
  n % 4 = 2 ∧ 
  n % 6 = 4 ∧ 
  ∃ m : ℕ, n = m ^ 2 ∧
  ∀ k : ℕ, k > 0 → k % 3 = 1 → k % 4 = 2 → k % 6 = 4 → (∃ l : ℕ, k = l ^ 2) → k ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_sports_team_size_l2498_249809


namespace NUMINAMATH_CALUDE_max_side_length_exists_max_side_length_l2498_249838

/-- A triangle with integer side lengths and perimeter 24 -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  perimeter : a + b + c = 24

/-- The maximum side length of a triangle is 11 -/
theorem max_side_length (t : Triangle) : t.a ≤ 11 ∧ t.b ≤ 11 ∧ t.c ≤ 11 :=
sorry

/-- There exists a triangle with maximum side length 11 -/
theorem exists_max_side_length : ∃ (t : Triangle), t.a = 11 ∨ t.b = 11 ∨ t.c = 11 :=
sorry

end NUMINAMATH_CALUDE_max_side_length_exists_max_side_length_l2498_249838


namespace NUMINAMATH_CALUDE_divisible_by_132_iff_in_list_l2498_249871

def is_valid_number (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), 
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    n = 1000 * x + 100 * y + 90 + z

theorem divisible_by_132_iff_in_list (n : ℕ) :
  is_valid_number n ∧ n % 132 = 0 ↔ n ∈ [3696, 4092, 6996, 7392] := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_132_iff_in_list_l2498_249871


namespace NUMINAMATH_CALUDE_A_characterization_l2498_249860

/-- The set A defined by the quadratic equation kx^2 - 3x + 2 = 0 -/
def A (k : ℝ) : Set ℝ := {x | k * x^2 - 3 * x + 2 = 0}

/-- Theorem stating the conditions for A to be empty or contain exactly one element -/
theorem A_characterization (k : ℝ) :
  (A k = ∅ ↔ k > 9/8) ∧
  (∃! x, x ∈ A k ↔ k = 0 ∨ k = 9/8) ∧
  (k = 0 → A k = {2/3}) ∧
  (k = 9/8 → A k = {4/3}) :=
sorry

end NUMINAMATH_CALUDE_A_characterization_l2498_249860


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l2498_249803

def product_20_to_30 : ℕ := 20 * 21 * 22 * 23 * 24 * 25 * 26 * 27 * 28 * 29 * 30

theorem units_digit_of_fraction (h : product_20_to_30 % 8000 = 6) :
  (product_20_to_30 / 8000) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l2498_249803


namespace NUMINAMATH_CALUDE_road_greening_costs_l2498_249851

/-- Represents a road greening project with two plans. -/
structure RoadGreeningProject where
  total_length : ℝ
  plan_a_type_a : ℝ
  plan_a_type_b : ℝ
  plan_a_cost : ℝ
  plan_b_type_a : ℝ
  plan_b_type_b : ℝ
  plan_b_cost : ℝ

/-- Calculates the cost per stem of type A and B flowers. -/
def calculate_flower_costs (project : RoadGreeningProject) : ℝ × ℝ := sorry

/-- Calculates the minimum total cost of the project. -/
def calculate_min_cost (project : RoadGreeningProject) : ℝ := sorry

/-- Theorem stating the correct flower costs and minimum project cost. -/
theorem road_greening_costs (project : RoadGreeningProject) 
  (h1 : project.total_length = 1500)
  (h2 : project.plan_a_type_a = 2)
  (h3 : project.plan_a_type_b = 3)
  (h4 : project.plan_a_cost = 22)
  (h5 : project.plan_b_type_a = 1)
  (h6 : project.plan_b_type_b = 5)
  (h7 : project.plan_b_cost = 25) :
  let (cost_a, cost_b) := calculate_flower_costs project
  calculate_flower_costs project = (5, 4) ∧ 
  calculate_min_cost project = 36000 := by
  sorry

end NUMINAMATH_CALUDE_road_greening_costs_l2498_249851


namespace NUMINAMATH_CALUDE_cube_ending_in_eight_and_nine_l2498_249887

theorem cube_ending_in_eight_and_nine :
  ∀ a b : ℕ,
  (10 ≤ a ∧ a < 100) →
  (10 ≤ b ∧ b < 100) →
  (1000 ≤ a^3 ∧ a^3 < 10000) →
  (1000 ≤ b^3 ∧ b^3 < 10000) →
  a^3 % 10 = 8 →
  b^3 % 10 = 9 →
  a = 12 ∧ b = 19 :=
by sorry

end NUMINAMATH_CALUDE_cube_ending_in_eight_and_nine_l2498_249887


namespace NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l2498_249852

-- Define the condition for a hyperbola with foci on the x-axis
def is_hyperbola_x_axis (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0

-- Define the equation
def equation (m n x y : ℝ) : Prop :=
  m * x^2 - n * y^2 = 1

-- Theorem statement
theorem mn_positive_necessary_not_sufficient :
  ∀ m n : ℝ,
    (is_hyperbola_x_axis m n → m * n > 0) ∧
    ¬(m * n > 0 → is_hyperbola_x_axis m n) :=
by sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l2498_249852


namespace NUMINAMATH_CALUDE_jose_age_is_19_l2498_249827

-- Define the ages of the individuals
def inez_age : ℕ := 18
def alice_age : ℕ := inez_age - 3
def zack_age : ℕ := inez_age + 5
def jose_age : ℕ := zack_age - 4

-- Theorem to prove Jose's age
theorem jose_age_is_19 : jose_age = 19 := by
  sorry

end NUMINAMATH_CALUDE_jose_age_is_19_l2498_249827


namespace NUMINAMATH_CALUDE_two_digit_product_less_than_five_digit_l2498_249847

theorem two_digit_product_less_than_five_digit : ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  a * b < 10000 := by
sorry

end NUMINAMATH_CALUDE_two_digit_product_less_than_five_digit_l2498_249847


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_12_with_digit_sum_24_l2498_249856

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_12_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n.mod 12 = 0 → digit_sum n = 24 → n ≤ 996 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_12_with_digit_sum_24_l2498_249856


namespace NUMINAMATH_CALUDE_solution_set_theorem_range_of_b_theorem_l2498_249842

-- Define the real number a
variable (a : ℝ)

-- Define the condition that the solution set of (1-a)x^2 - 4x + 6 > 0 is (-3, 1)
def solution_set_condition : Prop :=
  ∀ x : ℝ, ((1 - a) * x^2 - 4 * x + 6 > 0) ↔ (-3 < x ∧ x < 1)

-- Theorem for the first question
theorem solution_set_theorem (h : solution_set_condition a) :
  ∀ x : ℝ, (2 * x^2 + (2 - a) * x - a > 0) ↔ (x < -1 ∨ x > 3/2) :=
sorry

-- Theorem for the second question
theorem range_of_b_theorem (h : solution_set_condition a) :
  ∀ b : ℝ, (∀ x : ℝ, a * x^2 + b * x + 3 ≥ 0) ↔ (-6 ≤ b ∧ b ≤ 6) :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_range_of_b_theorem_l2498_249842


namespace NUMINAMATH_CALUDE_quadratic_max_value_quadratic_max_value_achieved_l2498_249898

theorem quadratic_max_value (s : ℝ) : -3 * s^2 + 24 * s - 7 ≤ 41 := by sorry

theorem quadratic_max_value_achieved : ∃ s : ℝ, -3 * s^2 + 24 * s - 7 = 41 := by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_quadratic_max_value_achieved_l2498_249898


namespace NUMINAMATH_CALUDE_product_of_roots_l2498_249883

theorem product_of_roots (a b c : ℝ) (h : (5 + 3 * Real.sqrt 5) * a + (3 + Real.sqrt 5) * b + c = 0) :
  a * b = -((15 - 9 * Real.sqrt 5) / 20) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2498_249883


namespace NUMINAMATH_CALUDE_ice_skating_given_skiing_l2498_249884

-- Define the probability space
variable (Ω : Type) [MeasurableSpace Ω] [Fintype Ω] (P : Measure Ω)

-- Define events
variable (A B : Set Ω)

-- Define the probabilities
variable (hA : P A = 0.6)
variable (hB : P B = 0.5)
variable (hAorB : P (A ∪ B) = 0.7)

-- Define the theorem
theorem ice_skating_given_skiing :
  P (A ∩ B) / P B = 0.8 :=
sorry

end NUMINAMATH_CALUDE_ice_skating_given_skiing_l2498_249884


namespace NUMINAMATH_CALUDE_weight_of_A_l2498_249820

theorem weight_of_A (a b c d : ℝ) : 
  (a + b + c) / 3 = 70 →
  (a + b + c + d) / 4 = 70 →
  ((b + c + d + (d + 3)) / 4 = 68) →
  a = 81 := by sorry

end NUMINAMATH_CALUDE_weight_of_A_l2498_249820


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2498_249829

theorem sufficient_not_necessary (a : ℝ) : 
  (a = 3 → a^2 = 9) ∧ (∃ b : ℝ, b ≠ 3 ∧ b^2 = 9) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2498_249829


namespace NUMINAMATH_CALUDE_money_spent_per_week_l2498_249815

/-- 
Given that Mike earned money from mowing lawns and weed eating, and the money lasts for a certain number of weeks,
this theorem proves the amount of money he spent per week.
-/
theorem money_spent_per_week 
  (lawn_money : ℕ) 
  (weed_money : ℕ) 
  (weeks : ℕ) 
  (h1 : lawn_money = 14) 
  (h2 : weed_money = 26) 
  (h3 : weeks = 8) : 
  (lawn_money + weed_money) / weeks = 5 := by
sorry

end NUMINAMATH_CALUDE_money_spent_per_week_l2498_249815


namespace NUMINAMATH_CALUDE_right_triangle_median_ratio_bound_l2498_249834

theorem right_triangle_median_ratio_bound (a b c s_a s_b s_c : ℝ) 
  (h_right : c^2 = a^2 + b^2)
  (h_s_a : s_a^2 = a^2/4 + b^2)
  (h_s_b : s_b^2 = b^2/4 + a^2)
  (h_s_c : s_c = c/2)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  (s_a + s_b) / s_c ≤ Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_median_ratio_bound_l2498_249834


namespace NUMINAMATH_CALUDE_expression_simplification_l2498_249826

theorem expression_simplification (y : ℝ) (h : y ≠ 0) :
  (20 * y^3) * (7 * y^2) * (1 / (2*y)^3) = 17.5 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2498_249826


namespace NUMINAMATH_CALUDE_current_rate_calculation_l2498_249865

/-- Given a boat with speed in still water and distance travelled downstream in a specific time,
    calculate the rate of the current. -/
theorem current_rate_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (time_minutes : ℝ) :
  boat_speed = 24 ∧ downstream_distance = 6.75 ∧ time_minutes = 15 →
  ∃ (current_rate : ℝ), current_rate = 3 ∧
    boat_speed + current_rate = downstream_distance / (time_minutes / 60) :=
by sorry

end NUMINAMATH_CALUDE_current_rate_calculation_l2498_249865


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2498_249823

def is_geometric_sequence (α : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, α (n + 1) = α n * r

theorem geometric_sequence_property (α : ℕ → ℝ) (h_geo : is_geometric_sequence α) 
  (h_prod : α 4 * α 5 * α 6 = 27) : α 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2498_249823


namespace NUMINAMATH_CALUDE_circle_center_distance_l2498_249831

/-- The distance between the center of the circle x^2 + y^2 = 4x + 6y + 3 and the point (5, -2) is √34 -/
theorem circle_center_distance :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 = 4*x + 6*y + 3
  let center : ℝ × ℝ := (2, 3)
  let point : ℝ × ℝ := (5, -2)
  (∃ x y, circle_eq x y) →
  Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2) = Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_circle_center_distance_l2498_249831


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l2498_249885

theorem sqrt_sum_equality : Real.sqrt 8 + Real.sqrt 18 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l2498_249885


namespace NUMINAMATH_CALUDE_negative_143_coterminal_with_37_l2498_249828

/-- An angle is coterminal with 37° if it can be represented as 37° + 180°k, where k is an integer -/
def is_coterminal_with_37 (angle : ℝ) : Prop :=
  ∃ k : ℤ, angle = 37 + 180 * k

/-- Theorem: -143° is coterminal with 37° -/
theorem negative_143_coterminal_with_37 : is_coterminal_with_37 (-143) := by
  sorry

end NUMINAMATH_CALUDE_negative_143_coterminal_with_37_l2498_249828


namespace NUMINAMATH_CALUDE_sqrt_last_digit_exists_l2498_249877

/-- A p-adic number -/
structure PAdicNumber (p : ℕ) where
  digits : ℕ → ℕ
  last_digit : ℕ

/-- The concept of square root in p-arithmetic -/
def has_sqrt_p_adic (α : PAdicNumber p) : Prop :=
  ∃ β : PAdicNumber p, β.digits 0 ^ 2 ≡ α.digits 0 [MOD p]

/-- The main theorem -/
theorem sqrt_last_digit_exists (p : ℕ) (α : PAdicNumber p) :
  has_sqrt_p_adic α → ∃ x : ℕ, x ^ 2 ≡ α.last_digit [MOD p] :=
sorry

end NUMINAMATH_CALUDE_sqrt_last_digit_exists_l2498_249877


namespace NUMINAMATH_CALUDE_flower_beds_count_l2498_249859

/-- Given that 10 seeds are put in each flower bed and 60 seeds were planted altogether,
    prove that the number of flower beds is 6. -/
theorem flower_beds_count (seeds_per_bed : ℕ) (total_seeds : ℕ) (num_beds : ℕ)
  (h1 : seeds_per_bed = 10)
  (h2 : total_seeds = 60)
  (h3 : num_beds * seeds_per_bed = total_seeds) :
  num_beds = 6 := by
  sorry

end NUMINAMATH_CALUDE_flower_beds_count_l2498_249859


namespace NUMINAMATH_CALUDE_math_english_time_difference_l2498_249896

/-- Represents an exam with a number of questions and a duration in hours -/
structure Exam where
  questions : ℕ
  duration : ℚ

/-- Calculates the time per question in minutes for a given exam -/
def timePerQuestion (e : Exam) : ℚ :=
  (e.duration * 60) / e.questions

theorem math_english_time_difference :
  let english : Exam := { questions := 30, duration := 1 }
  let math : Exam := { questions := 15, duration := 1.5 }
  timePerQuestion math - timePerQuestion english = 4 := by
  sorry

end NUMINAMATH_CALUDE_math_english_time_difference_l2498_249896


namespace NUMINAMATH_CALUDE_coin_combinations_theorem_l2498_249805

/-- Represents the denominations of coins available in kopecks -/
def coin_denominations : List Nat := [1, 2, 5, 10, 20, 50]

/-- Represents the total amount to be made in kopecks -/
def total_amount : Nat := 100

/-- 
  Calculates the number of ways to make the total amount using the given coin denominations
  
  @param coins The list of available coin denominations
  @param amount The total amount to be made
  @return The number of ways to make the total amount
-/
def count_ways (coins : List Nat) (amount : Nat) : Nat :=
  sorry

theorem coin_combinations_theorem : 
  count_ways coin_denominations total_amount = 4562 := by
  sorry

end NUMINAMATH_CALUDE_coin_combinations_theorem_l2498_249805


namespace NUMINAMATH_CALUDE_min_value_product_min_value_product_achieved_l2498_249894

theorem min_value_product (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (h : 1/x + 1/y + 1/z + 1/u = 8) : 
  x^3 * y^2 * z * u^2 ≥ 1/432 :=
sorry

theorem min_value_product_achieved (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (h : 1/x + 1/y + 1/z + 1/u = 8) : 
  ∃ (x' y' z' u' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ u' > 0 ∧ 
    1/x' + 1/y' + 1/z' + 1/u' = 8 ∧ 
    x'^3 * y'^2 * z' * u'^2 = 1/432 :=
sorry

end NUMINAMATH_CALUDE_min_value_product_min_value_product_achieved_l2498_249894


namespace NUMINAMATH_CALUDE_average_book_width_l2498_249814

theorem average_book_width :
  let book_widths : List ℚ := [7, 3/4, 5/4, 3, 8, 5/2, 12]
  let num_books : ℕ := 7
  (book_widths.sum / num_books : ℚ) = 241/49 := by
  sorry

end NUMINAMATH_CALUDE_average_book_width_l2498_249814


namespace NUMINAMATH_CALUDE_distance_from_origin_l2498_249850

theorem distance_from_origin (x : ℝ) : |x| = 4 → x = 4 ∨ x = -4 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l2498_249850


namespace NUMINAMATH_CALUDE_no_inscribed_sphere_when_black_exceeds_white_l2498_249800

/-- Represents a face of a polyhedron -/
structure Face where
  area : ℝ
  color : Bool  -- True for black, False for white

/-- Represents a convex polyhedron -/
structure Polyhedron where
  faces : List Face
  is_convex : Bool
  no_adjacent_black : Bool

/-- Checks if a sphere can be inscribed in the polyhedron -/
def can_inscribe_sphere (p : Polyhedron) : Prop :=
  sorry

/-- Calculates the total area of faces of a given color -/
def total_area (p : Polyhedron) (color : Bool) : ℝ :=
  sorry

/-- Main theorem -/
theorem no_inscribed_sphere_when_black_exceeds_white (p : Polyhedron) :
  p.is_convex ∧ 
  p.no_adjacent_black ∧ 
  (total_area p true > total_area p false) →
  ¬(can_inscribe_sphere p) :=
by
  sorry

end NUMINAMATH_CALUDE_no_inscribed_sphere_when_black_exceeds_white_l2498_249800


namespace NUMINAMATH_CALUDE_unique_A_exists_l2498_249897

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def satisfies_conditions (A : ℕ) : Prop :=
  is_single_digit A ∧
  72 % A = 0 ∧
  (354100 + 10 * A + 6) % 4 = 0 ∧
  (354100 + 10 * A + 6) % 9 = 0

theorem unique_A_exists :
  ∃! A, satisfies_conditions A :=
sorry

end NUMINAMATH_CALUDE_unique_A_exists_l2498_249897


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_product_l2498_249862

theorem absolute_value_equation_solution_product : ∃ (x₁ x₂ : ℝ),
  (|2 * x₁ - 1| + 4 = 24) ∧
  (|2 * x₂ - 1| + 4 = 24) ∧
  (x₁ ≠ x₂) ∧
  (x₁ * x₂ = -99.75) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_product_l2498_249862


namespace NUMINAMATH_CALUDE_jan1_2010_is_sunday_l2498_249802

-- Define days of the week
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

-- Theorem statement
theorem jan1_2010_is_sunday :
  advanceDay DayOfWeek.Saturday 3653 = DayOfWeek.Sunday := by
  sorry


end NUMINAMATH_CALUDE_jan1_2010_is_sunday_l2498_249802


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l2498_249867

theorem quadratic_completion_of_square (b : ℝ) (n : ℝ) : 
  (∀ x, x^2 + b*x + 19 = (x + n)^2 - 6) → b > 0 → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l2498_249867


namespace NUMINAMATH_CALUDE_product_of_solutions_l2498_249806

theorem product_of_solutions (x : ℝ) : (|x| = 3 * (|x| - 2)) → ∃ y : ℝ, (|y| = 3 * (|y| - 2)) ∧ x * y = -9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2498_249806


namespace NUMINAMATH_CALUDE_physics_class_grade_distribution_l2498_249824

theorem physics_class_grade_distribution (total_students : ℕ) 
  (prob_A prob_B prob_C prob_D : ℚ) : 
  total_students = 40 →
  prob_A = (1/2) * prob_B →
  prob_C = 2 * prob_B →
  prob_D = (3/10) * prob_B →
  prob_A + prob_B + prob_C + prob_D = 1 →
  (prob_B * total_students : ℚ) = 200/19 :=
by sorry

end NUMINAMATH_CALUDE_physics_class_grade_distribution_l2498_249824


namespace NUMINAMATH_CALUDE_hexagon_shape_partition_ways_l2498_249873

/-- A shape formed by gluing together congruent regular hexagons -/
structure HexagonShape where
  num_hexagons : ℕ
  num_quadrilaterals : ℕ

/-- The number of ways to partition a HexagonShape -/
def partition_ways (shape : HexagonShape) : ℕ :=
  2 ^ shape.num_hexagons

/-- The theorem to prove -/
theorem hexagon_shape_partition_ways :
  ∀ (shape : HexagonShape),
    shape.num_hexagons = 7 →
    shape.num_quadrilaterals = 21 →
    partition_ways shape = 128 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_shape_partition_ways_l2498_249873
