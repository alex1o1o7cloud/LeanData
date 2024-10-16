import Mathlib

namespace NUMINAMATH_CALUDE_f_properties_l206_20642

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

theorem f_properties :
  (∀ x > 1, f x > 0) ∧
  (∀ x, 0 < x → x < 1 → f x < 0) ∧
  (∀ x > 0, f x ≥ -1 / (2 * Real.exp 1)) ∧
  (∀ x > 0, f x ≥ x - 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l206_20642


namespace NUMINAMATH_CALUDE_expand_expression_l206_20632

theorem expand_expression (x : ℝ) : (20 * x - 25) * (3 * x) = 60 * x^2 - 75 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l206_20632


namespace NUMINAMATH_CALUDE_james_vehicle_count_l206_20607

theorem james_vehicle_count :
  let trucks : ℕ := 12
  let buses : ℕ := 2
  let taxis : ℕ := 4
  let cars : ℕ := 30
  let truck_capacity : ℕ := 2
  let bus_capacity : ℕ := 15
  let taxi_capacity : ℕ := 2
  let motorbike_capacity : ℕ := 1
  let car_capacity : ℕ := 3
  let total_passengers : ℕ := 156
  let motorbikes : ℕ := total_passengers - 
    (trucks * truck_capacity + buses * bus_capacity + 
     taxis * taxi_capacity + cars * car_capacity)
  trucks + buses + taxis + motorbikes + cars = 52 := by
sorry

end NUMINAMATH_CALUDE_james_vehicle_count_l206_20607


namespace NUMINAMATH_CALUDE_nested_f_application_l206_20638

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^3 else x + 9

theorem nested_f_application : f (f (f (f (f 3)))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_nested_f_application_l206_20638


namespace NUMINAMATH_CALUDE_james_total_toys_l206_20605

/-- The number of toy cars James buys -/
def toy_cars : ℕ := 20

/-- The number of toy soldiers James buys -/
def toy_soldiers : ℕ := 2 * toy_cars

/-- The total number of toys James buys -/
def total_toys : ℕ := toy_cars + toy_soldiers

/-- Theorem stating that the total number of toys James buys is 60 -/
theorem james_total_toys : total_toys = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_total_toys_l206_20605


namespace NUMINAMATH_CALUDE_s_point_implies_a_value_l206_20667

/-- Definition of an S point for two functions -/
def is_S_point (f g : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = g x₀ ∧ deriv f x₀ = deriv g x₀

/-- The main theorem -/
theorem s_point_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, is_S_point (λ x => a * x^2 - 1) (λ x => Real.log (a * x)) x₀) →
  a = 2 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_s_point_implies_a_value_l206_20667


namespace NUMINAMATH_CALUDE_eventual_stability_l206_20621

/-- Represents a line of 2018 natural numbers -/
def Line := Fin 2018 → ℕ

/-- Applies the frequency counting operation to a line -/
def frequency_count (l : Line) : Line := sorry

/-- Predicate to check if two lines are identical -/
def identical (l1 l2 : Line) : Prop := ∀ i, l1 i = l2 i

/-- Theorem stating that repeated frequency counting eventually leads to identical lines -/
theorem eventual_stability (initial : Line) : 
  ∃ n : ℕ, ∀ m ≥ n, identical (frequency_count^[m] initial) (frequency_count^[m+1] initial) := by
  sorry

end NUMINAMATH_CALUDE_eventual_stability_l206_20621


namespace NUMINAMATH_CALUDE_rectangle_dimensions_quadratic_equation_l206_20652

theorem rectangle_dimensions_quadratic_equation 
  (L W : ℝ) 
  (h1 : L + W = 15) 
  (h2 : L * W = 2 * W^2) : 
  (L = (15 + Real.sqrt 25) / 2 ∧ W = (15 - Real.sqrt 25) / 2) ∨ 
  (L = (15 - Real.sqrt 25) / 2 ∧ W = (15 + Real.sqrt 25) / 2) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_quadratic_equation_l206_20652


namespace NUMINAMATH_CALUDE_bottles_left_l206_20629

theorem bottles_left (initial_bottles drunk_bottles : ℕ) :
  initial_bottles = 17 →
  drunk_bottles = 3 →
  initial_bottles - drunk_bottles = 14 :=
by sorry

end NUMINAMATH_CALUDE_bottles_left_l206_20629


namespace NUMINAMATH_CALUDE_number_line_points_l206_20604

/-- Given a number line with points A, B, and C, prove the positions of B and C -/
theorem number_line_points (A B C : ℚ) : 
  A = 2 → B = A - 7 → C = B + (1 + 2/3) →
  B = -5 ∧ C = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_number_line_points_l206_20604


namespace NUMINAMATH_CALUDE_certain_triangle_angle_sum_l206_20640

/-- A triangle is a shape with three sides -/
structure Triangle where
  sides : Fin 3 → ℝ
  positive : ∀ i, sides i > 0

/-- The sum of interior angles of a triangle is 180° -/
axiom triangle_angle_sum (t : Triangle) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180

/-- For any triangle, the sum of its interior angles is always 180° -/
theorem certain_triangle_angle_sum (t : Triangle) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180 := by
  sorry

end NUMINAMATH_CALUDE_certain_triangle_angle_sum_l206_20640


namespace NUMINAMATH_CALUDE_quadratic_shared_root_property_l206_20670

/-- A quadratic polynomial P(x) = x^2 + bx + c -/
def P (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The theorem stating that if P(x) and P(P(P(x))) share a root, then P(0) * P(1) = 0 -/
theorem quadratic_shared_root_property (b c : ℝ) :
  (∃ r : ℝ, P b c r = 0 ∧ P b c (P b c (P b c r)) = 0) →
  P b c 0 * P b c 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_shared_root_property_l206_20670


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_angle_l206_20693

/-- The equation of a line passing through a given point with a given angle -/
theorem line_equation_through_point_with_angle 
  (x₀ y₀ : ℝ) (θ : ℝ) :
  x₀ = Real.sqrt 3 →
  y₀ = -2 * Real.sqrt 3 →
  θ = 135 * π / 180 →
  ∃ (A B C : ℝ), A * x₀ + B * y₀ + C = 0 ∧
                 A * x + B * y + C = 0 ∧
                 A = 1 ∧ B = 1 ∧ C = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_angle_l206_20693


namespace NUMINAMATH_CALUDE_power_72_equals_m3n2_l206_20687

theorem power_72_equals_m3n2 (a m n : ℝ) (h1 : 2^a = m) (h2 : 3^a = n) : 72^a = m^3 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_72_equals_m3n2_l206_20687


namespace NUMINAMATH_CALUDE_sum_in_range_l206_20634

theorem sum_in_range : ∃ (x : ℚ), 
  (x = 3 + 3/8 + 4 + 1/3 + 6 + 1/21 - 2) ∧ 
  (11.5 < x) ∧ 
  (x < 12) := by
  sorry

end NUMINAMATH_CALUDE_sum_in_range_l206_20634


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l206_20675

theorem rectangular_field_dimensions (m : ℝ) : 
  (3 * m + 8) * (m - 3) = 120 → m = 7 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l206_20675


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l206_20609

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if the slope of one of its asymptotes is 2, then its eccentricity is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 2) : 
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l206_20609


namespace NUMINAMATH_CALUDE_prime_pythagorean_triple_l206_20659

theorem prime_pythagorean_triple (p m n : ℕ) 
  (hp : Nat.Prime p) 
  (hm : m > 0) 
  (hn : n > 0) 
  (heq : p^2 + m^2 = n^2) : 
  m > p := by
  sorry

end NUMINAMATH_CALUDE_prime_pythagorean_triple_l206_20659


namespace NUMINAMATH_CALUDE_class_average_weight_l206_20622

/-- The average weight of a group of children given their total weight and count -/
def average_weight (total_weight : ℚ) (count : ℕ) : ℚ :=
  total_weight / count

/-- The total weight of a group of children given their average weight and count -/
def total_weight (avg_weight : ℚ) (count : ℕ) : ℚ :=
  avg_weight * count

theorem class_average_weight 
  (boys_count : ℕ) 
  (girls_count : ℕ) 
  (boys_avg_weight : ℚ) 
  (girls_avg_weight : ℚ) 
  (h1 : boys_count = 8)
  (h2 : girls_count = 6)
  (h3 : boys_avg_weight = 140)
  (h4 : girls_avg_weight = 130) :
  average_weight 
    (total_weight boys_avg_weight boys_count + total_weight girls_avg_weight girls_count) 
    (boys_count + girls_count) = 135 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l206_20622


namespace NUMINAMATH_CALUDE_stripe_difference_l206_20637

/-- The number of stripes on one of Olga's shoes -/
def olga_stripes_per_shoe : ℕ := 3

/-- The total number of stripes on Olga's shoes -/
def olga_total_stripes : ℕ := 2 * olga_stripes_per_shoe

/-- The total number of stripes on Hortense's shoes -/
def hortense_total_stripes : ℕ := 2 * olga_total_stripes

/-- The total number of stripes on all their shoes -/
def total_stripes : ℕ := 22

/-- The number of stripes on Rick's shoes -/
def rick_total_stripes : ℕ := total_stripes - olga_total_stripes - hortense_total_stripes

theorem stripe_difference : olga_total_stripes - rick_total_stripes = 2 := by
  sorry

end NUMINAMATH_CALUDE_stripe_difference_l206_20637


namespace NUMINAMATH_CALUDE_smallest_m_value_l206_20600

theorem smallest_m_value (m : ℕ) : 
  (∃! quad : (ℕ × ℕ × ℕ × ℕ) → Prop, 
    (∃ (n : ℕ), n = 80000 ∧ 
      (∀ a b c d : ℕ, quad (a, b, c, d) → 
        Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 100 ∧
        Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = m))) →
  m = 2250000 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_value_l206_20600


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l206_20647

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l206_20647


namespace NUMINAMATH_CALUDE_dice_roll_probability_l206_20696

/-- The probability of rolling an even number on a fair 6-sided die -/
def prob_even : ℚ := 1/2

/-- The probability of rolling an odd number less than 3 on a fair 6-sided die -/
def prob_odd_lt_3 : ℚ := 1/6

/-- The number of ways to arrange two even numbers and one odd number -/
def num_arrangements : ℕ := 3

theorem dice_roll_probability :
  num_arrangements * (prob_even^2 * prob_odd_lt_3) = 1/8 := by
sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l206_20696


namespace NUMINAMATH_CALUDE_ricky_sold_nine_l206_20665

/-- Represents the number of glasses of lemonade sold by each person -/
structure LemonadeSales where
  katya : ℕ
  ricky : ℕ
  tina : ℕ

/-- The conditions of the lemonade sales problem -/
def lemonade_problem (sales : LemonadeSales) : Prop :=
  sales.katya = 8 ∧
  sales.tina = 2 * (sales.katya + sales.ricky) ∧
  sales.tina = sales.katya + 26

/-- Theorem stating that under the given conditions, Ricky sold 9 glasses of lemonade -/
theorem ricky_sold_nine (sales : LemonadeSales) 
  (h : lemonade_problem sales) : sales.ricky = 9 := by
  sorry

end NUMINAMATH_CALUDE_ricky_sold_nine_l206_20665


namespace NUMINAMATH_CALUDE_basketball_win_percentage_l206_20691

theorem basketball_win_percentage 
  (games_played : ℕ) 
  (games_won : ℕ) 
  (remaining_games : ℕ) 
  (additional_wins_needed : ℕ) : 
  games_played = 50 →
  games_won = 35 →
  remaining_games = 25 →
  additional_wins_needed = 13 →
  (games_won + additional_wins_needed : ℚ) / (games_played + remaining_games) = 64 / 100 := by
  sorry

end NUMINAMATH_CALUDE_basketball_win_percentage_l206_20691


namespace NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_5_8_2_l206_20641

theorem smallest_three_digit_divisible_by_5_8_2 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 5 ∣ n ∧ 8 ∣ n ∧ 2 ∣ n → n ≥ 120 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_5_8_2_l206_20641


namespace NUMINAMATH_CALUDE_inequality_solution_max_value_expression_max_value_attained_l206_20627

-- Problem 1
theorem inequality_solution (x : ℝ) :
  (|x + 1| + 2 * |x - 1| < 3 * x + 5) ↔ (x > -1/2) :=
sorry

-- Problem 2
theorem max_value_expression (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  a * b + (1 - a - b) * (a + b) ≤ 1/3 :=
sorry

theorem max_value_attained :
  ∃ (a b : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ a * b + (1 - a - b) * (a + b) = 1/3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_max_value_expression_max_value_attained_l206_20627


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l206_20684

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l206_20684


namespace NUMINAMATH_CALUDE_books_returned_percentage_l206_20679

/-- Calculates the percentage of loaned books that were returned -/
def percentage_books_returned (initial_books : ℕ) (final_books : ℕ) (loaned_books : ℕ) : ℚ :=
  let returned_books := final_books - (initial_books - loaned_books)
  (returned_books : ℚ) / (loaned_books : ℚ) * 100

/-- Proves that the percentage of loaned books returned is 70% -/
theorem books_returned_percentage :
  percentage_books_returned 75 60 50 = 70 := by
  sorry

#eval percentage_books_returned 75 60 50

end NUMINAMATH_CALUDE_books_returned_percentage_l206_20679


namespace NUMINAMATH_CALUDE_total_crayons_l206_20628

theorem total_crayons (people : ℕ) (crayons_per_person : ℕ) (h1 : people = 3) (h2 : crayons_per_person = 8) :
  people * crayons_per_person = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l206_20628


namespace NUMINAMATH_CALUDE_product_cde_eq_1000_l206_20624

theorem product_cde_eq_1000 
  (a b c d e f : ℝ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 0.5) :
  c * d * e = 1000 := by
sorry

end NUMINAMATH_CALUDE_product_cde_eq_1000_l206_20624


namespace NUMINAMATH_CALUDE_polynomial_product_no_x_terms_l206_20653

theorem polynomial_product_no_x_terms
  (a b : ℚ)
  (h1 : a ≠ 0)
  (h2 : ∀ x : ℚ, (a * x^2 + b * x + 1) * (3 * x - 2) = 3 * a * x^3 - 2) :
  a = 9/4 ∧ b = 3/2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_no_x_terms_l206_20653


namespace NUMINAMATH_CALUDE_hyperbola_and_asymptotes_l206_20633

/-- Given an ellipse and a hyperbola with the same foci, prove the equation of the hyperbola and its asymptotes -/
theorem hyperbola_and_asymptotes (x y : ℝ) : 
  (∃ (a b c : ℝ), 
    -- Ellipse equation
    (x^2 / 36 + y^2 / 27 = 1) ∧ 
    -- Hyperbola has same foci as ellipse
    (c^2 = a^2 + b^2) ∧ 
    -- Length of conjugate axis of hyperbola
    (2 * b = 4) ∧ 
    -- Foci on x-axis
    (c = 3)) →
  -- Equation of hyperbola
  (x^2 / 5 - y^2 / 4 = 1) ∧
  -- Equations of asymptotes
  (y = (2 * Real.sqrt 5 / 5) * x ∨ y = -(2 * Real.sqrt 5 / 5) * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_and_asymptotes_l206_20633


namespace NUMINAMATH_CALUDE_constant_term_expansion_l206_20671

/-- The constant term in the expansion of (x + 1/x + 1)^4 -/
def constant_term : ℕ := 19

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

theorem constant_term_expansion :
  constant_term = 1 + binomial 4 2 * binomial 2 1 + binomial 4 4 * binomial 4 2 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l206_20671


namespace NUMINAMATH_CALUDE_algebraic_expression_correct_l206_20668

/-- The algebraic expression for "three times x minus the cube of y" -/
def algebraic_expression (x y : ℝ) : ℝ := 3 * x - y^3

/-- Theorem stating that the algebraic expression is correct -/
theorem algebraic_expression_correct (x y : ℝ) : 
  algebraic_expression x y = 3 * x - y^3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_correct_l206_20668


namespace NUMINAMATH_CALUDE_fraction_product_l206_20601

theorem fraction_product : (4/5 : ℚ) * (5/6 : ℚ) * (6/7 : ℚ) * (7/8 : ℚ) * (8/9 : ℚ) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l206_20601


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l206_20608

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 7 ∧ x ≠ -3 →
  (8 * x - 5) / (x^2 - 4 * x - 21) = (51 / 10) / (x - 7) + (29 / 10) / (x + 3) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l206_20608


namespace NUMINAMATH_CALUDE_twelve_sided_die_expected_value_l206_20617

/-- The number of sides on the die -/
def n : ℕ := 12

/-- The expected value of rolling an n-sided die with faces numbered from 1 to n -/
def expected_value (n : ℕ) : ℚ :=
  (n + 1 : ℚ) / 2

/-- Theorem: The expected value of rolling a twelve-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem twelve_sided_die_expected_value :
  expected_value n = 13/2 := by sorry

end NUMINAMATH_CALUDE_twelve_sided_die_expected_value_l206_20617


namespace NUMINAMATH_CALUDE_tree_height_from_shadows_l206_20631

/-- Given a person and a tree casting shadows, calculates the height of the tree -/
theorem tree_height_from_shadows 
  (h s S : ℝ) 
  (h_pos : h > 0) 
  (s_pos : s > 0) 
  (S_pos : S > 0) 
  (h_val : h = 1.5) 
  (s_val : s = 0.5) 
  (S_val : S = 10) : 
  h / s * S = 30 := by
sorry

end NUMINAMATH_CALUDE_tree_height_from_shadows_l206_20631


namespace NUMINAMATH_CALUDE_l_shape_area_l206_20655

/-- The area of an L-shaped figure formed by cutting a smaller rectangle from a larger rectangle -/
theorem l_shape_area (big_length big_width small_length small_width : ℕ) 
  (h1 : big_length = 10)
  (h2 : big_width = 6)
  (h3 : small_length = 4)
  (h4 : small_width = 3) :
  big_length * big_width - small_length * small_width = 48 := by
  sorry

#check l_shape_area

end NUMINAMATH_CALUDE_l_shape_area_l206_20655


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l206_20603

def abe_blue : ℕ := 1
def abe_red : ℕ := 2
def bob_blue : ℕ := 2
def bob_yellow : ℕ := 2
def bob_red : ℕ := 1

def abe_total : ℕ := abe_blue + abe_red
def bob_total : ℕ := bob_blue + bob_yellow + bob_red

theorem jelly_bean_probability : 
  (abe_blue * bob_blue + abe_red * bob_red : ℚ) / (abe_total * bob_total) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l206_20603


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l206_20619

theorem sqrt_expression_equality : 
  Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/6) * Real.sqrt 12 + Real.sqrt 24 = 4 - Real.sqrt 2 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l206_20619


namespace NUMINAMATH_CALUDE_john_racecar_earnings_l206_20648

/-- The amount of money John made from his racecar after one race -/
def money_made (initial_cost maintenance_cost : ℝ) (discount prize_percent : ℝ) (prize : ℝ) : ℝ :=
  prize * prize_percent - initial_cost * (1 - discount) - maintenance_cost

/-- Theorem stating the amount John made from his racecar -/
theorem john_racecar_earnings (x : ℝ) :
  money_made 20000 x 0.2 0.9 70000 = 47000 - x := by
  sorry

end NUMINAMATH_CALUDE_john_racecar_earnings_l206_20648


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l206_20673

theorem triangle_angle_sum (A B C : ℝ) (h1 : A = 90) (h2 : B = 50) : C = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l206_20673


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_cube_l206_20678

def y : ℕ := 2^63^74^95^86^47^5

theorem smallest_multiplier_for_perfect_cube (n : ℕ) :
  (∀ m : ℕ, 0 < m ∧ m < 18 → ¬ ∃ k : ℕ, y * m = k^3) ∧
  ∃ k : ℕ, y * 18 = k^3 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_cube_l206_20678


namespace NUMINAMATH_CALUDE_equal_cost_at_45_l206_20630

/-- Represents the number of students in a class -/
def num_students : ℕ := 45

/-- Represents the original ticket price in yuan -/
def ticket_price : ℕ := 30

/-- Calculates the cost of Option 1 (20% discount for all) -/
def option1_cost (n : ℕ) : ℚ :=
  n * ticket_price * (4 / 5)

/-- Calculates the cost of Option 2 (10% discount and 5 free tickets) -/
def option2_cost (n : ℕ) : ℚ :=
  (n - 5) * ticket_price * (9 / 10)

/-- Theorem stating that for 45 students, the costs of both options are equal -/
theorem equal_cost_at_45 :
  option1_cost num_students = option2_cost num_students :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_at_45_l206_20630


namespace NUMINAMATH_CALUDE_cos_identity_l206_20625

theorem cos_identity : Real.cos (70 * π / 180) + 8 * Real.cos (20 * π / 180) * Real.cos (40 * π / 180) * Real.cos (80 * π / 180) = 2 * (Real.cos (35 * π / 180))^2 := by
  sorry

end NUMINAMATH_CALUDE_cos_identity_l206_20625


namespace NUMINAMATH_CALUDE_sum_f_positive_l206_20674

/-- The function f(x) = x + x³ -/
def f (x : ℝ) : ℝ := x + x^3

/-- Theorem: For any real numbers x₁, x₂, x₃ satisfying the given conditions,
    the sum f(x₁) + f(x₂) + f(x₃) is always positive -/
theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
    (h₁ : x₁ + x₂ > 0) (h₂ : x₂ + x₃ > 0) (h₃ : x₃ + x₁ > 0) : 
    f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l206_20674


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l206_20685

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_x_value :
  let A : Point := ⟨3, -2⟩
  let B : Point := ⟨-9, 4⟩
  let C : Point := ⟨x, 0⟩
  collinear A B C → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_x_value_l206_20685


namespace NUMINAMATH_CALUDE_p_tilde_at_two_l206_20686

def p (x : ℝ) : ℝ := x^2 - 4*x + 3

def p_tilde (x : ℝ) : ℝ := p (p x)

theorem p_tilde_at_two : p_tilde 2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_p_tilde_at_two_l206_20686


namespace NUMINAMATH_CALUDE_soccer_tournament_matches_l206_20656

/-- The number of matches in a round-robin tournament with n teams -/
def numMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of matches between two groups of teams -/
def numMatchesBetweenGroups (n m : ℕ) : ℕ := n * m

theorem soccer_tournament_matches :
  (numMatches 3 = 3) ∧
  (numMatches 4 = 6) ∧
  (numMatchesBetweenGroups 3 4 = 12) := by
  sorry

#eval numMatches 3  -- Expected output: 3
#eval numMatches 4  -- Expected output: 6
#eval numMatchesBetweenGroups 3 4  -- Expected output: 12

end NUMINAMATH_CALUDE_soccer_tournament_matches_l206_20656


namespace NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l206_20650

theorem largest_coefficient_binomial_expansion :
  ∀ n : ℕ, 
    n ≤ 11 → 
    (Nat.choose 11 n : ℚ) ≤ (Nat.choose 11 6 : ℚ) ∧
    (Nat.choose 11 6 : ℚ) = (Nat.choose 11 5 : ℚ) ∧
    (∀ k : ℕ, k < 5 → (Nat.choose 11 k : ℚ) < (Nat.choose 11 6 : ℚ)) :=
by
  sorry

#check largest_coefficient_binomial_expansion

end NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l206_20650


namespace NUMINAMATH_CALUDE_total_slices_today_l206_20610

def lunch_slices : ℕ := 7
def dinner_slices : ℕ := 5

theorem total_slices_today : lunch_slices + dinner_slices = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_today_l206_20610


namespace NUMINAMATH_CALUDE_locus_of_midpoint_is_circle_l206_20623

/-- Given a circle with center O and radius R, and a point P inside the circle,
    we rotate a right angle around P. The legs of the right angle intersect
    the circle at points A and B. This theorem proves that the locus of the
    midpoint of chord AB is a circle. -/
theorem locus_of_midpoint_is_circle
  (O : ℝ × ℝ)  -- Center of the circle
  (R : ℝ)      -- Radius of the circle
  (P : ℝ × ℝ)  -- Point inside the circle
  (h_R_pos : R > 0)  -- R is positive
  (h_P_inside : dist P O < R)  -- P is inside the circle
  (A B : ℝ × ℝ)  -- Points on the circle
  (h_A_on_circle : dist A O = R)  -- A is on the circle
  (h_B_on_circle : dist B O = R)  -- B is on the circle
  (h_right_angle : (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0)  -- ∠APB is a right angle
  : ∃ (C : ℝ × ℝ) (r : ℝ),
    let a := dist P O
    let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)  -- Midpoint of AB
    C = (a / 2, 0) ∧ r = (1 / 2) * Real.sqrt (2 * R^2 - a^2) ∧
    dist M C = r :=
by sorry

end NUMINAMATH_CALUDE_locus_of_midpoint_is_circle_l206_20623


namespace NUMINAMATH_CALUDE_car_cost_sharing_l206_20666

theorem car_cost_sharing
  (total_cost : ℕ)
  (car_wash_funds : ℕ)
  (initial_friends : ℕ)
  (dropouts : ℕ)
  (h1 : total_cost = 1700)
  (h2 : car_wash_funds = 500)
  (h3 : initial_friends = 6)
  (h4 : dropouts = 1) :
  (total_cost - car_wash_funds) / (initial_friends - dropouts) -
  (total_cost - car_wash_funds) / initial_friends = 40 :=
by sorry

end NUMINAMATH_CALUDE_car_cost_sharing_l206_20666


namespace NUMINAMATH_CALUDE_enclosed_area_of_special_curve_l206_20645

/-- The area enclosed by a curve consisting of 9 congruent circular arcs, 
    each of length 2π/3, with centers on the vertices of a regular hexagon 
    with side length 3, is equal to 13.5√3 + π. -/
theorem enclosed_area_of_special_curve (
  n : ℕ) (arc_length : ℝ) (hexagon_side : ℝ) (enclosed_area : ℝ) : 
  n = 9 → 
  arc_length = 2 * Real.pi / 3 → 
  hexagon_side = 3 → 
  enclosed_area = 13.5 * Real.sqrt 3 + Real.pi → 
  enclosed_area = 
    (3 * Real.sqrt 3 / 2 * hexagon_side^2) + (n * arc_length * (arc_length / (2 * Real.pi))) :=
by sorry

end NUMINAMATH_CALUDE_enclosed_area_of_special_curve_l206_20645


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l206_20669

theorem line_passes_through_fixed_point :
  ∀ (a : ℝ), (3 * a - 1 + 1 - 3 * a = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l206_20669


namespace NUMINAMATH_CALUDE_problem_statement_l206_20683

theorem problem_statement (x y z k : ℝ) 
  (h1 : x + 1/y = k)
  (h2 : 2*y + 2/z = k)
  (h3 : 3*z + 3/x = k)
  (h4 : x*y*z = 3) :
  k = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l206_20683


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l206_20649

theorem unique_solution_cube_equation : 
  ∃! (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 ∧ x = 27 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l206_20649


namespace NUMINAMATH_CALUDE_equal_edge_length_relation_l206_20611

/-- Represents a hexagonal prism -/
structure HexagonalPrism :=
  (edge_length : ℝ)
  (total_edge_length : ℝ)
  (h_total : total_edge_length = 18 * edge_length)

/-- Represents a quadrangular pyramid -/
structure QuadrangularPyramid :=
  (edge_length : ℝ)
  (total_edge_length : ℝ)
  (h_total : total_edge_length = 8 * edge_length)

/-- 
Given a hexagonal prism and a quadrangular pyramid with equal edge lengths,
if the total edge length of the hexagonal prism is 81 cm,
then the total edge length of the quadrangular pyramid is 36 cm.
-/
theorem equal_edge_length_relation 
  (prism : HexagonalPrism) 
  (pyramid : QuadrangularPyramid) 
  (h_equal_edges : prism.edge_length = pyramid.edge_length) 
  (h_prism_total : prism.total_edge_length = 81) : 
  pyramid.total_edge_length = 36 := by
  sorry

end NUMINAMATH_CALUDE_equal_edge_length_relation_l206_20611


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l206_20663

theorem min_value_of_sum_of_squares (x y : ℝ) (h : 2 * (x^2 + y^2) = x^2 + y + x*y) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b : ℝ), 2 * (a^2 + b^2) = a^2 + b + a*b → x^2 + y^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l206_20663


namespace NUMINAMATH_CALUDE_remainder_theorem_l206_20661

theorem remainder_theorem (r : ℝ) : (r^13 + 1) % (r - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l206_20661


namespace NUMINAMATH_CALUDE_noodles_left_proof_l206_20698

-- Define the initial number of noodles
def initial_noodles : Float := 54.0

-- Define the number of noodles given away
def noodles_given : Float := 12.0

-- Theorem to prove
theorem noodles_left_proof : initial_noodles - noodles_given = 42.0 := by
  sorry

end NUMINAMATH_CALUDE_noodles_left_proof_l206_20698


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l206_20639

theorem largest_integer_less_than_100_remainder_5_mod_8 :
  ∀ n : ℕ, n < 100 → n % 8 = 5 → n ≤ 99 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l206_20639


namespace NUMINAMATH_CALUDE_equation_solution_l206_20618

theorem equation_solution : 
  ∃ x : ℚ, x ≠ -2 ∧ (x^2 + 3*x + 4) / (x + 2) = x + 6 :=
by
  use -8/5
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l206_20618


namespace NUMINAMATH_CALUDE_digit_sum_inequalities_l206_20620

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

theorem digit_sum_inequalities :
  (∀ k : ℕ, sumOfDigits k ≤ 8 * sumOfDigits (8 * k)) ∧
  (∀ N : ℕ, sumOfDigits N ≤ 5 * sumOfDigits (5^5 * N)) := by sorry

end NUMINAMATH_CALUDE_digit_sum_inequalities_l206_20620


namespace NUMINAMATH_CALUDE_part_one_part_two_l206_20644

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 2*m + 1}
def B : Set ℝ := {x | (x - 7) / (x - 2) < 0}

-- Part 1
theorem part_one : A 2 ∩ (Set.univ \ B) = Set.Ioc 1 2 := by sorry

-- Part 2
theorem part_two : ∀ m : ℝ, A m ∪ B = B ↔ m ∈ Set.Iic (-2) ∪ {3} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l206_20644


namespace NUMINAMATH_CALUDE_leg_ratio_is_sqrt_seven_l206_20614

/-- Configuration of squares and triangles -/
structure SquareTriangleConfig where
  /-- Side length of the inner square -/
  s : ℝ
  /-- Length of the shorter leg of each triangle -/
  a : ℝ
  /-- Length of the longer leg of each triangle -/
  b : ℝ
  /-- Side length of the outer square -/
  t : ℝ
  /-- The triangles are right triangles -/
  triangle_right : a^2 + b^2 = t^2
  /-- The area of the outer square is twice the area of the inner square -/
  area_relation : t^2 = 2 * s^2
  /-- The shorter legs of two triangles form one side of the inner square -/
  inner_side : 2 * a = s

/-- The ratio of the longer leg to the shorter leg is √7 -/
theorem leg_ratio_is_sqrt_seven (config : SquareTriangleConfig) :
  config.b / config.a = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_leg_ratio_is_sqrt_seven_l206_20614


namespace NUMINAMATH_CALUDE_other_number_proof_l206_20689

theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 16) (h2 : Nat.lcm a b = 396) (h3 : a = 176) : b = 36 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l206_20689


namespace NUMINAMATH_CALUDE_tangent_problem_l206_20643

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  (1 + Real.tan α) / (1 - Real.tan α) = 3 / 22 := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l206_20643


namespace NUMINAMATH_CALUDE_x_value_l206_20615

theorem x_value : ∀ x : ℕ, x = 225 + 2 * 15 * 9 + 81 → x = 576 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l206_20615


namespace NUMINAMATH_CALUDE_ant_meeting_point_l206_20612

/-- Triangle XYZ with given side lengths -/
structure Triangle where
  xy : ℝ
  yz : ℝ
  xz : ℝ

/-- Point P where ants meet -/
def MeetingPoint (t : Triangle) : ℝ := sorry

/-- Theorem stating that YP = 5 in the given triangle -/
theorem ant_meeting_point (t : Triangle) 
  (h_xy : t.xy = 5) 
  (h_yz : t.yz = 7) 
  (h_xz : t.xz = 8) : 
  MeetingPoint t = 5 := by sorry

end NUMINAMATH_CALUDE_ant_meeting_point_l206_20612


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l206_20688

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 3*x - 4 ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l206_20688


namespace NUMINAMATH_CALUDE_f_not_odd_nor_even_f_minimum_value_l206_20658

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + |x - 2| - 1

-- Theorem for the parity of f(x)
theorem f_not_odd_nor_even :
  ¬(∀ x, f x = f (-x)) ∧ ¬(∀ x, f x = -f (-x)) :=
sorry

-- Theorem for the minimum value of f(x)
theorem f_minimum_value :
  ∀ x, f x ≥ 3 ∧ ∃ y, f y = 3 :=
sorry

end NUMINAMATH_CALUDE_f_not_odd_nor_even_f_minimum_value_l206_20658


namespace NUMINAMATH_CALUDE_more_birch_than_fir_l206_20692

/-- Represents a forest with fir and birch trees -/
structure Forest where
  fir_trees : ℕ
  birch_trees : ℕ

/-- A forest satisfies the Baron's condition if each fir tree has exactly 10 birch trees at 1 km distance -/
def satisfies_baron_condition (f : Forest) : Prop :=
  f.birch_trees = 10 * f.fir_trees

/-- Theorem: In a forest satisfying the Baron's condition, there are more birch trees than fir trees -/
theorem more_birch_than_fir (f : Forest) (h : satisfies_baron_condition f) : 
  f.birch_trees > f.fir_trees :=
sorry


end NUMINAMATH_CALUDE_more_birch_than_fir_l206_20692


namespace NUMINAMATH_CALUDE_perfect_square_difference_l206_20651

theorem perfect_square_difference (x y : ℕ) (h : x > 0 ∧ y > 0) 
  (eq : 3 * x^2 + x = 4 * y^2 + y) : 
  ∃ (k : ℕ), x - y = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_difference_l206_20651


namespace NUMINAMATH_CALUDE_leak_emptying_time_l206_20697

/-- Proves that given a tank with a leak, if Pipe A can fill the tank in 5 hours without the leak
    and in 10 hours with the leak, then the leak alone can empty the full tank in 10 hours. -/
theorem leak_emptying_time (fill_rate_no_leak fill_rate_with_leak leak_rate : ℝ) :
  fill_rate_no_leak = 1 / 5 →
  fill_rate_with_leak = 1 / 10 →
  fill_rate_no_leak - leak_rate = fill_rate_with_leak →
  (1 : ℝ) / leak_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_leak_emptying_time_l206_20697


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l206_20672

theorem floor_abs_negative_real : ⌊|(-58.7 : ℝ)|⌋ = 58 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l206_20672


namespace NUMINAMATH_CALUDE_correct_average_points_l206_20636

/-- Represents Melissa's basketball season statistics -/
structure BasketballSeason where
  totalGames : ℕ
  totalPoints : ℕ
  wonGames : ℕ
  averagePointDifference : ℕ

/-- Calculates the average points scored in won and lost games -/
def calculateAveragePoints (season : BasketballSeason) : ℕ × ℕ :=
  sorry

/-- Theorem stating the correct average points for won and lost games -/
theorem correct_average_points (season : BasketballSeason) 
  (h1 : season.totalGames = 20)
  (h2 : season.totalPoints = 400)
  (h3 : season.wonGames = 8)
  (h4 : season.averagePointDifference = 15) :
  calculateAveragePoints season = (29, 14) := by
  sorry

end NUMINAMATH_CALUDE_correct_average_points_l206_20636


namespace NUMINAMATH_CALUDE_product_of_fractions_l206_20646

theorem product_of_fractions : 
  let f (n : ℕ) := (n^4 - 1) / (n^4 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 880 / 91 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l206_20646


namespace NUMINAMATH_CALUDE_product_increased_equals_nineteen_l206_20662

theorem product_increased_equals_nineteen (x : ℝ) : 5 * x + 4 = 19 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_increased_equals_nineteen_l206_20662


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_square_l206_20681

/-- Given a square with side length a, there exists an isosceles triangle with the specified properties --/
theorem isosceles_triangle_from_square (a : ℝ) (h : a > 0) :
  ∃ (x y z : ℝ),
    -- The base of the triangle
    x = a * Real.sqrt 3 ∧
    -- The height of the triangle
    y = (2 * x) / 3 ∧
    -- The equal sides of the triangle
    z = (5 * a * Real.sqrt 3) / 6 ∧
    -- Area equality
    (1 / 2) * x * y = a^2 ∧
    -- Sum of base and height equals sum of equal sides
    x + y = 2 * z ∧
    -- Pythagorean theorem
    y^2 + (x / 2)^2 = z^2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_square_l206_20681


namespace NUMINAMATH_CALUDE_initials_count_l206_20699

/-- The number of letters available for initials -/
def num_letters : ℕ := 10

/-- The number of initials in each set -/
def initials_per_set : ℕ := 3

/-- The total number of possible three-letter sets of initials -/
def total_sets : ℕ := num_letters ^ initials_per_set

theorem initials_count : total_sets = 1000 := by
  sorry

end NUMINAMATH_CALUDE_initials_count_l206_20699


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l206_20664

/-- Given a line 2ax - by + 2 = 0 (where a > 0, b > 0) passing through the point (-1, 2),
    the minimum value of 1/a + 1/b is 4. -/
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : 2 * a * (-1) - b * 2 + 2 = 0) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x * (-1) - y * 2 + 2 = 0 → 1 / x + 1 / y ≥ 1 / a + 1 / b) ∧
  1 / a + 1 / b = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l206_20664


namespace NUMINAMATH_CALUDE_february_highest_percentage_difference_l206_20654

/-- Represents the sales data for a vendor in a given month -/
structure SalesData where
  quantity : Nat
  price : Float

/-- Calculates the revenue from sales data -/
def revenue (data : SalesData) : Float :=
  data.quantity.toFloat * data.price

/-- Calculates the percentage difference between two revenues -/
def percentageDifference (r1 r2 : Float) : Float :=
  (max r1 r2 - min r1 r2) / (min r1 r2) * 100

/-- Represents a month -/
inductive Month
  | January | February | March | April | May | June

/-- Andy's sales data for each month -/
def andySales : Month → SalesData
  | .January => ⟨100, 2⟩
  | .February => ⟨150, 1.5⟩
  | .March => ⟨120, 2.5⟩
  | .April => ⟨80, 4⟩
  | .May => ⟨140, 1.75⟩
  | .June => ⟨110, 3⟩

/-- Bella's sales data for each month -/
def bellaSales : Month → SalesData
  | .January => ⟨90, 2.2⟩
  | .February => ⟨100, 1.75⟩
  | .March => ⟨80, 3⟩
  | .April => ⟨85, 3.5⟩
  | .May => ⟨135, 2⟩
  | .June => ⟨160, 2.5⟩

/-- Theorem: February has the highest percentage difference in revenue -/
theorem february_highest_percentage_difference :
  ∀ m : Month, m ≠ Month.February →
    percentageDifference (revenue (andySales Month.February)) (revenue (bellaSales Month.February)) ≥
    percentageDifference (revenue (andySales m)) (revenue (bellaSales m)) :=
by sorry

end NUMINAMATH_CALUDE_february_highest_percentage_difference_l206_20654


namespace NUMINAMATH_CALUDE_intersection_M_N_l206_20695

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l206_20695


namespace NUMINAMATH_CALUDE_gcd_b_consecutive_is_one_l206_20606

def b (n : ℕ) : ℤ := (7^n - 1) / 6

theorem gcd_b_consecutive_is_one (n : ℕ) : 
  Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1))) = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_b_consecutive_is_one_l206_20606


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l206_20635

/-- The number of trailing zeros in the product of all multiples of 5 from 5 to 2015 -/
def trailingZeros : ℕ :=
  let n := 2015 / 5  -- number of terms in the product
  let factorsOf2 := (n / 2) + (n / 4) + (n / 8) + (n / 16) + (n / 32) + (n / 64) + (n / 128) + (n / 256)
  factorsOf2

theorem product_trailing_zeros : trailingZeros = 398 := by
  sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l206_20635


namespace NUMINAMATH_CALUDE_magnified_tissue_diameter_l206_20694

/-- Given a circular piece of tissue and an electron microscope, 
    calculates the diameter of the magnified image. -/
def magnified_diameter (actual_diameter : ℝ) (magnification_factor : ℝ) : ℝ :=
  actual_diameter * magnification_factor

/-- Theorem stating that for the given conditions, 
    the magnified diameter is 2 centimeters. -/
theorem magnified_tissue_diameter :
  let actual_diameter : ℝ := 0.002
  let magnification_factor : ℝ := 1000
  magnified_diameter actual_diameter magnification_factor = 2 := by
  sorry

end NUMINAMATH_CALUDE_magnified_tissue_diameter_l206_20694


namespace NUMINAMATH_CALUDE_inequality_implication_l206_20676

theorem inequality_implication (x y : ℝ) (h : x < y) : -x + 3 > -y + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l206_20676


namespace NUMINAMATH_CALUDE_power_seven_137_mod_nine_l206_20626

theorem power_seven_137_mod_nine : 7^137 % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_137_mod_nine_l206_20626


namespace NUMINAMATH_CALUDE_johnny_practice_days_l206_20657

/-- The number of days Johnny has been practicing up to now -/
def current_practice_days : ℕ := 40

/-- The number of days in the future when Johnny will have tripled his practice -/
def future_days : ℕ := 80

/-- Represents that Johnny practices the same amount each day -/
axiom consistent_practice : True

/-- In 80 days, Johnny will have 3 times as much practice as he does currently -/
axiom future_practice : current_practice_days + future_days = 3 * current_practice_days

/-- The number of days ago when Johnny had half as much practice -/
def half_practice_days : ℕ := current_practice_days / 2

theorem johnny_practice_days : half_practice_days = 20 := by sorry

end NUMINAMATH_CALUDE_johnny_practice_days_l206_20657


namespace NUMINAMATH_CALUDE_max_volume_at_eight_l206_20680

/-- The side length of the original square plate in cm -/
def plate_side : ℝ := 48

/-- The volume of the container as a function of the cut square's side length -/
def volume (x : ℝ) : ℝ := (plate_side - 2*x)^2 * x

/-- The derivative of the volume function -/
def volume_derivative (x : ℝ) : ℝ := (plate_side - 2*x) * (plate_side - 6*x)

theorem max_volume_at_eight :
  ∃ (max_x : ℝ), max_x = 8 ∧
  ∀ (x : ℝ), 0 < x ∧ x < plate_side / 2 → volume x ≤ volume max_x :=
sorry

end NUMINAMATH_CALUDE_max_volume_at_eight_l206_20680


namespace NUMINAMATH_CALUDE_coin_value_difference_l206_20660

-- Define the coin types
inductive Coin
| Penny
| Nickel
| Dime

-- Define the function to calculate the value of a coin in cents
def coinValue : Coin → Nat
| Coin.Penny => 1
| Coin.Nickel => 5
| Coin.Dime => 10

-- Define the total number of coins
def totalCoins : Nat := 3000

-- Define the theorem
theorem coin_value_difference :
  ∃ (p n d : Nat),
    p + n + d = totalCoins ∧
    p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧
    (∀ (p' n' d' : Nat),
      p' + n' + d' = totalCoins →
      p' ≥ 1 → n' ≥ 1 → d' ≥ 1 →
      coinValue Coin.Penny * p' + coinValue Coin.Nickel * n' + coinValue Coin.Dime * d' ≤
      coinValue Coin.Penny * p + coinValue Coin.Nickel * n + coinValue Coin.Dime * d) ∧
    (∀ (p' n' d' : Nat),
      p' + n' + d' = totalCoins →
      p' ≥ 1 → n' ≥ 1 → d' ≥ 1 →
      coinValue Coin.Penny * p + coinValue Coin.Nickel * n + coinValue Coin.Dime * d -
      (coinValue Coin.Penny * p' + coinValue Coin.Nickel * n' + coinValue Coin.Dime * d') = 26973) :=
by sorry


end NUMINAMATH_CALUDE_coin_value_difference_l206_20660


namespace NUMINAMATH_CALUDE_only_setA_is_pythagorean_triple_l206_20602

/-- A function to check if a triple of integers forms a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℤ) : Prop :=
  a * a + b * b = c * c

/-- The given sets of numbers -/
def setA : List ℤ := [5, 12, 13]
def setB : List ℤ := [7, 9, 11]
def setC : List ℤ := [6, 9, 12]
def setD : List ℚ := [3/10, 4/10, 5/10]

/-- Theorem stating that only setA is a Pythagorean triple -/
theorem only_setA_is_pythagorean_triple :
  (isPythagoreanTriple setA[0]! setA[1]! setA[2]!) ∧
  (¬ isPythagoreanTriple setB[0]! setB[1]! setB[2]!) ∧
  (¬ isPythagoreanTriple setC[0]! setC[1]! setC[2]!) ∧
  (∀ (a b c : ℚ), a ∈ setD → b ∈ setD → c ∈ setD → ¬ isPythagoreanTriple a.num b.num c.num) :=
by sorry


end NUMINAMATH_CALUDE_only_setA_is_pythagorean_triple_l206_20602


namespace NUMINAMATH_CALUDE_smaller_number_l206_20690

theorem smaller_number (x y : ℝ) (sum_eq : x + y = 18) (diff_eq : x - y = 8) : 
  min x y = 5 := by sorry

end NUMINAMATH_CALUDE_smaller_number_l206_20690


namespace NUMINAMATH_CALUDE_jace_travel_distance_l206_20677

/-- Calculates the total distance traveled given a constant speed and two driving periods -/
def total_distance (speed : ℝ) (time1 : ℝ) (time2 : ℝ) : ℝ :=
  speed * (time1 + time2)

/-- Theorem stating that given the specified conditions, the total distance traveled is 780 miles -/
theorem jace_travel_distance :
  let speed : ℝ := 60
  let time1 : ℝ := 4
  let time2 : ℝ := 9
  total_distance speed time1 time2 = 780 := by
  sorry

end NUMINAMATH_CALUDE_jace_travel_distance_l206_20677


namespace NUMINAMATH_CALUDE_expression_evaluation_l206_20613

theorem expression_evaluation : 6^2 - 4*5 + 2^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l206_20613


namespace NUMINAMATH_CALUDE_tan_and_cos_relations_l206_20616

theorem tan_and_cos_relations (θ : Real) (h : Real.tan θ = 2) :
  Real.tan (π / 4 - θ) = -1 / 3 ∧ Real.cos (2 * θ) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_and_cos_relations_l206_20616


namespace NUMINAMATH_CALUDE_fraction_problem_l206_20682

theorem fraction_problem (a b c : ℝ) 
  (h1 : a * b / (a + b) = 3)
  (h2 : b * c / (b + c) = 6)
  (h3 : a * c / (a + c) = 9) :
  c / (a * b) = -35 / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l206_20682
