import Mathlib

namespace NUMINAMATH_CALUDE_bread_theorem_l2897_289764

def bread_problem (slices_per_loaf : ℕ) (num_friends : ℕ) (num_loaves : ℕ) : ℕ :=
  (slices_per_loaf * num_loaves) / num_friends

theorem bread_theorem :
  bread_problem 15 10 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bread_theorem_l2897_289764


namespace NUMINAMATH_CALUDE_angelina_speed_l2897_289744

/-- Angelina's walk from home to grocery to gym -/
def angelina_walk (v : ℝ) : Prop :=
  let home_to_grocery_distance : ℝ := 180
  let grocery_to_gym_distance : ℝ := 240
  let home_to_grocery_time : ℝ := home_to_grocery_distance / v
  let grocery_to_gym_time : ℝ := grocery_to_gym_distance / (2 * v)
  home_to_grocery_time = grocery_to_gym_time + 40

theorem angelina_speed : ∃ v : ℝ, angelina_walk v ∧ 2 * v = 3 := by sorry

end NUMINAMATH_CALUDE_angelina_speed_l2897_289744


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_equivalence_l2897_289731

theorem binomial_coefficient_divisibility_equivalence 
  (n : ℕ) (p : ℕ) (h_prime : Prime p) : 
  (∀ k : ℕ, k ≤ n → ¬(p ∣ Nat.choose n k)) ↔ 
  (∃ s m : ℕ, s > 0 ∧ m < p ∧ n = p^s * m - 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_equivalence_l2897_289731


namespace NUMINAMATH_CALUDE_investment_problem_l2897_289792

theorem investment_problem (total_interest : ℝ) (amount_at_11_percent : ℝ) :
  total_interest = 0.0975 →
  amount_at_11_percent = 3750 →
  ∃ (total_amount : ℝ) (amount_at_9_percent : ℝ),
    total_amount = amount_at_9_percent + amount_at_11_percent ∧
    0.09 * amount_at_9_percent + 0.11 * amount_at_11_percent = total_interest * total_amount ∧
    total_amount = 10000 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l2897_289792


namespace NUMINAMATH_CALUDE_product_remainder_by_10_l2897_289773

theorem product_remainder_by_10 : (8623 * 2475 * 56248 * 1234) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_10_l2897_289773


namespace NUMINAMATH_CALUDE_problem_solution_l2897_289715

theorem problem_solution : (3 - Real.pi) ^ 0 - 3 ^ (-1 : ℤ) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2897_289715


namespace NUMINAMATH_CALUDE_point_above_line_l2897_289753

/-- A point is above a line if its y-coordinate is greater than the y-coordinate of the point on the line with the same x-coordinate. -/
def IsAboveLine (x y : ℝ) (a b c : ℝ) : Prop :=
  y > (a * x + c) / b

/-- The theorem states that for a point P(-2, t) to be above the line 2x - 3y + 6 = 0, t must be greater than 2/3. -/
theorem point_above_line (t : ℝ) :
  IsAboveLine (-2) t 2 (-3) 6 ↔ t > 2/3 := by
  sorry

#check point_above_line

end NUMINAMATH_CALUDE_point_above_line_l2897_289753


namespace NUMINAMATH_CALUDE_hyperbola_intersecting_line_l2897_289795

/-- Given a hyperbola and an ellipse with specific properties, prove the equation of a line intersecting the hyperbola. -/
theorem hyperbola_intersecting_line 
  (a : ℝ) 
  (h_a_pos : a > 0)
  (C : Set (ℝ × ℝ)) 
  (h_C : C = {(x, y) | x^2/a^2 - y^2/4 = 1})
  (E : Set (ℝ × ℝ))
  (h_E : E = {(x, y) | x^2/16 + y^2/8 = 1})
  (h_foci : {(-4, 0), (4, 0)} ⊆ C)
  (A B : ℝ × ℝ)
  (h_AB : A ∈ C ∧ B ∈ C)
  (h_midpoint : (A.1 + B.1)/2 = 6 ∧ (A.2 + B.2)/2 = 1) :
  ∃ (k m : ℝ), k * A.1 + m * A.2 = 1 ∧ k * B.1 + m * B.2 = 1 ∧ k = 2 ∧ m = -1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersecting_line_l2897_289795


namespace NUMINAMATH_CALUDE_number_ratio_problem_l2897_289746

theorem number_ratio_problem (N : ℝ) : 
  (1/3 : ℝ) * (2/5 : ℝ) * N = 15 ∧ (40/100 : ℝ) * N = 180 → 
  15 / N = 1 / 7.5 :=
by sorry

end NUMINAMATH_CALUDE_number_ratio_problem_l2897_289746


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2897_289712

theorem negative_fraction_comparison : -3/5 > -5/7 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2897_289712


namespace NUMINAMATH_CALUDE_two_points_same_color_distance_l2897_289749

-- Define a type for colors
inductive Color
| Yellow
| Red

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Theorem statement
theorem two_points_same_color_distance (x : ℝ) (h : x > 0) :
  ∃ (c : Color) (p1 p2 : Point), coloring p1 = c ∧ coloring p2 = c ∧ distance p1 p2 = x := by
  sorry

end NUMINAMATH_CALUDE_two_points_same_color_distance_l2897_289749


namespace NUMINAMATH_CALUDE_value_of_c_l2897_289752

theorem value_of_c : ∃ c : ℝ, 
  (∀ x : ℝ, x * (4 * x + 2) < c ↔ -5/2 < x ∧ x < 3) ∧ c = 45 := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l2897_289752


namespace NUMINAMATH_CALUDE_div_remainder_theorem_l2897_289793

theorem div_remainder_theorem : 
  ∃ k : ℕ, 3^19 = k * 1162261460 + 7 :=
sorry

end NUMINAMATH_CALUDE_div_remainder_theorem_l2897_289793


namespace NUMINAMATH_CALUDE_additional_peaches_l2897_289797

theorem additional_peaches (initial_peaches total_peaches : ℕ) 
  (h1 : initial_peaches = 20)
  (h2 : total_peaches = 45) :
  total_peaches - initial_peaches = 25 := by
  sorry

end NUMINAMATH_CALUDE_additional_peaches_l2897_289797


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2897_289780

theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
    sheep * 7 = horses * 5 →
    horses * 230 = 12880 →
    sheep = 40 := by
sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2897_289780


namespace NUMINAMATH_CALUDE_unit_vectors_equal_squared_magnitude_l2897_289734

/-- Two unit vectors in a plane have equal squared magnitudes. -/
theorem unit_vectors_equal_squared_magnitude
  (e₁ e₂ : ℝ × ℝ)
  (h₁ : ‖e₁‖ = 1)
  (h₂ : ‖e₂‖ = 1) :
  ‖e₁‖^2 = ‖e₂‖^2 := by
  sorry

end NUMINAMATH_CALUDE_unit_vectors_equal_squared_magnitude_l2897_289734


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2897_289791

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2897_289791


namespace NUMINAMATH_CALUDE_problem_statement_l2897_289738

theorem problem_statement (a c : ℤ) : 
  (∃ (x : ℤ), x^2 = 2*a - 1 ∧ (x = 3 ∨ x = -3)) → 
  c = ⌊Real.sqrt 17⌋ → 
  a + c = 9 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2897_289738


namespace NUMINAMATH_CALUDE_nanning_gdp_scientific_notation_l2897_289723

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem nanning_gdp_scientific_notation :
  let gdp : ℝ := 1060 * 10^9  -- 1060 billion
  let scientific_form := toScientificNotation gdp
  scientific_form.coefficient = 1.06 ∧ scientific_form.exponent = 11 :=
by sorry

end NUMINAMATH_CALUDE_nanning_gdp_scientific_notation_l2897_289723


namespace NUMINAMATH_CALUDE_average_page_count_l2897_289772

theorem average_page_count (n : ℕ) (g1 g2 g3 : ℕ) (p1 p2 p3 : ℕ) :
  n = g1 + g2 + g3 →
  g1 = g2 ∧ g2 = g3 →
  (g1 * p1 + g2 * p2 + g3 * p3) / n = 2 →
  n = 15 ∧ g1 = 5 ∧ p1 = 2 ∧ p2 = 3 ∧ p3 = 1 →
  (g1 * p1 + g2 * p2 + g3 * p3) / n = 2 :=
by sorry

end NUMINAMATH_CALUDE_average_page_count_l2897_289772


namespace NUMINAMATH_CALUDE_power_sum_seven_l2897_289720

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^7 + b^7 = 29 -/
theorem power_sum_seven (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (h6 : ∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^7 + b^7 = 29 := by
sorry

end NUMINAMATH_CALUDE_power_sum_seven_l2897_289720


namespace NUMINAMATH_CALUDE_time_to_finish_problems_l2897_289783

/-- The time required to finish all problems given the number of math and spelling problems and the rate of problem-solving. -/
theorem time_to_finish_problems
  (math_problems : ℕ)
  (spelling_problems : ℕ)
  (problems_per_hour : ℕ)
  (h1 : math_problems = 18)
  (h2 : spelling_problems = 6)
  (h3 : problems_per_hour = 4) :
  (math_problems + spelling_problems) / problems_per_hour = 6 :=
by sorry

end NUMINAMATH_CALUDE_time_to_finish_problems_l2897_289783


namespace NUMINAMATH_CALUDE_loan_duration_C_l2897_289736

/-- Calculates simple interest -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem loan_duration_C (principal_B principal_C total_interest : ℚ) 
  (time_B : ℚ) (rate : ℚ) :
  principal_B = 4000 →
  principal_C = 2000 →
  time_B = 2 →
  rate = 13.75 →
  total_interest = 2200 →
  simpleInterest principal_B rate time_B + simpleInterest principal_C rate (4 : ℚ) = total_interest :=
by sorry

end NUMINAMATH_CALUDE_loan_duration_C_l2897_289736


namespace NUMINAMATH_CALUDE_solution_system_trigonometric_equations_l2897_289702

theorem solution_system_trigonometric_equations :
  ∀ x y : ℝ,
  (Real.sin x)^2 = Real.sin y ∧ (Real.cos x)^4 = Real.cos y →
  (∃ l m : ℤ, x = l * Real.pi ∧ y = 2 * m * Real.pi) ∨
  (∃ l m : ℤ, x = l * Real.pi + Real.pi / 2 ∧ y = 2 * m * Real.pi + Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_system_trigonometric_equations_l2897_289702


namespace NUMINAMATH_CALUDE_combined_height_l2897_289737

theorem combined_height (kirill_height brother_height : ℕ) : 
  kirill_height = 49 →
  brother_height = kirill_height + 14 →
  kirill_height + brother_height = 112 := by
sorry

end NUMINAMATH_CALUDE_combined_height_l2897_289737


namespace NUMINAMATH_CALUDE_triangle_max_area_l2897_289784

theorem triangle_max_area (a b c : ℝ) (h1 : a = 75) (h2 : c = 2 * b) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (∀ x : ℝ, x > 0 → area ≤ 1100) ∧ (∃ x : ℝ, x > 0 ∧ area = 1100) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2897_289784


namespace NUMINAMATH_CALUDE_circumradius_inradius_ratio_rational_l2897_289771

/-- Given a triangle with rational side lengths, prove that the ratio of its circumradius to inradius is rational. -/
theorem circumradius_inradius_ratio_rational 
  (a b c : ℚ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let p : ℚ := (a + b + c) / 2
  ∃ (q : ℚ), q = a * b * c / (4 * (p - a) * (p - b) * (p - c)) :=
sorry

end NUMINAMATH_CALUDE_circumradius_inradius_ratio_rational_l2897_289771


namespace NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l2897_289728

theorem largest_number_with_equal_quotient_and_remainder (A B C : ℕ) 
  (h1 : A = 8 * B + C) 
  (h2 : B = C) 
  (h3 : C < 8) : 
  A ≤ 63 ∧ ∃ (A' : ℕ), A' = 63 ∧ ∃ (B' C' : ℕ), A' = 8 * B' + C' ∧ B' = C' ∧ C' < 8 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l2897_289728


namespace NUMINAMATH_CALUDE_f_always_negative_iff_a_in_range_l2897_289775

/-- A quadratic function f(x) = ax^2 + ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 1

/-- The property that f(x) is always less than 0 on ℝ -/
def always_negative (a : ℝ) : Prop := ∀ x, f a x < 0

/-- Theorem stating that f(x) is always negative if and only if a is in the interval (-4, 0] -/
theorem f_always_negative_iff_a_in_range :
  ∀ a : ℝ, always_negative a ↔ -4 < a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_f_always_negative_iff_a_in_range_l2897_289775


namespace NUMINAMATH_CALUDE_polygon_sides_l2897_289740

/-- A polygon with equal internal angles and external angles equal to 2/3 of the adjacent internal angles has 5 sides. -/
theorem polygon_sides (n : ℕ) (internal_angle : ℝ) (external_angle : ℝ) : 
  n > 2 →
  internal_angle > 0 →
  external_angle > 0 →
  (n : ℝ) * internal_angle = (n - 2 : ℝ) * 180 →
  external_angle = (2 / 3) * internal_angle →
  internal_angle + external_angle = 180 →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l2897_289740


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l2897_289742

theorem complex_expression_equals_negative_two :
  (Real.sqrt 6 + Real.sqrt 2) * (Real.sqrt 3 - 2) * Real.sqrt (Real.sqrt 3 + 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l2897_289742


namespace NUMINAMATH_CALUDE_parabola_focus_l2897_289739

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the focus of a parabola
def focus (p : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  let (x, y) := p
  parabola x y ∧ x = 0 ∧ y = -1

-- Theorem statement
theorem parabola_focus :
  ∃ p : ℝ × ℝ, focus p parabola :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l2897_289739


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2897_289703

/-- Given an arithmetic sequence {aₙ}, prove that if a₅ + a₁₁ = 30 and a₄ = 7, then a₁₂ = 23 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h_sum : a 5 + a 11 = 30)
  (h_a4 : a 4 = 7) :
  a 12 = 23 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2897_289703


namespace NUMINAMATH_CALUDE_sixty_four_to_five_sixths_l2897_289778

theorem sixty_four_to_five_sixths (h : 64 = 2^6) : 64^(5/6) = 32 := by
  sorry

end NUMINAMATH_CALUDE_sixty_four_to_five_sixths_l2897_289778


namespace NUMINAMATH_CALUDE_zarnin_battle_station_staffing_l2897_289796

/-- The number of ways to fill positions from a pool of candidates -/
def fill_positions (total_candidates : Nat) (positions_to_fill : Nat) : Nat :=
  List.range positions_to_fill
  |>.map (fun i => total_candidates - i)
  |>.prod

/-- The problem statement -/
theorem zarnin_battle_station_staffing :
  fill_positions 20 5 = 1860480 := by
  sorry

end NUMINAMATH_CALUDE_zarnin_battle_station_staffing_l2897_289796


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2897_289724

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 + (1 + x)^6 + (1 + x)^7 + (1 + x)^8
           = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 502 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2897_289724


namespace NUMINAMATH_CALUDE_soccer_camp_ratio_l2897_289714

/-- Proves the ratio of kids going to soccer camp in the morning to the total number of kids going to soccer camp -/
theorem soccer_camp_ratio (total_kids : ℕ) (soccer_kids : ℕ) (afternoon_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : soccer_kids = total_kids / 2)
  (h3 : afternoon_kids = 750) :
  (soccer_kids - afternoon_kids) / soccer_kids = 1 / 4 := by
  sorry

#check soccer_camp_ratio

end NUMINAMATH_CALUDE_soccer_camp_ratio_l2897_289714


namespace NUMINAMATH_CALUDE_tax_calculation_l2897_289725

theorem tax_calculation (total_earnings deductions tax_paid : ℝ) 
  (h1 : total_earnings = 100000)
  (h2 : deductions = 30000)
  (h3 : tax_paid = 12000) : 
  ∃ (taxed_at_10_percent : ℝ),
    taxed_at_10_percent = 20000 ∧
    tax_paid = 0.1 * taxed_at_10_percent + 
               0.2 * (total_earnings - deductions - taxed_at_10_percent) :=
by sorry

end NUMINAMATH_CALUDE_tax_calculation_l2897_289725


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l2897_289781

/-- The area of a rectangle inscribed in the ellipse x^2/4 + y^2/8 = 1,
    with sides parallel to the coordinate axes and length twice its width -/
theorem inscribed_rectangle_area :
  ∀ (a b : ℝ),
  (a > 0) →
  (b > 0) →
  (a = 2 * b) →
  (a^2 / 4 + b^2 / 8 = 1) →
  4 * a * b = 32 / 3 := by
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l2897_289781


namespace NUMINAMATH_CALUDE_jam_jar_max_theorem_l2897_289719

/-- Represents the initial state of jam jars --/
structure JamJars :=
  (carlson_weight : ℕ)
  (baby_weight : ℕ)
  (carlson_min_jar : ℕ)

/-- Conditions for the jam jar problem --/
def valid_jam_jars (j : JamJars) : Prop :=
  j.carlson_weight = 13 * j.baby_weight ∧
  j.carlson_weight - j.carlson_min_jar = 8 * (j.baby_weight + j.carlson_min_jar)

/-- The maximum number of jars Carlson could have initially --/
def max_carlson_jars : ℕ := 23

/-- Theorem stating the maximum number of jars Carlson could have initially --/
theorem jam_jar_max_theorem (j : JamJars) (h : valid_jam_jars j) :
  (j.carlson_weight / j.carlson_min_jar : ℚ) ≤ max_carlson_jars :=
sorry

end NUMINAMATH_CALUDE_jam_jar_max_theorem_l2897_289719


namespace NUMINAMATH_CALUDE_guppies_ratio_l2897_289762

/-- The number of guppies Haylee has -/
def haylee_guppies : ℕ := 36

/-- The number of guppies Jose has -/
def jose_guppies : ℕ := haylee_guppies / 2

/-- The number of guppies Charliz has -/
def charliz_guppies : ℕ := jose_guppies / 3

/-- The total number of guppies all four friends have -/
def total_guppies : ℕ := 84

/-- The number of guppies Nicolai has -/
def nicolai_guppies : ℕ := total_guppies - (haylee_guppies + jose_guppies + charliz_guppies)

/-- Theorem stating that the ratio of Nicolai's guppies to Charliz's guppies is 4:1 -/
theorem guppies_ratio : nicolai_guppies / charliz_guppies = 4 := by sorry

end NUMINAMATH_CALUDE_guppies_ratio_l2897_289762


namespace NUMINAMATH_CALUDE_smallest_candy_count_l2897_289799

theorem smallest_candy_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n + 5) % 8 = 0 ∧ 
  (n - 8) % 5 = 0 ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m + 5) % 8 = 0 ∧ (m - 8) % 5 = 0 → n ≤ m) ∧
  n = 123 :=
by sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l2897_289799


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l2897_289716

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℝ) : Prop :=
  ∃ m : ℝ, n = m^2

-- Define what it means for a quadratic radical to be in its simplest form
def isSimplestQuadraticRadical (x : ℝ) : Prop :=
  x > 0 ∧ ¬(isPerfectSquare x) ∧ ∀ y z : ℝ, (y > 0 ∧ z > 0 ∧ x = y * z) → ¬(isPerfectSquare y)

-- State the theorem
theorem simplest_quadratic_radical :
  ¬(isSimplestQuadraticRadical 0.5) ∧
  ¬(isSimplestQuadraticRadical 8) ∧
  ¬(isSimplestQuadraticRadical 27) ∧
  ∀ a : ℝ, isSimplestQuadraticRadical (a^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l2897_289716


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l2897_289717

theorem ratio_of_numbers (a b : ℕ) (h1 : a = 45) (h2 : b = 60) (h3 : Nat.lcm a b = 180) :
  (a : ℚ) / b = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l2897_289717


namespace NUMINAMATH_CALUDE_marble_group_size_l2897_289710

theorem marble_group_size : 
  ∀ (x : ℕ), 
  (144 / x : ℚ) = (144 / (x + 2) : ℚ) + 1 → 
  x = 16 := by
sorry

end NUMINAMATH_CALUDE_marble_group_size_l2897_289710


namespace NUMINAMATH_CALUDE_circles_relationship_l2897_289730

/-- The positional relationship between two circles -/
theorem circles_relationship (C1 C2 : ℝ × ℝ) (r1 r2 : ℝ) : 
  (C1.1 + 1)^2 + (C1.2 + 1)^2 = 4 →
  (C2.1 - 2)^2 + (C2.2 - 1)^2 = 4 →
  r1 = 2 →
  r2 = 2 →
  C1 = (-1, -1) →
  C2 = (2, 1) →
  (r1 - r2)^2 < (C1.1 - C2.1)^2 + (C1.2 - C2.2)^2 ∧ 
  (C1.1 - C2.1)^2 + (C1.2 - C2.2)^2 < (r1 + r2)^2 :=
by sorry


end NUMINAMATH_CALUDE_circles_relationship_l2897_289730


namespace NUMINAMATH_CALUDE_mary_flour_amount_l2897_289729

/-- Given a recipe that requires a total amount of flour and the amount still needed to be added,
    calculate the amount of flour already put in. -/
def flour_already_added (total : ℕ) (to_add : ℕ) : ℕ :=
  total - to_add

/-- Theorem: Mary has already put in 2 cups of flour -/
theorem mary_flour_amount : flour_already_added 8 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_flour_amount_l2897_289729


namespace NUMINAMATH_CALUDE_inequality_proof_l2897_289704

theorem inequality_proof (a : ℝ) (h : a ≠ -1) :
  (1 + a^3) / ((1 + a)^3) ≥ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2897_289704


namespace NUMINAMATH_CALUDE_andrea_rhinestones_l2897_289743

theorem andrea_rhinestones (total : ℕ) (bought : ℚ) (found : ℚ) : 
  total = 120 → 
  bought = 2 / 5 → 
  found = 1 / 6 → 
  total - (total * bought + total * found) = 52 := by
sorry

end NUMINAMATH_CALUDE_andrea_rhinestones_l2897_289743


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2897_289761

theorem exponent_multiplication (a : ℝ) : 2 * a^2 * a^4 = 2 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2897_289761


namespace NUMINAMATH_CALUDE_marks_fruit_purchase_l2897_289718

/-- The total cost of Mark's fruit purchase --/
def total_cost (tomato_price apple_price orange_price : ℝ)
                (tomato_weight apple_weight orange_weight : ℝ)
                (apple_discount : ℝ) : ℝ :=
  tomato_price * tomato_weight +
  apple_price * apple_weight * (1 - apple_discount) +
  orange_price * orange_weight

/-- Theorem stating the total cost of Mark's fruit purchase --/
theorem marks_fruit_purchase :
  total_cost 4.50 3.25 2.75 3 7 4 0.1 = 44.975 := by
  sorry

#eval total_cost 4.50 3.25 2.75 3 7 4 0.1

end NUMINAMATH_CALUDE_marks_fruit_purchase_l2897_289718


namespace NUMINAMATH_CALUDE_proportional_relationship_l2897_289721

/-- Given that x is directly proportional to y², y is inversely proportional to √z,
    and x = 7 when z = 16, prove that x = 7/9 when z = 144 -/
theorem proportional_relationship (k m : ℝ) (h1 : k > 0) (h2 : m > 0) : 
  (∀ x y z : ℝ, x = k * y^2 ∧ y = m / Real.sqrt z → 
    (x = 7 ∧ z = 16 → x * z = 112) ∧
    (z = 144 → x = 7/9)) := by
  sorry

end NUMINAMATH_CALUDE_proportional_relationship_l2897_289721


namespace NUMINAMATH_CALUDE_negation_of_existence_l2897_289763

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ 2^x₀ * (x₀ - a) > 1) ↔
  (∀ x : ℝ, x > 0 → 2^x * (x - a) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2897_289763


namespace NUMINAMATH_CALUDE_lcm_ten_times_gcd_characterization_l2897_289727

theorem lcm_ten_times_gcd_characterization (a b : ℕ+) :
  Nat.lcm a b = 10 * Nat.gcd a b ↔
  (∃ d : ℕ+, (a = d ∧ b = 10 * d) ∨
             (a = 2 * d ∧ b = 5 * d) ∨
             (a = 5 * d ∧ b = 2 * d) ∨
             (a = 10 * d ∧ b = d)) :=
by sorry

end NUMINAMATH_CALUDE_lcm_ten_times_gcd_characterization_l2897_289727


namespace NUMINAMATH_CALUDE_equation_solution_l2897_289741

theorem equation_solution (a : ℚ) : -3 / (a - 3) = 3 / (a + 2) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2897_289741


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2897_289774

theorem roots_of_polynomial : ∃ (a b c d : ℂ), 
  (a = 2 ∧ b = -2 ∧ c = 2*I ∧ d = -2*I) ∧ 
  (∀ x : ℂ, x^4 + 4*x^3 - 2*x^2 - 20*x + 24 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2897_289774


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l2897_289794

/-- The number of ways to place 4 different balls into 4 different boxes --/
def placeBalls (emptyBoxes : Nat) : Nat :=
  if emptyBoxes = 1 then 144
  else if emptyBoxes = 2 then 84
  else 0

theorem ball_placement_theorem :
  (placeBalls 1 = 144) ∧ (placeBalls 2 = 84) := by
  sorry

#eval placeBalls 1  -- Expected output: 144
#eval placeBalls 2  -- Expected output: 84

end NUMINAMATH_CALUDE_ball_placement_theorem_l2897_289794


namespace NUMINAMATH_CALUDE_cos_difference_from_sum_of_sin_and_cos_l2897_289726

theorem cos_difference_from_sum_of_sin_and_cos 
  (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 5/4) : 
  Real.cos (A - B) = 13/32 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_from_sum_of_sin_and_cos_l2897_289726


namespace NUMINAMATH_CALUDE_standard_flowchart_property_l2897_289747

/-- Represents a flowchart --/
structure Flowchart where
  start_points : Nat
  end_points : Nat

/-- A flowchart is standard if it has exactly one start point and at least one end point --/
def is_standard (f : Flowchart) : Prop :=
  f.start_points = 1 ∧ f.end_points ≥ 1

/-- Theorem stating that a standard flowchart has exactly one start point and at least one end point --/
theorem standard_flowchart_property (f : Flowchart) (h : is_standard f) :
  f.start_points = 1 ∧ f.end_points ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_standard_flowchart_property_l2897_289747


namespace NUMINAMATH_CALUDE_parabola_points_distance_l2897_289707

/-- A parabola defined by y = 9x^2 - 3x + 2 -/
def parabola (x y : ℝ) : Prop := y = 9 * x^2 - 3 * x + 2

/-- The origin (0,0) is the midpoint of two points -/
def origin_is_midpoint (p q : ℝ × ℝ) : Prop :=
  (p.1 + q.1) / 2 = 0 ∧ (p.2 + q.2) / 2 = 0

/-- The square of the distance between two points -/
def square_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem parabola_points_distance (p q : ℝ × ℝ) :
  parabola p.1 p.2 ∧ parabola q.1 q.2 ∧ origin_is_midpoint p q →
  square_distance p q = 580 / 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_distance_l2897_289707


namespace NUMINAMATH_CALUDE_right_triangle_from_equation_l2897_289700

theorem right_triangle_from_equation (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 + 338 = 10*a + 24*b + 26*c) : 
  a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_from_equation_l2897_289700


namespace NUMINAMATH_CALUDE_a_10_equals_21_l2897_289751

def S (n : ℕ+) : ℕ := n^2 + 2*n

def a (n : ℕ+) : ℕ := S n - S (n-1)

theorem a_10_equals_21 : a 10 = 21 := by sorry

end NUMINAMATH_CALUDE_a_10_equals_21_l2897_289751


namespace NUMINAMATH_CALUDE_magnitude_of_AB_l2897_289782

-- Define points A and B
def A : ℝ × ℝ := (-3, 4)
def B : ℝ × ℝ := (5, -2)

-- Define vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem: The magnitude of vector AB is 10
theorem magnitude_of_AB : Real.sqrt (vectorAB.1^2 + vectorAB.2^2) = 10 := by
  sorry


end NUMINAMATH_CALUDE_magnitude_of_AB_l2897_289782


namespace NUMINAMATH_CALUDE_problem_statement_l2897_289767

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - 2 * (Real.sin (ω * x / 2))^2

theorem problem_statement 
  (ω : ℝ) 
  (h_ω_pos : ω > 0)
  (h_period : ∀ x, f ω (x + 3 * Real.pi) = f ω x)
  (a b c A B C : ℝ)
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi)
  (h_b : b = 2)
  (h_fA : f ω A = Real.sqrt 3 - 1)
  (h_sides : Real.sqrt 3 * a = 2 * b * Real.sin A) :
  (∃ x ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, ∀ y ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, f ω y ≥ f ω x) ∧ 
  (∃ x ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, ∀ y ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, f ω y ≤ f ω x) ∧
  (1/2 * a * b * Real.sin C = (3 + Real.sqrt 3) / 3) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2897_289767


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l2897_289756

/-- The perimeter of a T shape formed by a vertical rectangle and a horizontal rectangle -/
def t_perimeter (v_width v_height h_width h_height : ℝ) : ℝ :=
  2 * v_height + 2 * h_width + h_height

/-- Theorem: The perimeter of the T shape is 22 inches -/
theorem t_shape_perimeter :
  t_perimeter 2 6 3 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l2897_289756


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2897_289732

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (2 * a - c) * Real.cos B = b * Real.cos C →
  a = 2 →
  c = 3 →
  B = π / 3 ∧ Real.sin C = (3 * Real.sqrt 14) / 14 := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2897_289732


namespace NUMINAMATH_CALUDE_biology_exam_failure_count_l2897_289790

theorem biology_exam_failure_count : 
  ∀ (total_students : ℕ) 
    (perfect_score_fraction : ℚ)
    (passing_score_fraction : ℚ),
  total_students = 80 →
  perfect_score_fraction = 2/5 →
  passing_score_fraction = 1/2 →
  (total_students : ℚ) * perfect_score_fraction +
  (total_students : ℚ) * (1 - perfect_score_fraction) * passing_score_fraction +
  (total_students : ℚ) * (1 - perfect_score_fraction) * (1 - passing_score_fraction) = 
  (total_students : ℚ) →
  (total_students : ℚ) * (1 - perfect_score_fraction) * (1 - passing_score_fraction) = 24 :=
by sorry

end NUMINAMATH_CALUDE_biology_exam_failure_count_l2897_289790


namespace NUMINAMATH_CALUDE_solve_problem_l2897_289779

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^(digits.length - 1 - i)) 0

def problem : Prop :=
  let base_6_num := [5, 4, 3, 2, 1, 0]
  let base_7_num := [4, 3, 2, 1, 0]
  (base_to_decimal base_6_num 6) - (base_to_decimal base_7_num 7) = 34052

theorem solve_problem : problem := by
  sorry

end NUMINAMATH_CALUDE_solve_problem_l2897_289779


namespace NUMINAMATH_CALUDE_total_bathing_suits_l2897_289768

theorem total_bathing_suits (men_suits women_suits : ℕ) 
  (h1 : men_suits = 14797) 
  (h2 : women_suits = 4969) : 
  men_suits + women_suits = 19766 := by
  sorry

end NUMINAMATH_CALUDE_total_bathing_suits_l2897_289768


namespace NUMINAMATH_CALUDE_not_equal_to_three_halves_l2897_289750

theorem not_equal_to_three_halves : ∃ x : ℚ, x ≠ (3/2 : ℚ) ∧
  (x = (5/3 : ℚ)) ∧
  ((9/6 : ℚ) = (3/2 : ℚ)) ∧
  ((3/2 : ℚ) = (3/2 : ℚ)) ∧
  ((7/4 : ℚ) = (3/2 : ℚ)) ∧
  ((9/6 : ℚ) = (3/2 : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_not_equal_to_three_halves_l2897_289750


namespace NUMINAMATH_CALUDE_line_symmetry_l2897_289754

-- Define the original line
def original_line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define the line of symmetry
def symmetry_line (x : ℝ) : Prop := x = 1

-- Define the symmetric line
def symmetric_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Theorem statement
theorem line_symmetry :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), original_line x₀ y₀ ∧ symmetry_line x₀) →
  (symmetric_line x y ↔ 
    ∃ (x₁ y₁ : ℝ), original_line x₁ y₁ ∧ 
    x - x₀ = x₀ - x₁ ∧ 
    y - y₀ = y₀ - y₁ ∧
    symmetry_line x₀) :=
by sorry

end NUMINAMATH_CALUDE_line_symmetry_l2897_289754


namespace NUMINAMATH_CALUDE_no_real_roots_l2897_289776

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 1) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l2897_289776


namespace NUMINAMATH_CALUDE_hexagon_intersection_collinearity_l2897_289735

/-- Represents a point in 2D space -/
structure Point :=
  (x y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a b c : ℝ)

/-- Represents a hexagon -/
structure Hexagon :=
  (A B C D E F : Point)

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop := sorry

/-- Returns the intersection point of two lines -/
def intersectionPoint (l1 l2 : Line) : Point := sorry

/-- Theorem: Collinearity of intersections in a hexagon with specific conditions -/
theorem hexagon_intersection_collinearity 
  (ABCDEF : Hexagon)
  (diagonalIntersection : Point)
  (hDiagonals : intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0) = diagonalIntersection ∧ 
                intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0) = diagonalIntersection ∧ 
                intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0) = diagonalIntersection)
  (A' : Point) (hA' : A' = intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
  (B' : Point) (hB' : B' = intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
  (C' : Point) (hC' : C' = intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
  (D' E' F' : Point)
  : collinear 
      (intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
      (intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0))
      (intersectionPoint (Line.mk 0 0 0) (Line.mk 0 0 0)) :=
by sorry

end NUMINAMATH_CALUDE_hexagon_intersection_collinearity_l2897_289735


namespace NUMINAMATH_CALUDE_inequality_range_inequality_solution_l2897_289706

-- Part 1
theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (1 - a) * x + a - 2 ≥ -2) ↔ a ∈ Set.Ici (1/3) :=
sorry

-- Part 2
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x < 1 }
  else if a > 0 then { x | -1/a < x ∧ x < 1 }
  else if a = -1 then { x | x ≠ 1 }
  else if a < -1 then { x | x > 1 ∨ x < -1/a }
  else { x | x < 1 ∨ x > -1/a }

theorem inequality_solution (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a * x^2 + (1 - a) * x + a - 2 < a - 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_inequality_solution_l2897_289706


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2897_289786

theorem cubic_equation_solution (p q : ℝ) : 
  (3 * p^2 - 5 * p - 21 = 0) → 
  (3 * q^2 - 5 * q - 21 = 0) → 
  p ≠ q →
  (9 * p^3 - 9 * q^3) / (p - q) = 88 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2897_289786


namespace NUMINAMATH_CALUDE_line_contains_point_l2897_289760

/-- A line in the xy-plane is represented by the equation 2 - kx = -4y for some real number k. -/
def line (k : ℝ) (x y : ℝ) : Prop := 2 - k * x = -4 * y

/-- The point (2, -1) lies on the line. -/
def point_on_line (k : ℝ) : Prop := line k 2 (-1)

/-- The value of k for which the line contains the point (2, -1) is -1. -/
theorem line_contains_point : ∃! k : ℝ, point_on_line k ∧ k = -1 := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l2897_289760


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l2897_289713

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of the expression (35)^7 + (93)^45 -/
def expression : ℕ := 35^7 + 93^45

/-- Theorem stating that the units digit of (35)^7 + (93)^45 is 8 -/
theorem units_digit_of_expression : unitsDigit expression = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l2897_289713


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_factorials_49_l2897_289755

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_of_sum_of_factorials_49 :
  (sum_of_factorials 49) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_factorials_49_l2897_289755


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l2897_289722

theorem smallest_integer_satisfying_conditions : ∃ (n : ℕ), n > 1 ∧ 
  (∃ (k : ℕ), (5 * n) / 3 = k + 2/3) ∧
  (∃ (k : ℕ), (7 * n) / 5 = k + 2/5) ∧
  (∃ (k : ℕ), (9 * n) / 7 = k + 2/7) ∧
  (∃ (k : ℕ), (11 * n) / 9 = k + 2/9) ∧
  (∀ (m : ℕ), m > 1 → 
    ((∃ (k : ℕ), (5 * m) / 3 = k + 2/3) ∧
     (∃ (k : ℕ), (7 * m) / 5 = k + 2/5) ∧
     (∃ (k : ℕ), (9 * m) / 7 = k + 2/7) ∧
     (∃ (k : ℕ), (11 * m) / 9 = k + 2/9)) → m ≥ n) ∧
  n = 316 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l2897_289722


namespace NUMINAMATH_CALUDE_non_adjacent_arrangements_l2897_289770

def number_of_people : ℕ := 6

def number_of_gaps (n : ℕ) : ℕ := n + 1

def permutations (n : ℕ) : ℕ := n.factorial

def arrangements_with_gaps (n : ℕ) : ℕ :=
  permutations (n - 2) * (number_of_gaps (n - 2)).choose 2

theorem non_adjacent_arrangements :
  arrangements_with_gaps number_of_people = 480 := by sorry

end NUMINAMATH_CALUDE_non_adjacent_arrangements_l2897_289770


namespace NUMINAMATH_CALUDE_max_principals_in_period_l2897_289798

/-- Represents the duration of the period in years -/
def period_duration : ℕ := 8

/-- Represents the duration of a principal's term in years -/
def term_duration : ℕ := 4

/-- Represents the maximum number of non-overlapping terms that can fit within the period -/
def max_principals : ℕ := period_duration / term_duration

theorem max_principals_in_period :
  max_principals = 2 :=
sorry

end NUMINAMATH_CALUDE_max_principals_in_period_l2897_289798


namespace NUMINAMATH_CALUDE_anns_age_l2897_289777

theorem anns_age (a b : ℕ) : 
  a + b = 50 → 
  b = (2 * a / 3 : ℚ) + 2 * (a - b) → 
  a = 26 := by
sorry

end NUMINAMATH_CALUDE_anns_age_l2897_289777


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9871_l2897_289745

theorem largest_prime_factor_of_9871 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 9871 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 9871 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9871_l2897_289745


namespace NUMINAMATH_CALUDE_base_ten_arithmetic_l2897_289708

theorem base_ten_arithmetic : (456 + 123) - 579 = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_ten_arithmetic_l2897_289708


namespace NUMINAMATH_CALUDE_lewis_harvest_earnings_l2897_289733

/-- Calculates the total earnings during harvest season after paying rent -/
def harvest_earnings (weekly_earnings : ℕ) (weekly_rent : ℕ) (harvest_weeks : ℕ) : ℕ :=
  (weekly_earnings - weekly_rent) * harvest_weeks

/-- Theorem: Lewis's earnings during harvest season -/
theorem lewis_harvest_earnings :
  harvest_earnings 403 49 233 = 82782 := by
  sorry

end NUMINAMATH_CALUDE_lewis_harvest_earnings_l2897_289733


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l2897_289769

/-- Simple interest calculation -/
theorem simple_interest_rate_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 720)
  (h2 : interest = 180)
  (h3 : time = 4)
  : (interest * 100) / (principal * time) = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_percent_l2897_289769


namespace NUMINAMATH_CALUDE_complex_number_subtraction_l2897_289758

theorem complex_number_subtraction : (5 * Complex.I) - (2 + 2 * Complex.I) = -2 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_subtraction_l2897_289758


namespace NUMINAMATH_CALUDE_congruence_solution_l2897_289785

theorem congruence_solution (n : ℤ) : 13 * n ≡ 8 [ZMOD 47] ↔ n ≡ 29 [ZMOD 47] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2897_289785


namespace NUMINAMATH_CALUDE_cubic_equation_solution_sum_l2897_289748

theorem cubic_equation_solution_sum (r s t : ℝ) : 
  r^3 - 5*r^2 + 6*r = 9 →
  s^3 - 5*s^2 + 6*s = 9 →
  t^3 - 5*t^2 + 6*t = 9 →
  r*s/t + s*t/r + t*r/s = -6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_sum_l2897_289748


namespace NUMINAMATH_CALUDE_triangle_area_angle_l2897_289759

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S = (√3/4)(a² + b² - c²), then C = π/3 -/
theorem triangle_area_angle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)
  S = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) →
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_angle_l2897_289759


namespace NUMINAMATH_CALUDE_b_plus_c_equals_three_l2897_289789

/-- A function f: ℝ → ℝ defined as f(x) = x^3 + bx^2 + cx -/
def f (b c : ℝ) : ℝ → ℝ := λ x ↦ x^3 + b*x^2 + c*x

/-- The derivative of f -/
def f_deriv (b c : ℝ) : ℝ → ℝ := λ x ↦ 3*x^2 + 2*b*x + c

/-- A function g: ℝ → ℝ defined as g(x) = f(x) - f'(x) -/
def g (b c : ℝ) : ℝ → ℝ := λ x ↦ f b c x - f_deriv b c x

/-- A predicate stating that a function is odd -/
def is_odd_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

/-- The main theorem -/
theorem b_plus_c_equals_three (b c : ℝ) :
  is_odd_function (g b c) → b + c = 3 := by sorry

end NUMINAMATH_CALUDE_b_plus_c_equals_three_l2897_289789


namespace NUMINAMATH_CALUDE_inequality_range_l2897_289788

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^4 + (a-2)*x^2 + a ≥ 0) ↔ a ≥ 4 - 2*Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l2897_289788


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2897_289757

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

/-- Given conditions for the geometric sequence -/
def GeometricSequenceConditions (a : ℕ → ℝ) : Prop :=
  GeometricSequence a ∧ (a 5 * a 11 = 4) ∧ (a 3 + a 13 = 5)

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h : GeometricSequenceConditions a) : 
  (a 14 / a 4 = 4) ∨ (a 14 / a 4 = 1/4) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2897_289757


namespace NUMINAMATH_CALUDE_multiplication_problem_l2897_289705

theorem multiplication_problem : ∃ x : ℕ, 72516 * x = 724797420 ∧ x = 10001 := by sorry

end NUMINAMATH_CALUDE_multiplication_problem_l2897_289705


namespace NUMINAMATH_CALUDE_area_is_two_l2897_289765

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (A B C : Circle) : Prop :=
  A.radius = 1 ∧
  B.radius = 1 ∧
  C.radius = 1 ∧
  -- A and B are tangent
  dist A.center B.center = 2 ∧
  -- C is tangent to the midpoint of AB
  C.center.1 = (A.center.1 + B.center.1) / 2 ∧
  C.center.2 = (A.center.2 + B.center.2) / 2 + 1

-- Define the area function
def area_inside_C_outside_AB (A B C : Circle) : ℝ := sorry

-- Theorem statement
theorem area_is_two (A B C : Circle) :
  problem_setup A B C → area_inside_C_outside_AB A B C = 2 := by sorry

end NUMINAMATH_CALUDE_area_is_two_l2897_289765


namespace NUMINAMATH_CALUDE_equation_solution_l2897_289766

theorem equation_solution : ∃! x : ℝ, (3 - x) / (x - 4) + 1 / (4 - x) = 1 ∧ x ≠ 4 :=
  by sorry

end NUMINAMATH_CALUDE_equation_solution_l2897_289766


namespace NUMINAMATH_CALUDE_annie_brownies_left_l2897_289787

/-- Calculates the number of brownies Annie has left after sharing -/
def brownies_left (initial : ℕ) (to_simon : ℕ) : ℕ :=
  let to_admin := initial / 2
  let after_admin := initial - to_admin
  let to_carl := after_admin / 2
  let after_carl := after_admin - to_carl
  after_carl - to_simon

/-- Proves that Annie has 3 brownies left after sharing -/
theorem annie_brownies_left :
  brownies_left 20 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_annie_brownies_left_l2897_289787


namespace NUMINAMATH_CALUDE_seaweed_for_human_consumption_l2897_289711

/-- Given that:
  - 400 pounds of seaweed are harvested
  - 50% of seaweed is used for starting fires
  - 150 pounds are fed to livestock
Prove that 25% of the remaining seaweed after starting fires can be eaten by humans -/
theorem seaweed_for_human_consumption 
  (total_seaweed : ℝ) 
  (fire_seaweed_percentage : ℝ) 
  (livestock_seaweed : ℝ) 
  (h1 : total_seaweed = 400)
  (h2 : fire_seaweed_percentage = 0.5)
  (h3 : livestock_seaweed = 150) :
  let remaining_seaweed := total_seaweed * (1 - fire_seaweed_percentage)
  let human_seaweed := remaining_seaweed - livestock_seaweed
  human_seaweed / remaining_seaweed = 0.25 := by
sorry

end NUMINAMATH_CALUDE_seaweed_for_human_consumption_l2897_289711


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l2897_289701

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 8 * b = 5 * c)  -- 8 bowling balls weigh the same as 5 canoes
  (h2 : 3 * c = 135)    -- 3 canoes weigh 135 pounds
  : b = 28.125 :=       -- One bowling ball weighs 28.125 pounds
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l2897_289701


namespace NUMINAMATH_CALUDE_ball_max_height_l2897_289709

/-- The height function of the ball's path -/
def h (t : ℝ) : ℝ := -16 * t^2 + 80 * t + 21

/-- Theorem stating that the maximum height of the ball is 121 feet -/
theorem ball_max_height :
  ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 121 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l2897_289709
