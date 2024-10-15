import Mathlib

namespace NUMINAMATH_CALUDE_output_is_76_l2297_229777

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 ≤ 40 then step1 + 10 else step1 - 7
  step2 * 2

theorem output_is_76 : function_machine 15 = 76 := by
  sorry

end NUMINAMATH_CALUDE_output_is_76_l2297_229777


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2297_229745

theorem sufficient_not_necessary : 
  (∀ x : ℝ, (|x| < 2 → x^2 - x - 6 < 0)) ∧ 
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ ¬(|x| < 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2297_229745


namespace NUMINAMATH_CALUDE_daily_pay_rate_is_twenty_l2297_229706

/-- Calculates the daily pay rate given the total days, worked days, forfeit amount, and net earnings -/
def calculate_daily_pay_rate (total_days : ℕ) (worked_days : ℕ) (forfeit_amount : ℚ) (net_earnings : ℚ) : ℚ :=
  let idle_days := total_days - worked_days
  let total_forfeit := idle_days * forfeit_amount
  (net_earnings + total_forfeit) / worked_days

/-- Theorem stating that given the specified conditions, the daily pay rate is $20 -/
theorem daily_pay_rate_is_twenty :
  calculate_daily_pay_rate 25 23 5 450 = 20 := by
  sorry

end NUMINAMATH_CALUDE_daily_pay_rate_is_twenty_l2297_229706


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2297_229784

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2297_229784


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_all_squares_equiangular_all_squares_rectangles_all_squares_regular_polygons_all_squares_similar_l2297_229750

/-- A square is a quadrilateral with four equal sides and four right angles. -/
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Two squares are congruent if they have the same side length. -/
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

/-- Theorem: Not all squares are congruent to each other. -/
theorem not_all_squares_congruent : ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry

/-- All squares are equiangular. -/
theorem all_squares_equiangular : True := by
  sorry

/-- All squares are rectangles. -/
theorem all_squares_rectangles : True := by
  sorry

/-- All squares are regular polygons. -/
theorem all_squares_regular_polygons : True := by
  sorry

/-- All squares are similar to each other. -/
theorem all_squares_similar : True := by
  sorry

end NUMINAMATH_CALUDE_not_all_squares_congruent_all_squares_equiangular_all_squares_rectangles_all_squares_regular_polygons_all_squares_similar_l2297_229750


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2297_229735

theorem simplify_trig_expression :
  (Real.sin (15 * π / 180) + Real.sin (45 * π / 180)) /
  (Real.cos (15 * π / 180) + Real.cos (45 * π / 180)) =
  Real.tan (30 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2297_229735


namespace NUMINAMATH_CALUDE_total_money_l2297_229796

theorem total_money (john emma lucas : ℚ) 
  (h1 : john = 4 / 5)
  (h2 : emma = 2 / 5)
  (h3 : lucas = 1 / 2) :
  john + emma + lucas = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2297_229796


namespace NUMINAMATH_CALUDE_kangaroo_hops_l2297_229721

/-- The distance covered in a single hop, given the remaining distance -/
def hop_distance (remaining : ℚ) : ℚ := (1 / 4) * remaining

/-- The sum of distances covered in n hops -/
def total_distance (n : ℕ) : ℚ :=
  (1 - (3/4)^n) / (1/4)

/-- The theorem stating that after 6 hops, the total distance covered is 3367/4096 -/
theorem kangaroo_hops : total_distance 6 = 3367 / 4096 := by sorry

end NUMINAMATH_CALUDE_kangaroo_hops_l2297_229721


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_positive_n_value_l2297_229749

theorem unique_solution_quadratic (n : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + n * x + 36 = 0) → n = 36 ∨ n = -36 :=
by sorry

theorem positive_n_value (n : ℝ) :
  (∃! x : ℝ, 9 * x^2 + n * x + 36 = 0) ∧ n > 0 → n = 36 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_positive_n_value_l2297_229749


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l2297_229724

theorem mean_proportional_problem (x : ℝ) :
  (156 : ℝ)^2 = 234 * x → x = 104 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l2297_229724


namespace NUMINAMATH_CALUDE_sin_eq_cos_condition_l2297_229798

open Real

theorem sin_eq_cos_condition (α : ℝ) :
  (∃ k : ℤ, α = π / 4 + 2 * k * π) → sin α = cos α ∧
  ¬ (sin α = cos α → ∃ k : ℤ, α = π / 4 + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_sin_eq_cos_condition_l2297_229798


namespace NUMINAMATH_CALUDE_existence_of_mn_l2297_229719

theorem existence_of_mn (k : ℕ+) : 
  (∃ m n : ℕ+, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 := by
sorry

end NUMINAMATH_CALUDE_existence_of_mn_l2297_229719


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l2297_229753

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a b : Line) (α : Plane) 
  (h1 : parallel a α) 
  (h2 : perpendicular b α) : 
  perpendicularLines a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l2297_229753


namespace NUMINAMATH_CALUDE_complex_modulus_one_l2297_229756

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l2297_229756


namespace NUMINAMATH_CALUDE_shirt_markup_proof_l2297_229760

/-- Proves that for a shirt with an initial price of $45 after an 80% markup from wholesale,
    increasing the price by $5 results in a 100% markup from the wholesale price. -/
theorem shirt_markup_proof (initial_price : ℝ) (initial_markup : ℝ) (price_increase : ℝ) :
  initial_price = 45 ∧
  initial_markup = 0.8 ∧
  price_increase = 5 →
  let wholesale_price := initial_price / (1 + initial_markup)
  let new_price := initial_price + price_increase
  (new_price - wholesale_price) / wholesale_price = 1 :=
by sorry

end NUMINAMATH_CALUDE_shirt_markup_proof_l2297_229760


namespace NUMINAMATH_CALUDE_hundredth_decimal_is_9_l2297_229794

/-- The decimal expansion of 10/11 -/
def decimal_expansion_10_11 : ℕ → ℕ := 
  fun n => if n % 2 = 0 then 0 else 9

/-- The 100th decimal digit in the expansion of 10/11 -/
def hundredth_decimal : ℕ := decimal_expansion_10_11 100

theorem hundredth_decimal_is_9 : hundredth_decimal = 9 := by sorry

end NUMINAMATH_CALUDE_hundredth_decimal_is_9_l2297_229794


namespace NUMINAMATH_CALUDE_composite_for_n_greater_than_two_l2297_229742

def number_with_ones_and_seven (n : ℕ) : ℕ :=
  7 * 10^(n-1) + (10^(n-1) - 1) / 9

theorem composite_for_n_greater_than_two :
  ∀ n : ℕ, n > 2 → ¬(Nat.Prime (number_with_ones_and_seven n)) :=
sorry

end NUMINAMATH_CALUDE_composite_for_n_greater_than_two_l2297_229742


namespace NUMINAMATH_CALUDE_birds_on_fence_l2297_229765

theorem birds_on_fence (initial_birds additional_birds : ℕ) :
  initial_birds = 12 → additional_birds = 8 →
  initial_birds + additional_birds = 20 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l2297_229765


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l2297_229768

/-- Represents the number of boys in the group -/
def num_boys : Nat := 3

/-- Represents the number of girls in the group -/
def num_girls : Nat := 2

/-- Represents the number of students to be selected -/
def num_selected : Nat := 2

/-- Event: Exactly 1 boy is selected -/
def event_one_boy (selected : Finset (Fin (num_boys + num_girls))) : Prop :=
  (selected.filter (λ i => i.val < num_boys)).card = 1

/-- Event: Exactly 2 girls are selected -/
def event_two_girls (selected : Finset (Fin (num_boys + num_girls))) : Prop :=
  (selected.filter (λ i => i.val ≥ num_boys)).card = 2

/-- Theorem: The events are mutually exclusive but not complementary -/
theorem events_mutually_exclusive_not_complementary :
  (∀ selected : Finset (Fin (num_boys + num_girls)), selected.card = num_selected →
    ¬(event_one_boy selected ∧ event_two_girls selected)) ∧
  (∃ selected : Finset (Fin (num_boys + num_girls)), selected.card = num_selected →
    ¬event_one_boy selected ∧ ¬event_two_girls selected) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l2297_229768


namespace NUMINAMATH_CALUDE_two_segment_train_journey_time_l2297_229700

/-- Calculates the total time for a two-segment train journey -/
theorem two_segment_train_journey_time
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ)
  (h1 : distance1 = 80)
  (h2 : speed1 = 50)
  (h3 : distance2 = 150)
  (h4 : speed2 = 75)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0) :
  distance1 / speed1 + distance2 / speed2 = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_two_segment_train_journey_time_l2297_229700


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2297_229772

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, x^2 + 5*x - 24 < 0 ↔ -8 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2297_229772


namespace NUMINAMATH_CALUDE_river_depth_l2297_229758

theorem river_depth (depth_may : ℝ) (depth_june : ℝ) (depth_july : ℝ) 
  (h1 : depth_june = depth_may + 10)
  (h2 : depth_july = 3 * depth_june)
  (h3 : depth_july = 45) : 
  depth_may = 5 := by
sorry

end NUMINAMATH_CALUDE_river_depth_l2297_229758


namespace NUMINAMATH_CALUDE_bill_animals_l2297_229770

-- Define the number of rats
def num_rats : ℕ := 60

-- Define the relationship between rats and chihuahuas
def num_chihuahuas : ℕ := num_rats / 6

-- Define the total number of animals
def total_animals : ℕ := num_rats + num_chihuahuas

-- Theorem to prove
theorem bill_animals : total_animals = 70 := by
  sorry

end NUMINAMATH_CALUDE_bill_animals_l2297_229770


namespace NUMINAMATH_CALUDE_parabola_vertex_l2297_229711

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = -3 * (x - 1)^2 - 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, -2)

/-- Theorem: The vertex of the parabola y = -3(x-1)^2 - 2 is at the point (1, -2) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2297_229711


namespace NUMINAMATH_CALUDE_sale_discount_theorem_l2297_229799

/-- Calculates the final amount paid after applying a discount based on the purchase amount -/
def final_amount_paid (initial_amount : ℕ) (discount_per_hundred : ℕ) : ℕ :=
  initial_amount - (initial_amount / 100) * discount_per_hundred

/-- Theorem stating that for a $250 purchase with $10 off per $100 spent, the final amount is $230 -/
theorem sale_discount_theorem :
  final_amount_paid 250 10 = 230 := by
  sorry

end NUMINAMATH_CALUDE_sale_discount_theorem_l2297_229799


namespace NUMINAMATH_CALUDE_max_mondays_in_45_days_l2297_229773

/-- The maximum number of Mondays in 45 consecutive days -/
def max_mondays : ℕ := 7

/-- A function that returns the day number of the nth Monday in a sequence, 
    assuming the first day is a Monday -/
def monday_sequence (n : ℕ) : ℕ := 1 + 7 * n

theorem max_mondays_in_45_days : 
  (∃ (start : ℕ), ∀ (i : ℕ), i < max_mondays → 
    start + monday_sequence i ≤ 45) ∧ 
  (∀ (start : ℕ), ∃ (i : ℕ), i = max_mondays → 
    45 < start + monday_sequence i) :=
sorry

end NUMINAMATH_CALUDE_max_mondays_in_45_days_l2297_229773


namespace NUMINAMATH_CALUDE_count_divisors_multiple_of_five_l2297_229736

/-- The number of positive divisors of 7560 that are multiples of 5 -/
def divisors_multiple_of_five : ℕ :=
  (Finset.range 4).card * (Finset.range 4).card * 1 * (Finset.range 2).card

theorem count_divisors_multiple_of_five :
  7560 = 2^3 * 3^3 * 5^1 * 7^1 →
  divisors_multiple_of_five = 32 := by
sorry

end NUMINAMATH_CALUDE_count_divisors_multiple_of_five_l2297_229736


namespace NUMINAMATH_CALUDE_sqrt_five_irrational_l2297_229713

theorem sqrt_five_irrational :
  ∀ (x : ℝ), x ^ 2 = 5 → ¬ (∃ (a b : ℤ), b ≠ 0 ∧ x = a / b) :=
by sorry

def zero_rational : ℚ := 0

def three_point_fourteen_rational : ℚ := 314 / 100

def negative_eight_sevenths_rational : ℚ := -8 / 7

#check sqrt_five_irrational
#check zero_rational
#check three_point_fourteen_rational
#check negative_eight_sevenths_rational

end NUMINAMATH_CALUDE_sqrt_five_irrational_l2297_229713


namespace NUMINAMATH_CALUDE_min_sum_theorem_l2297_229730

-- Define the equation
def equation (x y : ℝ) : Prop := -x^2 + 7*x + y - 10 = 0

-- Define the sum function
def sum (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem min_sum_theorem :
  ∃ (min : ℝ), min = 1 ∧ 
  (∀ x y : ℝ, equation x y → sum x y ≥ min) ∧
  (∃ x y : ℝ, equation x y ∧ sum x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_sum_theorem_l2297_229730


namespace NUMINAMATH_CALUDE_michelle_savings_denomination_l2297_229727

/-- Given a total savings amount and a number of bills, calculate the denomination of each bill. -/
def billDenomination (totalSavings : ℕ) (numBills : ℕ) : ℕ :=
  totalSavings / numBills

/-- Theorem: Given Michelle's total savings of $800 and 8 bills, the denomination of each bill is $100. -/
theorem michelle_savings_denomination :
  billDenomination 800 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_michelle_savings_denomination_l2297_229727


namespace NUMINAMATH_CALUDE_min_quadrilateral_area_l2297_229763

/-- Definition of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The ellipse passes through the point (1, √2/2) -/
axiom point_on_ellipse : ellipse 1 (Real.sqrt 2 / 2)

/-- The point (1,0) is a focus of the ellipse -/
axiom focus_point : ∃ c, c^2 = 1 ∧ c^2 = 2 - 1

/-- Definition of perpendicular lines through (1,0) -/
def perpendicular_lines (m₁ m₂ : ℝ) : Prop := 
  m₁ * m₂ = -1 ∧ m₁ ≠ 0 ∧ m₂ ≠ 0

/-- Definition of the area of the quadrilateral formed by intersection points -/
noncomputable def quadrilateral_area (m₁ m₂ : ℝ) : ℝ := 
  4 * (m₁^2 + 1)^2 / ((m₁^2 + 2) * (2 * m₂^2 + 1))

/-- The main theorem to prove -/
theorem min_quadrilateral_area : 
  ∃ (m₁ m₂ : ℝ), perpendicular_lines m₁ m₂ ∧ 
  (∀ (n₁ n₂ : ℝ), perpendicular_lines n₁ n₂ → 
    quadrilateral_area m₁ m₂ ≤ quadrilateral_area n₁ n₂) ∧
  quadrilateral_area m₁ m₂ = 16/9 :=
sorry

end NUMINAMATH_CALUDE_min_quadrilateral_area_l2297_229763


namespace NUMINAMATH_CALUDE_prob_different_colors_specific_l2297_229761

/-- The probability of drawing two chips of different colors with replacement -/
def prob_different_colors (blue red yellow : ℕ) : ℚ :=
  let total := blue + red + yellow
  let prob_blue := blue / total
  let prob_red := red / total
  let prob_yellow := yellow / total
  let prob_not_blue := (red + yellow) / total
  let prob_not_red := (blue + yellow) / total
  let prob_not_yellow := (blue + red) / total
  prob_blue * prob_not_blue + prob_red * prob_not_red + prob_yellow * prob_not_yellow

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors_specific : 
  prob_different_colors 7 5 4 = 83 / 128 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_specific_l2297_229761


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2297_229718

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  h1 : a 2 = 12
  h2 : d = -2
  h3 : ∀ n : ℕ, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- The theorem to prove -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) :
  ∃ n : ℕ, seq.a n = -20 ∧ n = 18 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2297_229718


namespace NUMINAMATH_CALUDE_oblique_asymptote_of_f_l2297_229741

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 8 * x + 5) / (x + 4)

theorem oblique_asymptote_of_f :
  ∃ (a b : ℝ), a ≠ 0 ∧ (∀ ε > 0, ∃ M, ∀ x > M, |f x - (a * x + b)| < ε) ∧ a = 3 ∧ b = -4 :=
sorry

end NUMINAMATH_CALUDE_oblique_asymptote_of_f_l2297_229741


namespace NUMINAMATH_CALUDE_rectangle_area_l2297_229767

/-- The area of a rectangle with sides of length (a - b) and (c + d) is equal to ac + ad - bc - bd. -/
theorem rectangle_area (a b c d : ℝ) : 
  let length := a - b
  let width := c + d
  length * width = a*c + a*d - b*c - b*d := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2297_229767


namespace NUMINAMATH_CALUDE_algebraic_identities_l2297_229792

theorem algebraic_identities :
  (∃ (x : ℝ), x^2 = 3 ∧ x > 0) ∧ 
  (∃ (y : ℝ), y^2 = 2 ∧ y > 0) →
  (3 * Real.sqrt 3 - (Real.sqrt 12 + Real.sqrt (1/3)) = 2 * Real.sqrt 3 / 3) ∧
  ((1 + Real.sqrt 2) * (2 - Real.sqrt 2) = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l2297_229792


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l2297_229720

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = Real.sqrt 109 ∧
  (∀ (y : ℝ), y > 0 → ⌊y^2⌋ - ⌊y⌋^2 = 19 → y ≥ x) ∧
  ⌊x^2⌋ - ⌊x⌋^2 = 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l2297_229720


namespace NUMINAMATH_CALUDE_buqing_college_students_l2297_229710

/-- Represents the number of students in each college -/
structure CollegeStudents where
  a₁ : ℕ  -- Buqing College
  a₂ : ℕ  -- Jiazhen College
  a₃ : ℕ  -- Hede College
  a₄ : ℕ  -- Wangdao College

/-- Checks if the given numbers form an arithmetic sequence with the specified common difference -/
def isArithmeticSequence (a b c : ℕ) (d : ℕ) : Prop :=
  b = a + d ∧ c = b + d

/-- Checks if the given numbers form a geometric sequence -/
def isGeometricSequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, b = a * r ∧ c = b * r

/-- The main theorem to prove -/
theorem buqing_college_students 
  (s : CollegeStudents) 
  (total : s.a₁ + s.a₂ + s.a₃ + s.a₄ = 474) 
  (arith_seq : isArithmeticSequence s.a₁ s.a₂ s.a₃ 12)
  (geom_seq : isGeometricSequence s.a₁ s.a₃ s.a₄) : 
  s.a₁ = 96 := by
  sorry

end NUMINAMATH_CALUDE_buqing_college_students_l2297_229710


namespace NUMINAMATH_CALUDE_M_subset_P_l2297_229755

def M : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 2) + (Real.pi / 4)}
def P : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 2)}

theorem M_subset_P : M ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_M_subset_P_l2297_229755


namespace NUMINAMATH_CALUDE_two_inequalities_always_true_l2297_229714

theorem two_inequalities_always_true 
  (x y a b : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hxa : x < a) 
  (hyb : y < b) 
  (hx_neg : x < 0) 
  (hy_neg : y < 0) 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) :
  ∃! n : ℕ, n = (Bool.toNat (x + y < a + b)) + 
               (Bool.toNat (x - y < a - b)) + 
               (Bool.toNat (x * y < a * b)) + 
               (Bool.toNat (x / y < a / b)) ∧ 
               n = 2 := by
sorry

end NUMINAMATH_CALUDE_two_inequalities_always_true_l2297_229714


namespace NUMINAMATH_CALUDE_goods_train_length_l2297_229708

/-- The length of a goods train given its speed, platform length, and time to cross the platform. -/
theorem goods_train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  speed = 96 →
  platform_length = 360 →
  crossing_time = 32 →
  let speed_mps := speed * (5 / 18)
  let total_distance := speed_mps * crossing_time
  let train_length := total_distance - platform_length
  train_length = 493.44 := by
  sorry

end NUMINAMATH_CALUDE_goods_train_length_l2297_229708


namespace NUMINAMATH_CALUDE_books_read_l2297_229787

/-- The number of books read in the 'crazy silly school' series -/
theorem books_read (total_books : ℕ) (books_to_read : ℕ) (h1 : total_books = 22) (h2 : books_to_read = 10) :
  total_books - books_to_read = 12 := by
  sorry

#check books_read

end NUMINAMATH_CALUDE_books_read_l2297_229787


namespace NUMINAMATH_CALUDE_green_ducks_percentage_l2297_229779

theorem green_ducks_percentage (smaller_pond : ℕ) (larger_pond : ℕ) 
  (larger_pond_green_percent : ℝ) (total_green_percent : ℝ) :
  smaller_pond = 30 →
  larger_pond = 50 →
  larger_pond_green_percent = 12 →
  total_green_percent = 15 →
  (smaller_pond_green_percent : ℝ) * smaller_pond / 100 + 
    larger_pond_green_percent * larger_pond / 100 = 
    total_green_percent * (smaller_pond + larger_pond) / 100 →
  smaller_pond_green_percent = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_green_ducks_percentage_l2297_229779


namespace NUMINAMATH_CALUDE_temperature_conversion_l2297_229795

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 20 → k = 68 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2297_229795


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2297_229704

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a5 : a 5 = 2) : 
  a 4 - a 5 + a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2297_229704


namespace NUMINAMATH_CALUDE_abs_inequality_l2297_229703

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l2297_229703


namespace NUMINAMATH_CALUDE_least_bench_sections_l2297_229781

/- A single bench section can hold 5 adults or 13 children -/
def adults_per_bench : ℕ := 5
def children_per_bench : ℕ := 13

/- M bench sections are connected end to end -/
def bench_sections (M : ℕ) : ℕ := M

/- An equal number of adults and children are to occupy all benches completely -/
def equal_occupancy (M : ℕ) : Prop :=
  ∃ x : ℕ, x > 0 ∧ adults_per_bench * bench_sections M = x ∧ children_per_bench * bench_sections M = x

/- The least possible positive integer value of M -/
theorem least_bench_sections : 
  ∃ M : ℕ, M > 0 ∧ equal_occupancy M ∧ ∀ N : ℕ, N > 0 → equal_occupancy N → M ≤ N :=
by sorry

end NUMINAMATH_CALUDE_least_bench_sections_l2297_229781


namespace NUMINAMATH_CALUDE_average_score_is_94_l2297_229793

def june_score : ℕ := 97
def patty_score : ℕ := 85
def josh_score : ℕ := 100
def henry_score : ℕ := 94

def total_score : ℕ := june_score + patty_score + josh_score + henry_score
def num_children : ℕ := 4

theorem average_score_is_94 : total_score / num_children = 94 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_94_l2297_229793


namespace NUMINAMATH_CALUDE_symmetry_x_axis_example_l2297_229748

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis in 3D space -/
def symmetry_x_axis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

/-- The theorem stating that the point symmetric to (-2, 1, 5) with respect to the x-axis
    has coordinates (-2, -1, -5) -/
theorem symmetry_x_axis_example : 
  symmetry_x_axis { x := -2, y := 1, z := 5 } = { x := -2, y := -1, z := -5 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_x_axis_example_l2297_229748


namespace NUMINAMATH_CALUDE_total_spider_legs_l2297_229725

/-- The number of legs a single spider has -/
def spider_legs : ℕ := 8

/-- The number of spiders in the group -/
def group_size : ℕ := spider_legs / 2 + 10

/-- The total number of spider legs in the group -/
def total_legs : ℕ := group_size * spider_legs

/-- Theorem stating that the total number of spider legs in the group is 112 -/
theorem total_spider_legs : total_legs = 112 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_l2297_229725


namespace NUMINAMATH_CALUDE_divisibility_condition_l2297_229738

theorem divisibility_condition (n p : ℕ) (h_prime : Nat.Prime p) (h_range : 0 < n ∧ n ≤ 2*p) :
  (n^(p-1) ∣ ((p-1)^n + 1)) ↔ (n = 1 ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2297_229738


namespace NUMINAMATH_CALUDE_inequality_proof_l2297_229797

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2297_229797


namespace NUMINAMATH_CALUDE_tan_sum_thirteen_thirtytwo_l2297_229731

theorem tan_sum_thirteen_thirtytwo : 
  let tan13 := Real.tan (13 * π / 180)
  let tan32 := Real.tan (32 * π / 180)
  tan13 + tan32 + tan13 * tan32 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_thirteen_thirtytwo_l2297_229731


namespace NUMINAMATH_CALUDE_unique_divisible_by_72_l2297_229790

theorem unique_divisible_by_72 : ∃! n : ℕ,
  (n ≥ 1000000000 ∧ n < 10000000000) ∧
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = a * 1000000000 + 20222023 * 10 + b) ∧
  n % 72 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_72_l2297_229790


namespace NUMINAMATH_CALUDE_grocers_sales_problem_l2297_229707

/-- Proof of the grocer's sales problem -/
theorem grocers_sales_problem
  (sales : Fin 6 → ℕ)
  (h1 : sales 0 = 6435)
  (h2 : sales 1 = 6927)
  (h3 : sales 2 = 6855)
  (h5 : sales 4 = 6562)
  (h6 : sales 5 = 7991)
  (avg : (sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = 7000) :
  sales 3 = 7230 := by
  sorry


end NUMINAMATH_CALUDE_grocers_sales_problem_l2297_229707


namespace NUMINAMATH_CALUDE_xiaochun_current_age_l2297_229789

-- Define Xiaochun's current age
def xiaochun_age : ℕ := sorry

-- Define Xiaochun's brother's current age
def brother_age : ℕ := sorry

-- Condition 1: Xiaochun's age is 18 years less than his brother's age
axiom age_difference : xiaochun_age = brother_age - 18

-- Condition 2: In 3 years, Xiaochun's age will be half of his brother's age
axiom future_age_relation : xiaochun_age + 3 = (brother_age + 3) / 2

-- Theorem to prove
theorem xiaochun_current_age : xiaochun_age = 15 := by sorry

end NUMINAMATH_CALUDE_xiaochun_current_age_l2297_229789


namespace NUMINAMATH_CALUDE_g_is_zero_l2297_229774

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (2 * (Real.sin x)^4 + 3 * (Real.cos x)^2) - 
  Real.sqrt (2 * (Real.cos x)^4 + 3 * (Real.sin x)^2)

theorem g_is_zero : ∀ x : ℝ, g x = 0 := by sorry

end NUMINAMATH_CALUDE_g_is_zero_l2297_229774


namespace NUMINAMATH_CALUDE_factorial_simplification_l2297_229733

theorem factorial_simplification : (13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / 
  ((10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) + 3 * (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l2297_229733


namespace NUMINAMATH_CALUDE_det_C_equals_2142_l2297_229782

theorem det_C_equals_2142 (A B C : Matrix (Fin 3) (Fin 3) ℝ) : 
  A = ![![3, 2, 5], ![0, 2, 8], ![4, 1, 7]] →
  B = ![![-2, 3, 4], ![-1, -3, 5], ![0, 4, 3]] →
  C = A * B →
  Matrix.det C = 2142 := by
  sorry

end NUMINAMATH_CALUDE_det_C_equals_2142_l2297_229782


namespace NUMINAMATH_CALUDE_economic_formula_solution_l2297_229702

theorem economic_formula_solution (p x : ℂ) :
  (3 * p - x = 15000) → (x = 9 + 225 * Complex.I) → (p = 5003 + 75 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_economic_formula_solution_l2297_229702


namespace NUMINAMATH_CALUDE_two_common_tangents_range_l2297_229717

/-- Two circles in a 2D plane --/
structure TwoCircles where
  a : ℝ
  c1 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - 2)^2 + y^2 = 4
  c2 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - a)^2 + (y + 3)^2 = 9

/-- The condition for two circles to have exactly two common tangents --/
def has_two_common_tangents (circles : TwoCircles) : Prop :=
  1 < Real.sqrt ((circles.a - 2)^2 + 9) ∧ Real.sqrt ((circles.a - 2)^2 + 9) < 5

/-- Theorem stating the range of 'a' for which the circles have exactly two common tangents --/
theorem two_common_tangents_range (circles : TwoCircles) :
  has_two_common_tangents circles ↔ -2 < circles.a ∧ circles.a < 6 := by
  sorry


end NUMINAMATH_CALUDE_two_common_tangents_range_l2297_229717


namespace NUMINAMATH_CALUDE_no_five_integers_solution_l2297_229771

theorem no_five_integers_solution :
  ¬ ∃ (a b c d e : ℕ),
    (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c < d) ∧ (d < e) ∧
    ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} = 
     {15, 16, 17, 18, 19, 20, 21, 23} ∪ {x | x < 15} ∪ {y | y > 23}) :=
by
  sorry

#check no_five_integers_solution

end NUMINAMATH_CALUDE_no_five_integers_solution_l2297_229771


namespace NUMINAMATH_CALUDE_total_crayons_is_116_l2297_229712

/-- The total number of crayons Wanda, Dina, and Jacob have -/
def total_crayons (wanda_crayons dina_crayons : ℕ) : ℕ :=
  wanda_crayons + dina_crayons + (dina_crayons - 2)

/-- Theorem stating that the total number of crayons is 116 -/
theorem total_crayons_is_116 :
  total_crayons 62 28 = 116 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_is_116_l2297_229712


namespace NUMINAMATH_CALUDE_find_m_l2297_229769

theorem find_m : ∃ m : ℤ, 3^4 - 6 = 5^2 + m ∧ m = 50 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l2297_229769


namespace NUMINAMATH_CALUDE_total_laundry_time_is_344_l2297_229743

/-- Represents a load of laundry with washing and drying times -/
structure LaundryLoad where
  washTime : ℕ
  dryTime : ℕ

/-- Calculates the total time for a single load of laundry -/
def totalLoadTime (load : LaundryLoad) : ℕ :=
  load.washTime + load.dryTime

/-- Calculates the total time for all loads of laundry -/
def totalLaundryTime (loads : List LaundryLoad) : ℕ :=
  loads.map totalLoadTime |>.sum

/-- Theorem: The total laundry time for the given loads is 344 minutes -/
theorem total_laundry_time_is_344 :
  let whites : LaundryLoad := { washTime := 72, dryTime := 50 }
  let darks : LaundryLoad := { washTime := 58, dryTime := 65 }
  let colors : LaundryLoad := { washTime := 45, dryTime := 54 }
  let allLoads : List LaundryLoad := [whites, darks, colors]
  totalLaundryTime allLoads = 344 := by
  sorry


end NUMINAMATH_CALUDE_total_laundry_time_is_344_l2297_229743


namespace NUMINAMATH_CALUDE_t_formula_l2297_229729

theorem t_formula (S₁ S₂ t u : ℝ) (hu : u ≠ 0) (heq : u = (S₁ - S₂) / (t - 1)) :
  t = (S₁ - S₂ + u) / u :=
sorry

end NUMINAMATH_CALUDE_t_formula_l2297_229729


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2297_229791

theorem fraction_equivalence (a1 a2 b1 b2 : ℝ) :
  (∀ x : ℝ, x + a2 ≠ 0 → (x + a1) / (x + a2) = b1 / b2) ↔ (b2 = b1 ∧ b1 * a2 = a1 * b2) :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2297_229791


namespace NUMINAMATH_CALUDE_range_of_x_when_f_leq_1_range_of_m_when_f_minus_g_geq_m_plus_1_l2297_229737

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| - 2
def g (x : ℝ) : ℝ := -|x + 1| + 4

-- Theorem 1: Range of x when f(x) ≤ 1
theorem range_of_x_when_f_leq_1 :
  {x : ℝ | f x ≤ 1} = Set.Icc 0 6 := by sorry

-- Theorem 2: Range of m when f(x) - g(x) ≥ m+1 for all x ∈ ℝ
theorem range_of_m_when_f_minus_g_geq_m_plus_1 :
  {m : ℝ | ∀ x, f x - g x ≥ m + 1} = Set.Iic (-3) := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_f_leq_1_range_of_m_when_f_minus_g_geq_m_plus_1_l2297_229737


namespace NUMINAMATH_CALUDE_black_beads_count_l2297_229766

theorem black_beads_count (white_beads : ℕ) (total_pulled : ℕ) :
  white_beads = 51 →
  total_pulled = 32 →
  ∃ (black_beads : ℕ),
    (1 : ℚ) / 6 * black_beads + (1 : ℚ) / 3 * white_beads = total_pulled ∧
    black_beads = 90 :=
by sorry

end NUMINAMATH_CALUDE_black_beads_count_l2297_229766


namespace NUMINAMATH_CALUDE_problem_1_l2297_229734

theorem problem_1 : -2 - |(-2)| = -4 := by sorry

end NUMINAMATH_CALUDE_problem_1_l2297_229734


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2297_229757

/-- The range of k for an ellipse with equation x^2/4 + y^2/k = 1 and eccentricity e ∈ (1/2, 1) -/
theorem ellipse_k_range (e : ℝ) (h1 : 1/2 < e) (h2 : e < 1) :
  ∀ k : ℝ, (∃ x y : ℝ, x^2/4 + y^2/k = 1 ∧ e^2 = 1 - (min 4 k)/(max 4 k)) ↔
  (k ∈ Set.Ioo 0 3 ∪ Set.Ioi (16/3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2297_229757


namespace NUMINAMATH_CALUDE_melanie_gumball_sale_l2297_229709

/-- Represents the sale of gumballs -/
structure GumballSale where
  price_per_gumball : ℕ
  total_money : ℕ

/-- Calculates the number of gumballs sold -/
def gumballs_sold (sale : GumballSale) : ℕ :=
  sale.total_money / sale.price_per_gumball

/-- Theorem: Melanie sold 4 gumballs -/
theorem melanie_gumball_sale :
  let sale : GumballSale := { price_per_gumball := 8, total_money := 32 }
  gumballs_sold sale = 4 := by
  sorry

end NUMINAMATH_CALUDE_melanie_gumball_sale_l2297_229709


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2297_229722

theorem arithmetic_mean_of_fractions :
  let a := 3 / 5
  let b := 6 / 7
  let c := 4 / 5
  let arithmetic_mean := (a + b) / 2
  (arithmetic_mean ≠ c) ∧ (arithmetic_mean = 51 / 70) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2297_229722


namespace NUMINAMATH_CALUDE_hotdog_problem_l2297_229762

theorem hotdog_problem (h1 h2 : ℕ) : 
  h2 = h1 - 25 → 
  h1 + h2 = 125 → 
  h1 = 75 := by
sorry

end NUMINAMATH_CALUDE_hotdog_problem_l2297_229762


namespace NUMINAMATH_CALUDE_car_value_decrease_l2297_229754

/-- Given a car with an initial value and a value after a certain number of years,
    calculate the annual decrease in value. -/
def annual_decrease (initial_value : ℕ) (final_value : ℕ) (years : ℕ) : ℕ :=
  (initial_value - final_value) / years

theorem car_value_decrease :
  let initial_value : ℕ := 20000
  let final_value : ℕ := 14000
  let years : ℕ := 6
  annual_decrease initial_value final_value years = 1000 := by
  sorry

end NUMINAMATH_CALUDE_car_value_decrease_l2297_229754


namespace NUMINAMATH_CALUDE_total_is_245_l2297_229732

/-- Represents the distribution of money among three parties -/
structure MoneyDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The problem setup -/
def problem_setup (d : MoneyDistribution) : Prop :=
  d.y = 0.45 * d.x ∧ d.z = 0.30 * d.x ∧ d.y = 63

/-- The total amount distributed -/
def total_amount (d : MoneyDistribution) : ℝ :=
  d.x + d.y + d.z

/-- The theorem stating the total amount is 245 rupees -/
theorem total_is_245 (d : MoneyDistribution) (h : problem_setup d) : total_amount d = 245 := by
  sorry


end NUMINAMATH_CALUDE_total_is_245_l2297_229732


namespace NUMINAMATH_CALUDE_isosceles_triangle_m_condition_l2297_229744

/-- Represents an isosceles triangle ABC with side BC of length 8 and sides AB and AC as roots of x^2 - 10x + m = 0 --/
structure IsoscelesTriangle where
  m : ℝ
  ab_ac_roots : ∀ x : ℝ, x^2 - 10*x + m = 0 → (x = ab ∨ x = ac)
  isosceles : ab = ac
  bc_length : bc = 8

/-- The value of m in an isosceles triangle satisfies one of two conditions --/
theorem isosceles_triangle_m_condition (t : IsoscelesTriangle) :
  (∃ x : ℝ, x^2 - 10*x + t.m = 0 ∧ (∀ y : ℝ, y^2 - 10*y + t.m = 0 → y = x)) ∨
  (8^2 - 10*8 + t.m = 0) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_m_condition_l2297_229744


namespace NUMINAMATH_CALUDE_sum_of_five_and_seven_l2297_229786

theorem sum_of_five_and_seven : 5 + 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_and_seven_l2297_229786


namespace NUMINAMATH_CALUDE_rectangle_width_l2297_229775

theorem rectangle_width (square_area : ℝ) (rectangle_length : ℝ) (square_perimeter : ℝ) :
  square_area = 5 * (rectangle_length * 10) ∧
  rectangle_length = 50 ∧
  square_perimeter = 200 ∧
  square_area = (square_perimeter / 4) ^ 2 →
  10 = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l2297_229775


namespace NUMINAMATH_CALUDE_james_recovery_time_l2297_229715

def initial_healing_time : ℝ := 4

def skin_graft_healing_time (t : ℝ) : ℝ := t * 1.5

def total_recovery_time (t : ℝ) : ℝ := t + skin_graft_healing_time t

theorem james_recovery_time :
  total_recovery_time initial_healing_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_recovery_time_l2297_229715


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2297_229726

theorem complex_sum_problem (x y u v w z : ℝ) : 
  y = 2 → 
  w = -x - u → 
  Complex.mk x y + Complex.mk u v + Complex.mk w z = Complex.mk 2 (-1) → 
  v + z = -3 := by sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2297_229726


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2297_229747

theorem inequality_system_solution :
  ∀ x : ℝ, (3 * x - 1 > 2 * (x + 1) ∧ (x + 2) / 3 > x - 2) ↔ (3 < x ∧ x < 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2297_229747


namespace NUMINAMATH_CALUDE_infinite_nested_radical_twenty_l2297_229785

theorem infinite_nested_radical_twenty : ∃! (x : ℝ), x > 0 ∧ x = Real.sqrt (20 + x) ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_infinite_nested_radical_twenty_l2297_229785


namespace NUMINAMATH_CALUDE_cone_base_radius_l2297_229740

/-- The radius of the base of a cone, given its surface area and net shape. -/
theorem cone_base_radius (S : ℝ) (r : ℝ) : 
  S = 9 * Real.pi  -- Surface area condition
  → S = 3 * Real.pi * r^2  -- Surface area formula for a cone
  → r = Real.sqrt 3 :=  -- Conclusion: radius is √3
by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l2297_229740


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l2297_229705

/-- Given a school with the following demographics:
  - Total number of boys: 850
  - 44% are Muslims
  - 28% are Hindus
  - 153 boys belong to other communities
  Prove that 10% of the boys are Sikhs -/
theorem percentage_of_sikh_boys (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (other : ℕ) :
  total = 850 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  other = 153 →
  (total - (muslim_percent * total + hindu_percent * total + other : ℚ)) / total = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l2297_229705


namespace NUMINAMATH_CALUDE_modulus_of_z_l2297_229788

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = 3 + 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2297_229788


namespace NUMINAMATH_CALUDE_store_sales_increase_l2297_229776

/-- Proves that if a store's sales increased by 25% to $400 million, then the previous year's sales were $320 million. -/
theorem store_sales_increase (current_sales : ℝ) (increase_percent : ℝ) :
  current_sales = 400 ∧ increase_percent = 0.25 →
  (1 + increase_percent) * (current_sales / (1 + increase_percent)) = 320 := by
  sorry

end NUMINAMATH_CALUDE_store_sales_increase_l2297_229776


namespace NUMINAMATH_CALUDE_max_value_of_complex_difference_l2297_229780

theorem max_value_of_complex_difference (Z : ℂ) (h : Complex.abs Z = 1) :
  ∃ (max_val : ℝ), max_val = 6 ∧ ∀ (W : ℂ), Complex.abs W = 1 → Complex.abs (W - (3 - 4*I)) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_of_complex_difference_l2297_229780


namespace NUMINAMATH_CALUDE_marble_distribution_l2297_229728

theorem marble_distribution (total_marbles : ℕ) (num_friends : ℕ) (marbles_per_friend : ℕ) :
  total_marbles = 30 →
  num_friends = 5 →
  total_marbles = num_friends * marbles_per_friend →
  marbles_per_friend = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l2297_229728


namespace NUMINAMATH_CALUDE_complex_product_with_i_l2297_229746

theorem complex_product_with_i : (Complex.I * (-1 + 3 * Complex.I)) = (-3 - Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_product_with_i_l2297_229746


namespace NUMINAMATH_CALUDE_single_working_day_between_holidays_l2297_229783

def is_holiday (n : ℕ) : Prop := n % 6 = 0 ∨ Nat.Prime n

def working_day_between_holidays (n : ℕ) : Prop :=
  n > 1 ∧ n < 40 ∧ is_holiday (n - 1) ∧ ¬is_holiday n ∧ is_holiday (n + 1)

theorem single_working_day_between_holidays :
  ∃! n, working_day_between_holidays n :=
sorry

end NUMINAMATH_CALUDE_single_working_day_between_holidays_l2297_229783


namespace NUMINAMATH_CALUDE_least_three_digit_7_heavy_l2297_229716

/-- A number is 7-heavy if its remainder when divided by 7 is greater than 4 -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- A number is three-digit if it's between 100 and 999 inclusive -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_7_heavy : 
  is_three_digit 103 ∧ 
  is_7_heavy 103 ∧ 
  ∀ n : ℕ, is_three_digit n → is_7_heavy n → 103 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_7_heavy_l2297_229716


namespace NUMINAMATH_CALUDE_stack_map_a_front_view_l2297_229739

/-- Represents a column of stacks in the Stack Map --/
def Column := List Nat

/-- Represents the Stack Map A --/
def StackMapA : List Column := [
  [3, 1],       -- First column
  [2, 2, 1],    -- Second column
  [1, 4, 2],    -- Third column
  [5]           -- Fourth column
]

/-- Calculates the front view of a Stack Map --/
def frontView (map : List Column) : List Nat :=
  map.map (List.foldl max 0)

/-- Theorem: The front view of Stack Map A is [3, 2, 4, 5] --/
theorem stack_map_a_front_view :
  frontView StackMapA = [3, 2, 4, 5] := by
  sorry

end NUMINAMATH_CALUDE_stack_map_a_front_view_l2297_229739


namespace NUMINAMATH_CALUDE_conic_eccentricity_l2297_229759

theorem conic_eccentricity (m : ℝ) : 
  (m = Real.sqrt (2 * 8) ∨ m = -Real.sqrt (2 * 8)) →
  let e := if m > 0 
    then Real.sqrt 3 / 2 
    else Real.sqrt 5
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧
    (∀ x y : ℝ, x^2 + y^2/m = 1 ↔ (x/a)^2 + (y/b)^2 = 1) ∧
    e = Real.sqrt (|a^2 - b^2|) / max a b :=
by sorry

end NUMINAMATH_CALUDE_conic_eccentricity_l2297_229759


namespace NUMINAMATH_CALUDE_no_additional_coins_needed_l2297_229723

/-- The minimum number of additional coins needed given the number of friends and initial coins. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let required_coins := num_friends * (num_friends + 1) / 2
  if initial_coins ≥ required_coins then 0
  else required_coins - initial_coins

/-- Theorem stating that for 15 friends and 120 initial coins, no additional coins are needed. -/
theorem no_additional_coins_needed :
  min_additional_coins 15 120 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_additional_coins_needed_l2297_229723


namespace NUMINAMATH_CALUDE_triangle_perimeter_inside_polygon_l2297_229752

-- Define a polygon as a set of points in 2D space
def Polygon : Type := Set (ℝ × ℝ)

-- Define a triangle as a set of three points in 2D space
def Triangle : Type := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

-- Function to check if a triangle is inside a polygon
def isInside (t : Triangle) (p : Polygon) : Prop := sorry

-- Function to calculate the perimeter of a polygon
def perimeterPolygon (p : Polygon) : ℝ := sorry

-- Function to calculate the perimeter of a triangle
def perimeterTriangle (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_perimeter_inside_polygon (t : Triangle) (p : Polygon) :
  isInside t p → perimeterTriangle t ≤ perimeterPolygon p := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_inside_polygon_l2297_229752


namespace NUMINAMATH_CALUDE_marie_erasers_l2297_229778

theorem marie_erasers (initial : ℕ) (lost : ℕ) (final : ℕ) : 
  initial = 95 → lost = 42 → final = initial - lost → final = 53 := by
sorry

end NUMINAMATH_CALUDE_marie_erasers_l2297_229778


namespace NUMINAMATH_CALUDE_sasha_can_get_123_l2297_229701

/-- Represents an arithmetic expression --/
inductive Expr
  | Num : Nat → Expr
  | Add : Expr → Expr → Expr
  | Sub : Expr → Expr → Expr
  | Mul : Expr → Expr → Expr

/-- Evaluates an arithmetic expression --/
def eval : Expr → Int
  | Expr.Num n => n
  | Expr.Add e1 e2 => eval e1 + eval e2
  | Expr.Sub e1 e2 => eval e1 - eval e2
  | Expr.Mul e1 e2 => eval e1 * eval e2

/-- Checks if an expression uses each number from 1 to 5 exactly once --/
def usesAllNumbers : Expr → Bool := sorry

theorem sasha_can_get_123 : ∃ e : Expr, usesAllNumbers e ∧ eval e = 123 := by
  sorry

end NUMINAMATH_CALUDE_sasha_can_get_123_l2297_229701


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2297_229764

/-- Calculates the sample size for a stratified sampling based on gender -/
def calculateSampleSize (totalEmployees : ℕ) (maleEmployees : ℕ) (maleSampleSize : ℕ) : ℕ :=
  (totalEmployees * maleSampleSize) / maleEmployees

/-- Proves that the sample size is 24 given the conditions -/
theorem stratified_sample_size :
  let totalEmployees : ℕ := 120
  let maleEmployees : ℕ := 90
  let maleSampleSize : ℕ := 18
  calculateSampleSize totalEmployees maleEmployees maleSampleSize = 24 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2297_229764


namespace NUMINAMATH_CALUDE_prove_max_value_l2297_229751

def max_value_theorem (a b c : ℝ × ℝ) : Prop :=
  let norm_squared := λ v : ℝ × ℝ => v.1^2 + v.2^2
  norm_squared a = 9 ∧ 
  norm_squared b = 4 ∧ 
  norm_squared c = 16 →
  norm_squared (a.1 - 3*b.1, a.2 - 3*b.2) + 
  norm_squared (b.1 - 3*c.1, b.2 - 3*c.2) + 
  norm_squared (c.1 - 3*a.1, c.2 - 3*a.2) ≤ 428

theorem prove_max_value : ∀ a b c : ℝ × ℝ, max_value_theorem a b c := by
  sorry

end NUMINAMATH_CALUDE_prove_max_value_l2297_229751
