import Mathlib

namespace complex_absolute_value_equation_l1526_152626

theorem complex_absolute_value_equation :
  ∃ (t : ℝ), t > 0 ∧ Complex.abs (-3 + t * Complex.I) = 3 * Real.sqrt 10 ∧ t = 9 := by
  sorry

end complex_absolute_value_equation_l1526_152626


namespace birds_meeting_point_l1526_152615

/-- Theorem: Meeting point of two birds flying towards each other --/
theorem birds_meeting_point 
  (total_distance : ℝ) 
  (speed_bird1 : ℝ) 
  (speed_bird2 : ℝ) 
  (h1 : total_distance = 20)
  (h2 : speed_bird1 = 4)
  (h3 : speed_bird2 = 1) :
  (speed_bird1 * total_distance) / (speed_bird1 + speed_bird2) = 16 := by
  sorry

#check birds_meeting_point

end birds_meeting_point_l1526_152615


namespace parallelepiped_skew_lines_l1526_152635

/-- A parallelepiped is a three-dimensional figure formed by six parallelograms. -/
structure Parallelepiped :=
  (vertices : Finset (Fin 8))

/-- A line in the parallelepiped is defined by two distinct vertices. -/
def Line (p : Parallelepiped) := {l : Fin 8 × Fin 8 // l.1 ≠ l.2}

/-- Two lines are skew if they are not coplanar and do not intersect. -/
def areSkew (p : Parallelepiped) (l1 l2 : Line p) : Prop := sorry

/-- The set of all lines in the parallelepiped. -/
def allLines (p : Parallelepiped) : Finset (Line p) := sorry

/-- The set of all pairs of skew lines in the parallelepiped. -/
def skewLinePairs (p : Parallelepiped) : Finset (Line p × Line p) := sorry

theorem parallelepiped_skew_lines (p : Parallelepiped) :
  (allLines p).card = 28 → (skewLinePairs p).card = 174 := by sorry

end parallelepiped_skew_lines_l1526_152635


namespace social_media_earnings_per_hour_l1526_152634

/-- Calculates the earnings per hour for checking social media posts -/
theorem social_media_earnings_per_hour 
  (payment_per_post : ℝ) 
  (time_per_post : ℝ) 
  (seconds_per_hour : ℝ) 
  (h1 : payment_per_post = 0.25)
  (h2 : time_per_post = 10)
  (h3 : seconds_per_hour = 3600) :
  (payment_per_post * (seconds_per_hour / time_per_post)) = 90 := by
  sorry

end social_media_earnings_per_hour_l1526_152634


namespace union_A_B_when_a_is_one_intersection_A_B_empty_iff_a_leq_neg_three_or_geq_three_l1526_152687

-- Define set A
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 2}

-- Define set B (domain of the function)
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem 1
theorem union_A_B_when_a_is_one :
  A 1 ∪ B = {x | -1 < x ∧ x < 3} := by sorry

-- Theorem 2
theorem intersection_A_B_empty_iff_a_leq_neg_three_or_geq_three (a : ℝ) :
  A a ∩ B = ∅ ↔ a ≤ -3 ∨ a ≥ 3 := by sorry

end union_A_B_when_a_is_one_intersection_A_B_empty_iff_a_leq_neg_three_or_geq_three_l1526_152687


namespace basketball_tryouts_l1526_152649

theorem basketball_tryouts (girls boys callback : ℕ) 
  (h1 : girls = 17)
  (h2 : boys = 32)
  (h3 : callback = 10) :
  girls + boys - callback = 39 := by
sorry

end basketball_tryouts_l1526_152649


namespace empty_boxes_count_l1526_152601

/-- The number of empty boxes after n operations, where n is the number of non-empty boxes. -/
def empty_boxes (n : ℕ) : ℤ :=
  -1 + 6 * n

/-- The theorem stating that when there are 34 non-empty boxes, there are 203 empty boxes. -/
theorem empty_boxes_count : empty_boxes 34 = 203 := by
  sorry

end empty_boxes_count_l1526_152601


namespace fixed_term_deposit_result_l1526_152651

/-- Calculates the total amount after a fixed term deposit -/
def totalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

/-- Proves that the total amount after the fixed term deposit is 21998 yuan -/
theorem fixed_term_deposit_result : 
  let principal : ℝ := 20000
  let rate : ℝ := 0.0333
  let time : ℝ := 3
  totalAmount principal rate time = 21998 := by
sorry


end fixed_term_deposit_result_l1526_152651


namespace sector_arc_length_l1526_152664

/-- Given a circular sector with a central angle of 150° and a radius of 6 cm,
    the arc length is 5π cm. -/
theorem sector_arc_length :
  let θ : ℝ := 150  -- Central angle in degrees
  let r : ℝ := 6    -- Radius in cm
  let L : ℝ := (θ / 360) * (2 * Real.pi * r)  -- Arc length formula
  L = 5 * Real.pi := by
  sorry

end sector_arc_length_l1526_152664


namespace floor_ceil_abs_difference_l1526_152631

theorem floor_ceil_abs_difference : |⌊(-5.67 : ℝ)⌋| - ⌈(42.1 : ℝ)⌉ = -37 := by
  sorry

end floor_ceil_abs_difference_l1526_152631


namespace p_sufficient_not_necessary_for_q_l1526_152624

def p (x y : ℝ) : Prop := (x - 2) * (y - 5) ≠ 0

def q (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 5

theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, p x y → q x y) ∧ 
  (∃ x y : ℝ, q x y ∧ ¬(p x y)) :=
sorry

end p_sufficient_not_necessary_for_q_l1526_152624


namespace arithmetic_square_root_of_four_l1526_152629

theorem arithmetic_square_root_of_four : Real.sqrt 4 = 2 := by
  sorry

end arithmetic_square_root_of_four_l1526_152629


namespace equation_solutions_count_l1526_152697

theorem equation_solutions_count : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 - 5)^2 = 36) ∧ s.card = 2 := by
  sorry

end equation_solutions_count_l1526_152697


namespace sum_of_consecutive_integers_l1526_152663

theorem sum_of_consecutive_integers (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 22 → n = 4 := by
  sorry

end sum_of_consecutive_integers_l1526_152663


namespace fraction_equality_l1526_152673

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h : (x^2 + 4*x*y) / (y^2 - 4*x*y) = 3) :
  ∃ z : ℝ, (x^2 - 4*x*y) / (y^2 + 4*x*y) = z :=
by sorry

end fraction_equality_l1526_152673


namespace B_set_given_A_l1526_152618

def f (a b x : ℝ) : ℝ := x^2 + a*x + b

def A (a b : ℝ) : Set ℝ := {x | f a b x = x}

def B (a b : ℝ) : Set ℝ := {x | f a b (f a b x) = x}

theorem B_set_given_A (a b : ℝ) :
  A a b = {-1, 3} → B a b = {-1, Real.sqrt 3, -Real.sqrt 3, 3} := by
  sorry

end B_set_given_A_l1526_152618


namespace sum_N_equals_2250_l1526_152638

def N : ℕ → ℤ
  | 0 => 0
  | n + 1 => let k := 40 - n
              if n % 2 = 0 then
                N n + (3*k)^2 + (3*k-1)^2 + (3*k-2)^2
              else
                N n - (3*k)^2 - (3*k-1)^2 - (3*k-2)^2

theorem sum_N_equals_2250 : N 40 = 2250 := by
  sorry

end sum_N_equals_2250_l1526_152638


namespace unique_solution_x4_y2_71_l1526_152654

theorem unique_solution_x4_y2_71 :
  ∀ x y : ℕ+, x^4 = y^2 + 71 → x = 6 ∧ y = 35 := by
  sorry

end unique_solution_x4_y2_71_l1526_152654


namespace tan_three_properties_l1526_152625

theorem tan_three_properties (θ : Real) (h : Real.tan θ = 3) :
  (Real.cos θ / (Real.sin θ + 2 * Real.cos θ) = 1/5) ∧
  (Real.tan (θ - 5 * Real.pi / 4) = 1/2) := by
  sorry

end tan_three_properties_l1526_152625


namespace polynomial_root_l1526_152620

/-- Given a polynomial g(x) = 3x^4 - 2x^3 + x^2 + 4x + s, 
    prove that s = -2 when g(-1) = 0 -/
theorem polynomial_root (s : ℝ) : 
  (fun x : ℝ => 3 * x^4 - 2 * x^3 + x^2 + 4 * x + s) (-1) = 0 ↔ s = -2 := by
  sorry

end polynomial_root_l1526_152620


namespace sum_of_roots_l1526_152611

theorem sum_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y : ℝ, (y + 3) * (y - 4) = 22 ∧ x + y = 1 := by
  sorry

end sum_of_roots_l1526_152611


namespace arithmetic_sequence_8th_term_l1526_152659

/-- Given an arithmetic sequence with first term 2 and common difference 3,
    prove that the 8th term is 23. -/
theorem arithmetic_sequence_8th_term :
  let a : ℕ → ℤ := λ n => 2 + 3 * (n - 1)
  a 8 = 23 := by sorry

end arithmetic_sequence_8th_term_l1526_152659


namespace max_value_of_f_on_interval_l1526_152646

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (M : ℝ), M = 4 ∧ ∀ x ∈ Set.Icc (-1) 2, f x ≤ M :=
by
  sorry

end max_value_of_f_on_interval_l1526_152646


namespace hyperbola_eccentricity_l1526_152656

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  let asymptote_slope := b / a
  (e = 2 * asymptote_slope) → e = 2 * Real.sqrt 3 / 3 :=
by sorry

end hyperbola_eccentricity_l1526_152656


namespace actual_payment_calculation_l1526_152692

/-- Represents the restaurant's voucher system and discount policy -/
structure Restaurant where
  voucher_cost : ℕ := 25
  voucher_value : ℕ := 50
  max_vouchers : ℕ := 3
  hotpot_base_cost : ℕ := 50
  other_dishes_discount : ℚ := 0.4

/-- Represents a family's dining experience -/
structure DiningExperience where
  restaurant : Restaurant
  total_bill : ℕ
  voucher_savings : ℕ
  onsite_discount_savings : ℕ

/-- The theorem to be proved -/
theorem actual_payment_calculation (d : DiningExperience) :
  d.restaurant.hotpot_base_cost = 50 ∧
  d.restaurant.voucher_cost = 25 ∧
  d.restaurant.voucher_value = 50 ∧
  d.restaurant.max_vouchers = 3 ∧
  d.restaurant.other_dishes_discount = 0.4 ∧
  d.onsite_discount_savings = d.voucher_savings + 15 →
  d.total_bill - d.onsite_discount_savings = 185 := by
  sorry


end actual_payment_calculation_l1526_152692


namespace function_increasing_implies_a_leq_one_l1526_152684

/-- Given a function f(x) = e^(|x-a|), where a is a constant,
    if f(x) is increasing on [1, +∞), then a ≤ 1 -/
theorem function_increasing_implies_a_leq_one (a : ℝ) :
  (∀ x y, 1 ≤ x ∧ x < y → (Real.exp (|x - a|) < Real.exp (|y - a|))) →
  a ≤ 1 := by
  sorry

end function_increasing_implies_a_leq_one_l1526_152684


namespace product_equals_120_l1526_152645

theorem product_equals_120 (n : ℕ) (h : n = 3) : (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 := by
  sorry

end product_equals_120_l1526_152645


namespace mixture_volume_calculation_l1526_152616

/-- Proves that given a mixture with 20% water content, if adding 8.333333333333334 gallons
    of water increases the water percentage to 25%, then the initial volume of the mixture
    is 125 gallons. -/
theorem mixture_volume_calculation (initial_volume : ℝ) : 
  (0.20 * initial_volume + 8.333333333333334) / (initial_volume + 8.333333333333334) = 0.25 →
  initial_volume = 125 := by
  sorry

end mixture_volume_calculation_l1526_152616


namespace intersection_point_l1526_152600

theorem intersection_point (a : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = 2*a ∧ p.2 = |p.1 - a| - 1) → a = -1/2 := by
sorry

end intersection_point_l1526_152600


namespace bottles_per_crate_l1526_152662

theorem bottles_per_crate 
  (total_bottles : ℕ) 
  (num_crates : ℕ) 
  (unpacked_bottles : ℕ) 
  (h1 : total_bottles = 130) 
  (h2 : num_crates = 10) 
  (h3 : unpacked_bottles = 10) 
  : (total_bottles - unpacked_bottles) / num_crates = 12 := by
  sorry

end bottles_per_crate_l1526_152662


namespace zero_not_in_2_16_l1526_152655

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of having only one zero
def has_unique_zero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

-- Define the property of a zero being within an interval
def zero_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ f x = 0

-- Theorem statement
theorem zero_not_in_2_16 (h1 : has_unique_zero f)
  (h2 : zero_in_interval f 0 16)
  (h3 : zero_in_interval f 0 8)
  (h4 : zero_in_interval f 0 4)
  (h5 : zero_in_interval f 0 2) :
  ¬∃ x, 2 < x ∧ x < 16 ∧ f x = 0 :=
by sorry

end zero_not_in_2_16_l1526_152655


namespace expenditure_difference_l1526_152630

theorem expenditure_difference
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_increase_percent : ℝ)
  (quantity_purchased_percent : ℝ)
  (h1 : price_increase_percent = 25)
  (h2 : quantity_purchased_percent = 70)
  : abs (original_price * original_quantity * (1 + price_increase_percent / 100) * (quantity_purchased_percent / 100) - original_price * original_quantity) / (original_price * original_quantity) = 0.125 := by
sorry

end expenditure_difference_l1526_152630


namespace distance_from_origin_to_point_l1526_152693

theorem distance_from_origin_to_point : ∀ (x y : ℝ), 
  x = 12 ∧ y = 9 → Real.sqrt (x^2 + y^2) = 15 := by
  sorry

end distance_from_origin_to_point_l1526_152693


namespace divisibility_condition_l1526_152644

theorem divisibility_condition (n : ℕ) : (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 := by
  sorry

end divisibility_condition_l1526_152644


namespace april_order_proof_l1526_152690

/-- The number of cases of soda ordered in April -/
def april_cases : ℕ := sorry

/-- The number of cases of soda ordered in May -/
def may_cases : ℕ := 30

/-- The number of bottles per case -/
def bottles_per_case : ℕ := 20

/-- The total number of bottles ordered in April and May -/
def total_bottles : ℕ := 1000

theorem april_order_proof :
  april_cases = 20 ∧
  april_cases * bottles_per_case + may_cases * bottles_per_case = total_bottles :=
by sorry

end april_order_proof_l1526_152690


namespace inscribed_rectangle_width_l1526_152698

/-- Given a right-angled triangle with legs a and b, and a rectangle inscribed
    such that its width d satisfies d(d - (a + b)) = 0, prove that d = a + b -/
theorem inscribed_rectangle_width (a b d : ℝ) (h : d * (d - (a + b)) = 0) :
  d = a + b := by
  sorry

end inscribed_rectangle_width_l1526_152698


namespace signal_arrangements_l1526_152647

def num_red_flags : ℕ := 3
def num_white_flags : ℕ := 2
def total_flags : ℕ := num_red_flags + num_white_flags

theorem signal_arrangements : (total_flags.choose num_red_flags) = 10 := by
  sorry

end signal_arrangements_l1526_152647


namespace floor_sqrt_50_squared_l1526_152608

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end floor_sqrt_50_squared_l1526_152608


namespace inequality_solution_l1526_152606

theorem inequality_solution (x : ℝ) : 
  (x^2 / (x + 1) ≥ x^2 / (x - 1) + 7/6) ↔ 
  (x < (-1 - Real.sqrt 5) / 2 ∨ 
   (-1 < x ∧ x < 1) ∨ 
   x > (-1 + Real.sqrt 5) / 2) :=
by sorry

end inequality_solution_l1526_152606


namespace students_in_both_games_l1526_152637

theorem students_in_both_games (total : ℕ) (game_a : ℕ) (game_b : ℕ) 
  (h_total : total = 55) (h_game_a : game_a = 38) (h_game_b : game_b = 42) :
  ∃ x : ℕ, x = game_a + game_b - total ∧ x = 25 := by
  sorry

end students_in_both_games_l1526_152637


namespace pears_eaten_by_mike_l1526_152641

theorem pears_eaten_by_mike (jason_pears keith_pears remaining_pears : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : remaining_pears = 81) :
  jason_pears + keith_pears - remaining_pears = 12 := by
  sorry

end pears_eaten_by_mike_l1526_152641


namespace circle_tangent_to_x_axis_l1526_152689

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define what it means for a circle to be tangent to the x-axis
def tangentToXAxis (c : Circle) : Prop :=
  c.center.2 = c.radius

-- Define the equation of a circle
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_tangent_to_x_axis :
  ∀ (c : Circle),
    c.center = (5, 4) →
    tangentToXAxis c →
    ∀ (x y : ℝ), circleEquation c x y ↔ (x - 5)^2 + (y - 4)^2 = 16 :=
by
  sorry


end circle_tangent_to_x_axis_l1526_152689


namespace g_x_minus_3_l1526_152683

/-- The function g(x) = x^2 -/
def g (x : ℝ) : ℝ := x^2

/-- Theorem: For the function g(x) = x^2, g(x-3) = x^2 - 6x + 9 -/
theorem g_x_minus_3 (x : ℝ) : g (x - 3) = x^2 - 6*x + 9 := by
  sorry

end g_x_minus_3_l1526_152683


namespace pirate_gold_distribution_l1526_152671

theorem pirate_gold_distribution (total : ℕ) (jack jimmy tom sanji : ℕ) : 
  total = 280 ∧ 
  jimmy = jack + 11 ∧ 
  tom = jack - 15 ∧ 
  sanji = jack + 20 ∧ 
  total = jack + jimmy + tom + sanji → 
  sanji = 86 := by
sorry

end pirate_gold_distribution_l1526_152671


namespace algebraic_expression_value_l1526_152603

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 2 * Real.sqrt 3) :
  ((a^2 + b^2) / (2 * a) - b) * (a / (a - b)) = Real.sqrt 3 := by
  sorry

end algebraic_expression_value_l1526_152603


namespace kyle_practice_time_l1526_152695

/-- Kyle's daily basketball practice schedule -/
def KylePractice : Prop :=
  ∃ (total_time shooting_time running_time weightlifting_time : ℕ),
    -- Total practice time
    total_time = shooting_time + running_time + weightlifting_time
    -- Half time spent shooting
    ∧ 2 * shooting_time = total_time
    -- Running time is twice weightlifting time
    ∧ running_time = 2 * weightlifting_time
    -- Weightlifting time is 20 minutes
    ∧ weightlifting_time = 20
    -- Total time in hours is 2
    ∧ total_time = 120

/-- Theorem: Kyle's daily basketball practice is 2 hours -/
theorem kyle_practice_time : KylePractice := by
  sorry

end kyle_practice_time_l1526_152695


namespace ellipse_dot_product_l1526_152694

/-- An ellipse with given properties and a line intersecting it -/
structure EllipseWithLine where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (1 - b^2 / a^2).sqrt = Real.sqrt 2 / 2
  h_point : b^2 = 1
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ
  h_A : A.1 = -a ∧ A.2 = 0
  h_B : B.1^2 / a^2 + B.2^2 / b^2 = 1
  h_P : P.1 = a
  h_collinear : ∃ (t : ℝ), B = A + t • (P - A)

/-- The dot product of OB and OP is 2 -/
theorem ellipse_dot_product (e : EllipseWithLine) : e.B.1 * e.P.1 + e.B.2 * e.P.2 = 2 := by
  sorry

end ellipse_dot_product_l1526_152694


namespace min_value_implies_a_l1526_152622

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.cos x + (5/8) * a - (3/2)

theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f a x = 2) →
  a = 4 := by sorry

end min_value_implies_a_l1526_152622


namespace derivative_f_at_1_l1526_152667

def f (x : ℝ) : ℝ := x^2 + 2

theorem derivative_f_at_1 : 
  deriv f 1 = 2 := by sorry

end derivative_f_at_1_l1526_152667


namespace quadratic_roots_average_l1526_152632

theorem quadratic_roots_average (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 - 4 * a * x + b
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 →
  (x + y) / 2 = 2 := by
sorry

end quadratic_roots_average_l1526_152632


namespace equation_has_four_real_solutions_l1526_152612

theorem equation_has_four_real_solutions :
  let f : ℝ → ℝ := λ x => 6*x/(x^2 + x + 1) + 7*x/(x^2 - 7*x + 2) + 5/2
  (∃ (a b c d : ℝ), (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)) ∧
  (∀ (w x y z : ℝ), (∀ r, f r = 0 ↔ r = w ∨ r = x ∨ r = y ∨ r = z) →
    w = a ∧ x = b ∧ y = c ∧ z = d) :=
by sorry

end equation_has_four_real_solutions_l1526_152612


namespace tangent_line_equation_l1526_152633

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 2

-- Define the point of tangency
def x₀ : ℝ := 1
def y₀ : ℝ := 6

-- Define the slope of the tangent line
def m : ℝ := 9

-- Statement to prove
theorem tangent_line_equation :
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (9*x - y - 3 = 0) :=
by sorry

end tangent_line_equation_l1526_152633


namespace x_one_minus_f_equals_one_l1526_152650

theorem x_one_minus_f_equals_one :
  let x : ℝ := (3 + Real.sqrt 8) ^ 100
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by
sorry

end x_one_minus_f_equals_one_l1526_152650


namespace tangent_perpendicular_implies_a_solutions_of_g_eq_zero_l1526_152665

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -2 * x * Real.log x + x^2 - 2 * a * x + a^2

-- Define the derivative of f as g
def g (a : ℝ) (x : ℝ) : ℝ := -2 * (1 + Real.log x) + 2 * x - 2 * a

theorem tangent_perpendicular_implies_a (a : ℝ) :
  (g a 1 = -1) → a = -1/2 := by sorry

theorem solutions_of_g_eq_zero (a : ℝ) :
  (a < 0 → ∀ x, g a x ≠ 0) ∧
  (a = 0 → ∃! x, g a x = 0) ∧
  (a > 0 → ∃ x y, x ≠ y ∧ g a x = 0 ∧ g a y = 0) := by sorry

end

end tangent_perpendicular_implies_a_solutions_of_g_eq_zero_l1526_152665


namespace greatest_prime_factor_of_5_pow_5_plus_12_pow_3_l1526_152610

theorem greatest_prime_factor_of_5_pow_5_plus_12_pow_3 :
  (Nat.factors (5^5 + 12^3)).maximum? = some 19 := by
  sorry

end greatest_prime_factor_of_5_pow_5_plus_12_pow_3_l1526_152610


namespace polynomial_simplification_l1526_152681

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + 5 * y^9 + 3 * y^8) =
  15 * y^13 - y^12 + 3 * y^11 + 15 * y^10 - y^9 - 6 * y^8 := by
  sorry

end polynomial_simplification_l1526_152681


namespace ludvik_favorite_number_l1526_152607

/-- Ludvík's favorite number problem -/
theorem ludvik_favorite_number 
  (a b : ℝ) -- original dividend and divisor
  (h1 : (2 * a) / (b + 12) = (a - 42) / (b / 2)) -- equality of the two scenarios
  (h2 : (2 * a) / (b + 12) > 0) -- ensure the result is positive
  : (2 * a) / (b + 12) = 7 := by
  sorry

end ludvik_favorite_number_l1526_152607


namespace game_outcomes_l1526_152680

/-- The game state -/
inductive GameState
| A (n : ℕ)  -- Player A's turn with current number n
| B (n : ℕ)  -- Player B's turn with current number n

/-- The possible outcomes of the game -/
inductive Outcome
| AWin  -- Player A wins
| BWin  -- Player B wins
| Draw  -- Neither player has a winning strategy

/-- Definition of a winning strategy for a player -/
def has_winning_strategy (player : GameState → Prop) (s : GameState) : Prop :=
  ∃ (strategy : GameState → ℕ), 
    ∀ (game : ℕ → GameState),
      game 0 = s →
      (∀ n, player (game n) → game (n + 1) = GameState.B (strategy (game n))) →
      (∃ m, game m = GameState.A 1990 ∨ game m = GameState.B 1)

/-- The main theorem about the game outcomes -/
theorem game_outcomes (n₀ : ℕ) : 
  (has_winning_strategy (λ s => ∃ n, s = GameState.A n) (GameState.A n₀) ↔ n₀ ≥ 8) ∧
  (has_winning_strategy (λ s => ∃ n, s = GameState.B n) (GameState.A n₀) ↔ n₀ ≤ 5) ∧
  (¬ has_winning_strategy (λ s => ∃ n, s = GameState.A n) (GameState.A n₀) ∧
   ¬ has_winning_strategy (λ s => ∃ n, s = GameState.B n) (GameState.A n₀) ↔ n₀ = 6 ∨ n₀ = 7) :=
sorry

end game_outcomes_l1526_152680


namespace square_plus_reciprocal_square_l1526_152675

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end square_plus_reciprocal_square_l1526_152675


namespace seventh_group_draw_l1526_152677

/-- Represents the systematic sampling method for a population -/
structure SystematicSampling where
  populationSize : Nat
  groupCount : Nat
  sampleSize : Nat
  firstDrawn : Nat

/-- Calculates the number drawn in a specific group -/
def SystematicSampling.numberDrawnInGroup (s : SystematicSampling) (groupNumber : Nat) : Nat :=
  let groupSize := s.populationSize / s.groupCount
  let baseNumber := (groupNumber - 1) * groupSize
  baseNumber + (s.firstDrawn + groupNumber - 1) % 10

theorem seventh_group_draw (s : SystematicSampling) 
  (h1 : s.populationSize = 100)
  (h2 : s.groupCount = 10)
  (h3 : s.sampleSize = 10)
  (h4 : s.firstDrawn = 6) :
  s.numberDrawnInGroup 7 = 63 := by
  sorry

#check seventh_group_draw

end seventh_group_draw_l1526_152677


namespace coffee_calculation_l1526_152660

/-- Calculates the total tablespoons of coffee needed for guests with different preferences -/
def total_coffee_tablespoons (total_guests : ℕ) : ℕ :=
  let weak_guests := total_guests / 3
  let medium_guests := total_guests / 3
  let strong_guests := total_guests - (weak_guests + medium_guests)
  let weak_cups := weak_guests * 2
  let medium_cups := medium_guests * 3
  let strong_cups := strong_guests * 1
  let weak_tablespoons := weak_cups * 1
  let medium_tablespoons := (medium_cups * 3) / 2
  let strong_tablespoons := strong_cups * 2
  weak_tablespoons + medium_tablespoons + strong_tablespoons

theorem coffee_calculation :
  total_coffee_tablespoons 18 = 51 := by
  sorry

end coffee_calculation_l1526_152660


namespace inscribed_squares_ratio_l1526_152619

/-- Given a right triangle with sides 5, 12, and 13, let x be the side length of a square
    inscribed with one vertex at the right angle, and y be the side length of a square
    inscribed with one side on a leg of the triangle. Then x/y = 20/17. -/
theorem inscribed_squares_ratio (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧
  x^2 + (12 - x)^2 = 13^2 ∧
  x^2 + (5 - x)^2 = 12^2 ∧
  y^2 + (5 - y)^2 = (12 - y)^2 →
  x / y = 20 / 17 := by
sorry

end inscribed_squares_ratio_l1526_152619


namespace initial_distance_proof_l1526_152685

/-- The initial distance between Fred and Sam -/
def initial_distance : ℝ := 35

/-- Fred's walking speed in miles per hour -/
def fred_speed : ℝ := 2

/-- Sam's walking speed in miles per hour -/
def sam_speed : ℝ := 5

/-- The distance Sam walks before they meet -/
def sam_distance : ℝ := 25

theorem initial_distance_proof :
  initial_distance = sam_distance + (sam_distance * fred_speed) / sam_speed :=
by sorry

end initial_distance_proof_l1526_152685


namespace integer_solutions_of_equation_l1526_152666

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, x^4 = y^2 + 2*y + 2 ↔ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) := by
  sorry

end integer_solutions_of_equation_l1526_152666


namespace gcd_problem_l1526_152686

theorem gcd_problem (b : ℤ) (h : 345 ∣ b) :
  Nat.gcd (5*b^3 + 2*b^2 + 7*b + 69).natAbs b.natAbs = 69 := by
  sorry

end gcd_problem_l1526_152686


namespace expression_simplification_l1526_152636

theorem expression_simplification (a : ℝ) (h : a = 2) : 
  (1 / (a + 1) - (a + 2) / (a^2 - 1) * (a^2 - 2*a + 1) / (a^2 + 4*a + 4)) * (a + 2) = 1 := by
  sorry

end expression_simplification_l1526_152636


namespace probability_two_red_balls_l1526_152604

/-- The probability of selecting two red balls from a bag containing 6 red, 5 blue, and 2 green balls, when 2 balls are picked at random. -/
theorem probability_two_red_balls (red blue green : ℕ) (h1 : red = 6) (h2 : blue = 5) (h3 : green = 2) :
  let total := red + blue + green
  (Nat.choose red 2 : ℚ) / (Nat.choose total 2) = 5 / 26 := by
  sorry

end probability_two_red_balls_l1526_152604


namespace initial_number_of_men_l1526_152623

/-- Given a group of men where replacing two men (aged 20 and 22) with two women (average age 29)
    increases the average age by 2 years, prove that the initial number of men is 8. -/
theorem initial_number_of_men (M : ℕ) (A : ℝ) : 
  (2 * 29 - (20 + 22) : ℝ) = 2 * M → M = 8 := by
  sorry

end initial_number_of_men_l1526_152623


namespace circumcircle_area_of_special_triangle_l1526_152679

/-- Given a triangle ABC with sides a, b, c, area S, where a² + b² - c² = 4√3 * S and c = 1,
    the area of its circumcircle is π. -/
theorem circumcircle_area_of_special_triangle (a b c S : ℝ) : 
  a > 0 → b > 0 → c > 0 → S > 0 →
  a^2 + b^2 - c^2 = 4 * Real.sqrt 3 * S →
  c = 1 →
  ∃ (R : ℝ), R > 0 ∧ π * R^2 = π := by sorry

end circumcircle_area_of_special_triangle_l1526_152679


namespace smallest_n_multiple_of_seven_l1526_152674

theorem smallest_n_multiple_of_seven (x y : ℤ) 
  (hx : 7 ∣ (x + 2)) 
  (hy : 7 ∣ (y - 2)) : 
  (∃ n : ℕ+, 7 ∣ (x^2 - x*y + y^2 + n) ∧ 
    ∀ m : ℕ+, 7 ∣ (x^2 - x*y + y^2 + m) → n ≤ m) ∧
  (7 ∣ (x^2 - x*y + y^2 + 2)) :=
by sorry

end smallest_n_multiple_of_seven_l1526_152674


namespace max_n_geometric_sequence_l1526_152691

-- Define the geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem max_n_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  a 2 * a 4 = 4 →  -- a₂ · a₄ = 4
  a 1 + a 2 + a 3 = 14 →  -- a₁ + a₂ + a₃ = 14
  (∃ a₁ q, ∀ n, a n = geometric_sequence a₁ q n) →  -- {a_n} is a geometric sequence
  (∀ n > 4, a n * a (n+1) * a (n+2) ≤ 1/9) ∧  -- For all n > 4, the product is ≤ 1/9
  (a 4 * a 5 * a 6 > 1/9) :=  -- For n = 4, the product is > 1/9
by sorry

end max_n_geometric_sequence_l1526_152691


namespace subset_implies_a_equals_one_l1526_152605

theorem subset_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {-1, 0, a}
  let B : Set ℝ := {0, Real.sqrt a}
  B ⊆ A → a = 1 := by
sorry

end subset_implies_a_equals_one_l1526_152605


namespace polygon_sides_count_l1526_152658

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (2 * 360 : ℝ) = ((n - 2) * 180 : ℝ) →
  n = 6 := by
  sorry

end polygon_sides_count_l1526_152658


namespace g_one_value_l1526_152642

-- Define the polynomial f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the conditions
structure Conditions (a b c : ℝ) : Prop :=
  (a_lt_b : a < b)
  (b_lt_c : b < c)
  (one_lt_a : 1 < a)

-- Define the theorem
theorem g_one_value (a b c : ℝ) (h : Conditions a b c) :
  ∃ g : ℝ → ℝ,
    (∀ x, g x = 0 ↔ ∃ y, f a b c y = 0 ∧ x = 1 / y) →
    (∃ k, k ≠ 0 ∧ ∀ x, g x = k * (x^3 + (c/k)*x^2 + (b/k)*x + a/k)) →
    g 1 = (1 + a + b + c) / c :=
sorry

end g_one_value_l1526_152642


namespace smallest_n_square_and_fifth_power_l1526_152639

theorem smallest_n_square_and_fifth_power :
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (l : ℕ), 5 * n = l^5) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (k : ℕ), 4 * m = k^2) → 
    (∃ (l : ℕ), 5 * m = l^5) → 
    m ≥ 625) ∧
  n = 625 :=
sorry

end smallest_n_square_and_fifth_power_l1526_152639


namespace quadratic_points_property_l1526_152621

/-- Represents a quadratic function y = ax² - 4ax + c, where a > 0 -/
structure QuadraticFunction where
  a : ℝ
  c : ℝ
  h_a_pos : a > 0

/-- Represents the y-coordinates of the four points on the quadratic function -/
structure FourPoints where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  y₄ : ℝ

/-- 
  Theorem: For a quadratic function y = ax² - 4ax + c (a > 0) passing through points 
  A(-2, y₁), B(0, y₂), C(3, y₃), D(5, y₄), if y₂y₄ < 0, then y₁y₃ < 0.
-/
theorem quadratic_points_property (f : QuadraticFunction) (p : FourPoints) :
  (p.y₂ * p.y₄ < 0) → (p.y₁ * p.y₃ < 0) := by
  sorry

end quadratic_points_property_l1526_152621


namespace total_tickets_used_l1526_152678

/-- The cost of the shooting game in tickets -/
def shooting_game_cost : ℕ := 5

/-- The cost of the carousel in tickets -/
def carousel_cost : ℕ := 3

/-- The number of times Jen played the shooting game -/
def jen_games : ℕ := 2

/-- The number of times Russel rode the carousel -/
def russel_rides : ℕ := 3

/-- Theorem stating the total number of tickets used -/
theorem total_tickets_used : 
  shooting_game_cost * jen_games + carousel_cost * russel_rides = 19 := by
  sorry

end total_tickets_used_l1526_152678


namespace johns_candy_store_spending_l1526_152643

theorem johns_candy_store_spending (allowance : ℚ) (arcade_fraction : ℚ) (toy_fraction : ℚ) :
  allowance = 3.60 ∧ 
  arcade_fraction = 3/5 ∧ 
  toy_fraction = 1/3 →
  allowance * (1 - arcade_fraction) * (1 - toy_fraction) = 0.96 := by
  sorry

end johns_candy_store_spending_l1526_152643


namespace mammaad_arrangements_l1526_152672

theorem mammaad_arrangements : 
  let total_letters : ℕ := 7
  let m_count : ℕ := 3
  let a_count : ℕ := 3
  let d_count : ℕ := 1
  (total_letters.factorial) / (m_count.factorial * a_count.factorial * d_count.factorial) = 140 := by
  sorry

end mammaad_arrangements_l1526_152672


namespace root_of_cubic_polynomials_l1526_152670

theorem root_of_cubic_polynomials (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hk1 : a * k^3 + b * k^2 + c * k + d = 0)
  (hk2 : b * k^3 + c * k^2 + d * k + a = 0) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
by sorry

end root_of_cubic_polynomials_l1526_152670


namespace meeting_day_is_thursday_l1526_152669

-- Define the days of the week
inductive Day : Type
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

-- Define a function to determine if Joãozinho lies on a given day
def lies_on_day (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Thursday ∨ d = Day.Saturday

-- Define a function to get the next day
def next_day (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

-- Theorem statement
theorem meeting_day_is_thursday :
  ∀ (d : Day),
    lies_on_day d →
    (lies_on_day d → d ≠ Day.Saturday) →
    (lies_on_day d → next_day d ≠ Day.Wednesday) →
    d = Day.Thursday :=
by
  sorry


end meeting_day_is_thursday_l1526_152669


namespace gcd_of_75_and_100_l1526_152627

theorem gcd_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end gcd_of_75_and_100_l1526_152627


namespace robin_albums_l1526_152696

theorem robin_albums (total_pictures : ℕ) (pictures_per_album : ℕ) (h1 : total_pictures = 40) (h2 : pictures_per_album = 8) : total_pictures / pictures_per_album = 5 := by
  sorry

end robin_albums_l1526_152696


namespace second_street_sales_l1526_152668

/-- Represents the sales data for a door-to-door salesman selling security systems. -/
structure SalesData where
  commission_per_sale : ℕ
  total_commission : ℕ
  first_street_sales : ℕ
  second_street_sales : ℕ
  fourth_street_sales : ℕ

/-- Theorem stating the number of security systems sold on the second street. -/
theorem second_street_sales (data : SalesData) : data.second_street_sales = 4 :=
  by
  have h1 : data.commission_per_sale = 25 := by sorry
  have h2 : data.total_commission = 175 := by sorry
  have h3 : data.first_street_sales = data.second_street_sales / 2 := by sorry
  have h4 : data.fourth_street_sales = 1 := by sorry
  have h5 : data.first_street_sales + data.second_street_sales + data.fourth_street_sales = 
            data.total_commission / data.commission_per_sale := by sorry
  sorry

end second_street_sales_l1526_152668


namespace smallest_number_l1526_152653

theorem smallest_number : min (-5 : ℝ) (min (-0.8) (min 0 (abs (-6)))) = -5 := by
  sorry

end smallest_number_l1526_152653


namespace normal_distribution_std_dev_l1526_152614

/-- For a normal distribution with given properties, prove the standard deviation --/
theorem normal_distribution_std_dev (μ σ : ℝ) (h1 : μ = 17.5) (h2 : μ - 2 * σ = 12.5) : σ = 2.5 := by
  sorry

end normal_distribution_std_dev_l1526_152614


namespace consecutive_even_integers_sum_of_cubes_l1526_152661

theorem consecutive_even_integers_sum_of_cubes (x y z : ℤ) : 
  (∃ n : ℤ, x = 2*n ∧ y = 2*n + 2 ∧ z = 2*n + 4) →  -- consecutive even integers
  x^2 + y^2 + z^2 = 2960 →                         -- sum of squares is 2960
  x^3 + y^3 + z^3 = 90117 :=                       -- sum of cubes is 90117
by sorry

end consecutive_even_integers_sum_of_cubes_l1526_152661


namespace min_people_with_both_hat_and_glove_l1526_152676

theorem min_people_with_both_hat_and_glove (n : ℕ) (gloves hats both : ℕ) : 
  n > 0 → 
  gloves = n / 3 →
  hats = 2 * n / 3 →
  gloves + hats - both = n →
  both ≥ 0 :=
by sorry

end min_people_with_both_hat_and_glove_l1526_152676


namespace dave_phone_difference_l1526_152602

theorem dave_phone_difference (initial_apps initial_files final_apps final_files : ℕ) : 
  initial_apps = 11 → 
  initial_files = 3 → 
  final_apps = 2 → 
  final_files = 24 → 
  final_files - final_apps = 22 := by
  sorry

end dave_phone_difference_l1526_152602


namespace f_neg_one_gt_f_two_l1526_152688

-- Define f as a function from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Condition 1: y = f(x+1) is an even function
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + x) = f (1 - x)

-- Condition 2: f(x) is an increasing function on the interval [1, +∞)
def is_increasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f x < f y

-- Theorem statement
theorem f_neg_one_gt_f_two 
  (h1 : is_even_shifted f) 
  (h2 : is_increasing_on_interval f) : 
  f (-1) > f 2 := by
  sorry

end f_neg_one_gt_f_two_l1526_152688


namespace consecutive_sum_39_largest_l1526_152609

theorem consecutive_sum_39_largest (n m : ℕ) : 
  n + 1 = m → n + m = 39 → m = 20 := by
sorry

end consecutive_sum_39_largest_l1526_152609


namespace inequality_solution_l1526_152699

theorem inequality_solution (x : ℝ) :
  (x^2 + 2*x - 15) / (x + 5) < 0 ↔ -5 < x ∧ x < 3 :=
by sorry

end inequality_solution_l1526_152699


namespace square_side_length_l1526_152652

theorem square_side_length (area : ℝ) (side : ℝ) :
  area = 289 ∧ area = side * side → side = 17 := by
  sorry

end square_side_length_l1526_152652


namespace divisibility_by_240_l1526_152640

theorem divisibility_by_240 (p : ℕ) (hp : p.Prime) (hp_gt_7 : p > 7) : 
  240 ∣ (p^4 - 1) := by
  sorry

end divisibility_by_240_l1526_152640


namespace area_polygon1_area_polygon2_area_polygon3_l1526_152648

-- Define the polygons
def polygon1 := {(x, y) : ℝ × ℝ | |x| ≤ 1 ∧ |y| ≤ 1}
def polygon2 := {(x, y) : ℝ × ℝ | |x| + |y| ≤ 10}
def polygon3 := {(x, y) : ℝ × ℝ | |x| + |y| + |x+y| ≤ 2020}

-- Define the areas
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statements
theorem area_polygon1 : area polygon1 = 4 := by sorry

theorem area_polygon2 : area polygon2 = 200 := by sorry

theorem area_polygon3 : area polygon3 = 3060300 := by sorry

end area_polygon1_area_polygon2_area_polygon3_l1526_152648


namespace geometric_progression_terms_l1526_152617

theorem geometric_progression_terms (a q : ℝ) : 
  a + a * q = 20 → a * q^2 + a * q^3 = 20/9 →
  ((a = 15 ∧ q = 1/3) ∨ (a = 30 ∧ q = -1/3)) ∧
  (∃ (terms : Fin 4 → ℝ), 
    (terms 0 = a ∧ terms 1 = a * q ∧ terms 2 = a * q^2 ∧ terms 3 = a * q^3) ∧
    ((terms 0 = 15 ∧ terms 1 = 5 ∧ terms 2 = 5/3 ∧ terms 3 = 5/9) ∨
     (terms 0 = 30 ∧ terms 1 = -10 ∧ terms 2 = 10/3 ∧ terms 3 = -10/9))) :=
by sorry

end geometric_progression_terms_l1526_152617


namespace simplify_radical_sum_l1526_152613

theorem simplify_radical_sum : Real.sqrt 72 + Real.sqrt 32 + Real.sqrt 50 = 15 * Real.sqrt 2 := by
  sorry

end simplify_radical_sum_l1526_152613


namespace annas_age_problem_l1526_152628

theorem annas_age_problem :
  ∃! x : ℕ+, 
    (∃ m : ℕ, (x : ℤ) - 4 = m^2) ∧ 
    (∃ n : ℕ, (x : ℤ) + 3 = n^3) ∧ 
    x = 5 := by
  sorry

end annas_age_problem_l1526_152628


namespace two_common_points_l1526_152657

/-- Two curves in the xy-plane -/
structure Curves where
  curve1 : ℝ → ℝ → Prop
  curve2 : ℝ → ℝ → Prop

/-- The specific curves from the problem -/
def problem_curves : Curves where
  curve1 := λ x y => x^2 + 9*y^2 = 9
  curve2 := λ x y => 9*x^2 + y^2 = 1

/-- A point that satisfies both curves -/
def is_common_point (c : Curves) (x y : ℝ) : Prop :=
  c.curve1 x y ∧ c.curve2 x y

/-- The set of all common points -/
def common_points (c : Curves) : Set (ℝ × ℝ) :=
  {p | is_common_point c p.1 p.2}

/-- The theorem stating that there are exactly two common points -/
theorem two_common_points :
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧
  common_points problem_curves = {p1, p2} :=
sorry

end two_common_points_l1526_152657


namespace right_triangle_third_side_l1526_152682

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 3 ∧ b = 4) ∨ (a = 3 ∧ c = 4) ∨ (b = 3 ∧ c = 4) →
  a^2 + b^2 = c^2 →
  c = 5 ∨ c = Real.sqrt 7 := by
sorry

end right_triangle_third_side_l1526_152682
