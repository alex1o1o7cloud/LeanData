import Mathlib

namespace zealand_has_one_fifth_l1861_186171

/-- Represents the amount of money each person has -/
structure Money where
  wanda : ℚ
  xander : ℚ
  yusuf : ℚ
  zealand : ℚ

/-- The initial state of money distribution -/
def initial_money : Money :=
  { wanda := 6, xander := 5, yusuf := 4, zealand := 0 }

/-- The state of money after Zealand receives money from others -/
def final_money : Money :=
  { wanda := 5, xander := 4, yusuf := 3, zealand := 3 }

/-- The fraction of money Zealand has at the end -/
def zealand_fraction (m : Money) : ℚ :=
  m.zealand / (m.wanda + m.xander + m.yusuf + m.zealand)

/-- Theorem stating that Zealand ends up with 1/5 of the total money -/
theorem zealand_has_one_fifth :
  zealand_fraction final_money = 1/5 := by
  sorry

end zealand_has_one_fifth_l1861_186171


namespace line_through_points_l1861_186185

theorem line_through_points (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ n : ℤ, (b / a : ℝ) = n) (h4 : ∃ m : ℝ, ∀ x y : ℝ, y = k * x + m → (x = a ∧ y = a) ∨ (x = b ∧ y = 8 * b)) :
  k = 9 ∨ k = 15 :=
sorry

end line_through_points_l1861_186185


namespace quadratic_always_positive_implies_a_greater_than_one_l1861_186173

theorem quadratic_always_positive_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a > 1 := by
  sorry

end quadratic_always_positive_implies_a_greater_than_one_l1861_186173


namespace fraction_relation_l1861_186162

theorem fraction_relation (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 1 / 4) :
  t / q = 8 / 3 := by sorry

end fraction_relation_l1861_186162


namespace balance_scale_l1861_186199

/-- The weight of a green ball in terms of blue balls -/
def green_to_blue : ℚ := 2

/-- The weight of a yellow ball in terms of blue balls -/
def yellow_to_blue : ℚ := 8/3

/-- The weight of a white ball in terms of blue balls -/
def white_to_blue : ℚ := 5/3

/-- The number of green balls on the scale -/
def num_green : ℕ := 3

/-- The number of yellow balls on the scale -/
def num_yellow : ℕ := 3

/-- The number of white balls on the scale -/
def num_white : ℕ := 3

theorem balance_scale : 
  (num_green : ℚ) * green_to_blue + 
  (num_yellow : ℚ) * yellow_to_blue + 
  (num_white : ℚ) * white_to_blue = 19 := by sorry

end balance_scale_l1861_186199


namespace curve_is_parabola_l1861_186187

-- Define the curve
def curve (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + y^2) = |3*x - 4*y + 2| / 5

-- Define the fixed point F
def F : ℝ × ℝ := (2, 0)

-- Define the line
def line (x y : ℝ) : Prop :=
  3*x - 4*y + 2 = 0

-- Theorem statement
theorem curve_is_parabola :
  ∃ (f : ℝ × ℝ) (l : ℝ → ℝ → Prop),
    (∀ x y, curve x y ↔ 
      (Real.sqrt ((x - f.1)^2 + (y - f.2)^2) = 
       Real.sqrt ((3*x - 4*y + 2)^2) / 5)) ∧
    (f = F) ∧
    (∀ x y, l x y ↔ line x y) ∧
    (¬ l F.1 F.2) :=
  sorry

end curve_is_parabola_l1861_186187


namespace sin_225_plus_alpha_l1861_186152

theorem sin_225_plus_alpha (α : ℝ) (h : Real.sin (π/4 + α) = 5/13) :
  Real.sin (5*π/4 + α) = -5/13 := by
  sorry

end sin_225_plus_alpha_l1861_186152


namespace propositions_true_l1861_186180

theorem propositions_true (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 - b^2 = 1 → a - b < 1) ∧
  (Real.exp a - Real.exp b = 1 → a - b < 1) := by
  sorry

end propositions_true_l1861_186180


namespace nth_equation_pattern_l1861_186107

theorem nth_equation_pattern (n : ℕ) : (n + 1)^2 - 1 = n * (n + 2) := by
  sorry

end nth_equation_pattern_l1861_186107


namespace pythago_competition_l1861_186170

theorem pythago_competition (n : ℕ) : 
  (∀ s : ℕ, s ≤ n → ∃! (team : Fin 4 → ℕ), ∀ i j : Fin 4, i ≠ j → team i ≠ team j) →
  (∃ daniel : ℕ, daniel < 50 ∧ 
    (∃ eliza fiona greg : ℕ, 
      eliza = 50 ∧ fiona = 81 ∧ greg = 97 ∧
      daniel < eliza ∧ daniel < fiona ∧ daniel < greg ∧
      (∀ x : ℕ, x ≤ 4*n → (x ≤ daniel ↔ 2*x ≤ 4*n + 1)))) →
  n = 25 := by sorry

end pythago_competition_l1861_186170


namespace pages_after_break_l1861_186120

theorem pages_after_break (total_pages : ℕ) (break_percentage : ℚ) 
  (h1 : total_pages = 30) 
  (h2 : break_percentage = 7/10) : 
  total_pages - (total_pages * break_percentage).floor = 9 := by
  sorry

end pages_after_break_l1861_186120


namespace g_composition_of_three_l1861_186149

def g (x : ℝ) : ℝ := 7 * x + 3

theorem g_composition_of_three : g (g (g 3)) = 1200 := by
  sorry

end g_composition_of_three_l1861_186149


namespace divide_by_four_twice_l1861_186198

theorem divide_by_four_twice (x : ℝ) : x = 166.08 → (x / 4) / 4 = 10.38 := by
  sorry

end divide_by_four_twice_l1861_186198


namespace egg_roll_ratio_l1861_186155

-- Define the number of egg rolls each person ate
def matthew_egg_rolls : ℕ := 6
def alvin_egg_rolls : ℕ := 4

-- Define Patrick's egg rolls based on the condition
def patrick_egg_rolls : ℕ := matthew_egg_rolls / 3

-- Theorem to prove the ratio
theorem egg_roll_ratio :
  patrick_egg_rolls / alvin_egg_rolls = 1 / 2 := by
  sorry

end egg_roll_ratio_l1861_186155


namespace initial_salt_concentration_l1861_186153

/-- The initial volume of saltwater solution in gallons -/
def x : ℝ := 120

/-- The initial salt concentration as a percentage -/
def C : ℝ := 18.33333333333333

theorem initial_salt_concentration (C : ℝ) :
  (C / 100 * x + 16) / (3 / 4 * x + 8 + 16) = 1 / 3 → C = 18.33333333333333 :=
by sorry

end initial_salt_concentration_l1861_186153


namespace largest_of_four_consecutive_even_l1861_186139

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def consecutive_even (a b c d : ℤ) : Prop :=
  is_even a ∧ is_even b ∧ is_even c ∧ is_even d ∧
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2

theorem largest_of_four_consecutive_even (a b c d : ℤ) :
  consecutive_even a b c d → a + b + c + d = 92 → d = 26 := by
  sorry

end largest_of_four_consecutive_even_l1861_186139


namespace two_slices_per_pizza_l1861_186159

/-- Given a total number of pizza slices and a number of pizzas,
    calculate the number of slices per pizza. -/
def slices_per_pizza (total_slices : ℕ) (num_pizzas : ℕ) : ℕ :=
  total_slices / num_pizzas

/-- Prove that given 28 total slices and 14 pizzas, each pizza has 2 slices. -/
theorem two_slices_per_pizza :
  slices_per_pizza 28 14 = 2 := by
  sorry

end two_slices_per_pizza_l1861_186159


namespace triangle_angle_B_l1861_186166

theorem triangle_angle_B (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  C = π / 5 ∧  -- Given condition
  a * Real.cos B - b * Real.cos A = c →  -- Given equation
  B = 3 * π / 10 := by
sorry

end triangle_angle_B_l1861_186166


namespace triangle_line_equations_l1861_186116

/-- Triangle with vertices A(0,-5), B(-3,3), and C(2,0) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Given triangle -/
def givenTriangle : Triangle :=
  { A := (0, -5)
  , B := (-3, 3)
  , C := (2, 0) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) satisfies the line equation ax + by + c = 0 -/
def satisfiesLineEquation (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

theorem triangle_line_equations (t : Triangle) :
  (t = givenTriangle) →
  (∃ (lab : LineEquation), lab.a = 8 ∧ lab.b = 3 ∧ lab.c = 15 ∧
    satisfiesLineEquation t.A lab ∧ satisfiesLineEquation t.B lab) ∧
  (∃ (lac : LineEquation), lac.a = 5 ∧ lac.b = -2 ∧ lac.c = -10 ∧
    satisfiesLineEquation t.A lac ∧ satisfiesLineEquation t.C lac) :=
by sorry

end triangle_line_equations_l1861_186116


namespace inequality_proof_l1861_186156

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ∧
  (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end inequality_proof_l1861_186156


namespace stationery_cost_l1861_186157

theorem stationery_cost (x y : ℚ) 
  (h1 : 2 * x + 3 * y = 18) 
  (h2 : 3 * x + 2 * y = 22) : 
  x + y = 8 := by
  sorry

end stationery_cost_l1861_186157


namespace f_neg_three_gt_f_neg_five_l1861_186150

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y → f y < f x

-- State the theorem
theorem f_neg_three_gt_f_neg_five
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_decreasing : is_decreasing_on_nonneg f) :
  f (-3) > f (-5) :=
sorry

end f_neg_three_gt_f_neg_five_l1861_186150


namespace son_age_l1861_186115

theorem son_age (son_age man_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end son_age_l1861_186115


namespace fourth_term_is_plus_minus_three_l1861_186191

/-- A geometric sequence with a_3 = 9 and a_5 = 1 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ 
  a 3 = 9 ∧ 
  a 5 = 1

/-- The fourth term of the geometric sequence is ±3 -/
theorem fourth_term_is_plus_minus_three 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  a 4 = 3 ∨ a 4 = -3 :=
sorry

end fourth_term_is_plus_minus_three_l1861_186191


namespace digit_sum_properties_l1861_186102

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Permutation of digits relation -/
def is_digit_permutation (m k : ℕ) : Prop := sorry

theorem digit_sum_properties (M K : ℕ) (h : is_digit_permutation M K) : 
  (sum_of_digits (2 * M) = sum_of_digits (2 * K)) ∧ 
  (M % 2 = 0 → K % 2 = 0 → sum_of_digits (M / 2) = sum_of_digits (K / 2)) ∧
  (sum_of_digits (5 * M) = sum_of_digits (5 * K)) := by
  sorry

end digit_sum_properties_l1861_186102


namespace sum_of_specific_repeating_decimals_l1861_186160

/-- Represents a repeating decimal -/
def RepeatingDecimal (whole : ℕ) (repeating : List ℕ) : ℚ :=
  sorry

/-- The sum of three specific repeating decimals -/
theorem sum_of_specific_repeating_decimals :
  RepeatingDecimal 0 [1] + RepeatingDecimal 0 [1, 2] + RepeatingDecimal 0 [1, 2, 3] =
  RepeatingDecimal 0 [3, 5, 5, 4, 4, 6] :=
sorry

end sum_of_specific_repeating_decimals_l1861_186160


namespace max_alternations_theorem_l1861_186182

/-- Represents a painter's strategy for painting fence sections -/
def PainterStrategy := ℕ → Bool

/-- Represents the state of the fence after painting -/
def FenceState := List Bool

/-- Counts the number of color alternations in a fence state -/
def countAlternations (fence : FenceState) : ℕ := sorry

/-- Simulates the painting process and returns the final fence state -/
def paintFence (strategy1 strategy2 : PainterStrategy) : FenceState := sorry

/-- The maximum number of alternations the first painter can guarantee -/
def maxGuaranteedAlternations : ℕ := sorry

/-- Theorem stating the maximum number of alternations the first painter can guarantee -/
theorem max_alternations_theorem :
  ∀ (strategy2 : PainterStrategy),
  ∃ (strategy1 : PainterStrategy),
  countAlternations (paintFence strategy1 strategy2) ≥ 49 ∧
  maxGuaranteedAlternations = 49 := by sorry

end max_alternations_theorem_l1861_186182


namespace quadratic_integer_criterion_l1861_186181

/-- A quadratic trinomial ax^2 + bx + c where a, b, and c are real numbers -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic trinomial at a given x -/
def QuadraticTrinomial.eval (q : QuadraticTrinomial) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- Proposition: 2a, a+b, and c are all integers if and only if 
    ax^2 + bx + c takes integer values for all integer x -/
theorem quadratic_integer_criterion (q : QuadraticTrinomial) :
  (∀ x : ℤ, ∃ n : ℤ, q.eval x = n) ↔ 
  (∃ m n p : ℤ, 2 * q.a = m ∧ q.a + q.b = n ∧ q.c = p) :=
sorry

end quadratic_integer_criterion_l1861_186181


namespace curve_circle_intersection_l1861_186144

theorem curve_circle_intersection :
  ∃ (x y : ℝ), (2 * x - y + 1 = 0) ∧ (x^2 + (y - Real.sqrt 2)^2 = 2) := by
  sorry

end curve_circle_intersection_l1861_186144


namespace rectangle_perimeter_squares_l1861_186110

def rectangle_length : ℕ := 47
def rectangle_width : ℕ := 65
def square_sides : List ℕ := [3, 5, 6, 11, 17, 19, 22, 23, 24, 25]
def perimeter_squares : List ℕ := [17, 19, 22, 23, 24, 25]

theorem rectangle_perimeter_squares :
  (2 * (rectangle_length + rectangle_width) = 
   2 * (perimeter_squares[3] + perimeter_squares[4] + perimeter_squares[5] + perimeter_squares[2]) + 
   perimeter_squares[0] + perimeter_squares[1]) ∧
  (∀ s ∈ perimeter_squares, s ∈ square_sides) ∧
  (perimeter_squares.length = 6) :=
by sorry

end rectangle_perimeter_squares_l1861_186110


namespace polar_to_rectangular_conversion_l1861_186163

theorem polar_to_rectangular_conversion :
  let r : ℝ := 3 * Real.sqrt 2
  let θ : ℝ := 5 * Real.pi / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -3 * Real.sqrt 6 / 2) ∧ (y = 3 * Real.sqrt 2 / 2) :=
by sorry

end polar_to_rectangular_conversion_l1861_186163


namespace min_sum_dimensions_l1861_186188

/-- The minimum sum of dimensions for a rectangular box with volume 1645 and positive integer dimensions -/
theorem min_sum_dimensions (l w h : ℕ+) : 
  l * w * h = 1645 → l + w + h ≥ 129 := by
  sorry

end min_sum_dimensions_l1861_186188


namespace interesting_numbers_characterization_l1861_186129

def is_interesting (n : ℕ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧
    n = ⌊1/a⌋ + ⌊1/b⌋ + ⌊1/c⌋

theorem interesting_numbers_characterization :
  ∀ n : ℕ, is_interesting n ↔ n ≥ 7 :=
sorry

end interesting_numbers_characterization_l1861_186129


namespace fruit_seller_apples_l1861_186167

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℝ) * 0.4 = 300 → initial_apples = 750 := by
  sorry

end fruit_seller_apples_l1861_186167


namespace hyperbola_eccentricity_l1861_186100

/-- The eccentricity of a hyperbola with the given conditions is √5 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let right_vertex := (a, 0)
  let line := fun (x : ℝ) => -x + a
  let asymptote1 := fun (x : ℝ) => (b / a) * x
  let asymptote2 := fun (x : ℝ) => -(b / a) * x
  let B := (a^2 / (a + b), a * b / (a + b))
  let C := (a^2 / (a - b), -a * b / (a - b))
  let vector_AB := (B.1 - right_vertex.1, B.2 - right_vertex.2)
  let vector_BC := (C.1 - B.1, C.2 - B.2)
  vector_AB = (1/2 : ℝ) • vector_BC →
  ∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 5 :=
by sorry

end hyperbola_eccentricity_l1861_186100


namespace decimal_equals_fraction_l1861_186123

/-- The decimal representation of the number we're considering -/
def decimal : ℚ := 0.53247247247

/-- The fraction representation we're aiming for -/
def fraction : ℚ := 53171 / 99900

/-- Theorem stating that the decimal equals the fraction -/
theorem decimal_equals_fraction : decimal = fraction := by sorry

end decimal_equals_fraction_l1861_186123


namespace closet_probability_l1861_186141

def shirts : ℕ := 5
def shorts : ℕ := 7
def socks : ℕ := 8
def total_articles : ℕ := shirts + shorts + socks
def articles_picked : ℕ := 4

theorem closet_probability : 
  (Nat.choose shirts 2 * Nat.choose shorts 1 * Nat.choose socks 1) / 
  Nat.choose total_articles articles_picked = 112 / 969 := by
  sorry

end closet_probability_l1861_186141


namespace expected_remaining_bullets_value_l1861_186189

/-- The probability of hitting the target with each shot -/
def hit_probability : ℝ := 0.6

/-- The total number of available bullets -/
def total_bullets : ℕ := 4

/-- The expected number of remaining bullets after the first hit -/
def expected_remaining_bullets : ℝ :=
  3 * hit_probability +
  2 * hit_probability * (1 - hit_probability) +
  1 * hit_probability * (1 - hit_probability)^2 +
  0 * (1 - hit_probability)^3

theorem expected_remaining_bullets_value :
  expected_remaining_bullets = 2.376 := by sorry

end expected_remaining_bullets_value_l1861_186189


namespace exp_ge_linear_l1861_186148

theorem exp_ge_linear (x : ℝ) : x + 1 ≤ Real.exp x := by
  sorry

end exp_ge_linear_l1861_186148


namespace cube_equation_solution_l1861_186147

theorem cube_equation_solution (x y z : ℕ) (h : x^3 = 3*y^3 + 9*z^3) : x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end cube_equation_solution_l1861_186147


namespace smallest_divisible_by_1_to_10_l1861_186109

theorem smallest_divisible_by_1_to_10 : ∀ n : ℕ, n > 0 → (∀ i : ℕ, 1 ≤ i → i ≤ 10 → i ∣ n) → n ≥ 2520 := by
  sorry

end smallest_divisible_by_1_to_10_l1861_186109


namespace f_36_equals_2pq_l1861_186184

/-- A function satisfying f(xy) = f(x) + f(y) for all x and y -/
def LogLikeFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x * y) = f x + f y

/-- Main theorem: f(36) = 2(p + q) given the conditions -/
theorem f_36_equals_2pq (f : ℝ → ℝ) (p q : ℝ) 
  (h1 : LogLikeFunction f) 
  (h2 : f 2 = p) 
  (h3 : f 3 = q) : 
  f 36 = 2 * (p + q) := by
  sorry


end f_36_equals_2pq_l1861_186184


namespace tonyas_christmas_gifts_l1861_186145

/-- Tonya's Christmas gift problem -/
theorem tonyas_christmas_gifts (num_sisters : ℕ) (num_dolls : ℕ) (doll_cost : ℕ) (lego_cost : ℕ) 
  (h1 : num_sisters = 2)
  (h2 : num_dolls = 4)
  (h3 : doll_cost = 15)
  (h4 : lego_cost = 20) :
  (num_dolls * doll_cost) / lego_cost = 3 := by
  sorry

#check tonyas_christmas_gifts

end tonyas_christmas_gifts_l1861_186145


namespace primary_school_ages_l1861_186172

theorem primary_school_ages (x y : ℕ) : 
  7 ≤ x ∧ x ≤ 13 ∧ 7 ≤ y ∧ y ≤ 13 →
  (x + y) * (x - y) = 63 →
  x = 12 ∧ y = 9 := by
sorry

end primary_school_ages_l1861_186172


namespace company_workers_count_l1861_186114

theorem company_workers_count (total : ℕ) (men : ℕ) : 
  (total / 3 : ℚ) * (1 / 10 : ℚ) + (2 * total / 3 : ℚ) * (3 / 5 : ℚ) = men →
  men = 120 →
  total - men = 280 :=
by sorry

end company_workers_count_l1861_186114


namespace range_of_f_l1861_186101

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end range_of_f_l1861_186101


namespace exam_score_problem_l1861_186133

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℕ) (wrong_penalty : ℕ) (total_score : ℤ) :
  total_questions = 75 ∧ correct_score = 4 ∧ wrong_penalty = 1 ∧ total_score = 125 →
  ∃ (correct_answers : ℕ),
    correct_answers * correct_score - (total_questions - correct_answers) * wrong_penalty = total_score ∧
    correct_answers = 40 := by
  sorry

end exam_score_problem_l1861_186133


namespace solve_for_B_l1861_186128

theorem solve_for_B : ∃ B : ℝ, (4 * B + 5 = 25) ∧ (B = 5) := by
  sorry

end solve_for_B_l1861_186128


namespace curve_crosses_at_point_l1861_186194

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := t^2 - 2

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^3 - 9*t + 5

/-- The curve crosses itself if there exist two distinct real numbers that yield the same point -/
def curve_crosses_itself : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ x a = x b ∧ y a = y b

/-- The point where the curve crosses itself -/
def crossing_point : ℝ × ℝ := (7, 5)

/-- Theorem stating that the curve crosses itself at the specified point -/
theorem curve_crosses_at_point :
  curve_crosses_itself ∧ ∃ t : ℝ, (x t, y t) = crossing_point :=
sorry

end curve_crosses_at_point_l1861_186194


namespace maximum_at_one_implies_a_greater_than_neg_one_l1861_186146

/-- The function f(x) = ln x - (1/2)ax² - bx has a maximum at x = 1 -/
def has_maximum_at_one (a b : ℝ) : Prop :=
  ∀ x, x > 0 → (Real.log x - (1/2) * a * x^2 - b * x) ≤ (Real.log 1 - (1/2) * a * 1^2 - b * 1)

/-- If f(x) = ln x - (1/2)ax² - bx has a maximum at x = 1, then a > -1 -/
theorem maximum_at_one_implies_a_greater_than_neg_one (a b : ℝ) :
  has_maximum_at_one a b → a > -1 := by
  sorry

end maximum_at_one_implies_a_greater_than_neg_one_l1861_186146


namespace smallest_y_for_81_power_gt_7_power_42_l1861_186130

theorem smallest_y_for_81_power_gt_7_power_42 :
  ∃ y : ℕ, (∀ z : ℕ, 81^z ≤ 7^42 → z < y) ∧ 81^y > 7^42 :=
by
  -- The proof goes here
  sorry

end smallest_y_for_81_power_gt_7_power_42_l1861_186130


namespace count_complementary_sets_l1861_186193

/-- Represents a card with four attributes -/
structure Card :=
  (shape : Fin 3)
  (color : Fin 3)
  (shade : Fin 3)
  (size : Fin 3)

/-- The deck of all possible cards -/
def deck : Finset Card := sorry

/-- A set of three cards -/
def ThreeCardSet := Finset Card

/-- Predicate for a complementary set -/
def is_complementary (s : ThreeCardSet) : Prop := sorry

/-- The set of all complementary three-card sets -/
def complementary_sets : Finset ThreeCardSet := sorry

theorem count_complementary_sets :
  Finset.card complementary_sets = 6483 := by sorry

end count_complementary_sets_l1861_186193


namespace bobby_jump_difference_l1861_186124

/-- The number of jumps Bobby can do per minute as a child -/
def child_jumps_per_minute : ℕ := 30

/-- The number of jumps Bobby can do per second as an adult -/
def adult_jumps_per_second : ℕ := 1

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The difference in jumps per minute between Bobby as an adult and as a child -/
theorem bobby_jump_difference : 
  adult_jumps_per_second * seconds_per_minute - child_jumps_per_minute = 30 := by
  sorry

end bobby_jump_difference_l1861_186124


namespace sum_of_fractions_l1861_186143

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end sum_of_fractions_l1861_186143


namespace zhang_bing_age_problem_l1861_186192

def current_year : ℕ := 2023  -- Assuming current year is 2023

def birth_year : ℕ := 1953

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem zhang_bing_age_problem :
  ∃! x : ℕ, 
    birth_year < x ∧ 
    x ≤ current_year ∧
    (x - birth_year) % 9 = 0 ∧
    x - birth_year = sum_of_digits x ∧
    x - birth_year = 18 := by
  sorry

end zhang_bing_age_problem_l1861_186192


namespace photo_archive_album_size_l1861_186195

/-- Represents an album in the photo archive -/
structure Album where
  pages : ℕ
  photos_per_page : ℕ

/-- The photo archive system -/
structure PhotoArchive where
  album : Album
  /-- Ensures all albums are identical -/
  albums_identical : ∀ a b : Album, a = b

theorem photo_archive_album_size 
  (archive : PhotoArchive)
  (h1 : archive.album.photos_per_page = 4)
  (h2 : ∃ x : ℕ, 81 = (x - 1) * (archive.album.pages * archive.album.photos_per_page) + 5 * archive.album.photos_per_page)
  (h3 : ∃ y : ℕ, 171 = (y - 1) * (archive.album.pages * archive.album.photos_per_page) + 3 * archive.album.photos_per_page)
  : archive.album.pages * archive.album.photos_per_page = 32 := by
  sorry

end photo_archive_album_size_l1861_186195


namespace system_solution_l1861_186197

theorem system_solution (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) :
  (∃ x y : ℝ, a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂ ∧ x = 8 ∧ y = 5) →
  (∃ x y : ℝ, 4 * a₁ * x - 5 * b₁ * y = 3 * c₁ ∧ 4 * a₂ * x - 5 * b₂ * y = 3 * c₂ ∧ x = 6 ∧ y = -3) :=
by sorry

end system_solution_l1861_186197


namespace school_population_l1861_186118

theorem school_population (t : ℕ) : 
  let g := 4 * t          -- number of girls
  let b := 6 * g          -- number of boys
  let s := t / 2          -- number of staff members
  b + g + t + s = 59 * t / 2 := by
sorry

end school_population_l1861_186118


namespace battle_station_staffing_l1861_186105

theorem battle_station_staffing (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 5) :
  n * (n - 1) * (n - 2) * (n - 3) * (n - 4) = 360360 :=
by sorry

end battle_station_staffing_l1861_186105


namespace min_value_theorem_l1861_186108

/-- The function f(x) defined as |x-a| + |x+b| -/
def f (a b x : ℝ) : ℝ := |x - a| + |x + b|

/-- The theorem stating the minimum value of (a^2/b + b^2/a) given conditions on f(x) -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 3) (hequal : ∃ x, f a b x = 3) :
  ∀ c d, c > 0 → d > 0 → c^2 / d + d^2 / c ≥ 3 :=
by sorry

end min_value_theorem_l1861_186108


namespace bernoulli_joint_distribution_theorem_bernoulli_independence_theorem_l1861_186196

/-- Bernoulli random variable -/
structure BernoulliRV where
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- Joint distribution of two Bernoulli random variables -/
structure JointDistribution (X Y : BernoulliRV) where
  pxy : ℝ × ℝ → ℝ
  sum_to_one : (pxy (0, 0)) + (pxy (0, 1)) + (pxy (1, 0)) + (pxy (1, 1)) = 1

/-- Covariance of two Bernoulli random variables -/
def cov (X Y : BernoulliRV) : ℝ := sorry

/-- Main theorem -/
theorem bernoulli_joint_distribution_theorem (X Y : BernoulliRV) :
  ∃! (j : JointDistribution X Y),
    (j.pxy (1, 1) = cov X Y + X.p * Y.p) ∧
    (j.pxy (0, 1) = Y.p - (cov X Y + X.p * Y.p)) ∧
    (j.pxy (1, 0) = X.p - (cov X Y + X.p * Y.p)) ∧
    (j.pxy (0, 0) = 1 - X.p - Y.p + (cov X Y + X.p * Y.p)) :=
  sorry

/-- Independence theorem -/
theorem bernoulli_independence_theorem (X Y : BernoulliRV) (j : JointDistribution X Y) :
  (∀ x y, j.pxy (x, y) = (if x = 1 then X.p else 1 - X.p) * (if y = 1 then Y.p else 1 - Y.p)) ↔
  cov X Y = 0 :=
  sorry

end bernoulli_joint_distribution_theorem_bernoulli_independence_theorem_l1861_186196


namespace ratio_sum_squares_l1861_186132

theorem ratio_sum_squares : 
  ∀ (x y z : ℝ), 
    y = 2 * x → 
    z = 3 * x → 
    x + y + z = 12 → 
    x^2 + y^2 + z^2 = 56 := by
  sorry

end ratio_sum_squares_l1861_186132


namespace biology_physics_ratio_l1861_186151

/-- The ratio of students in Biology class to Physics class -/
theorem biology_physics_ratio :
  let girls_biology : ℕ := 3 * 25
  let boys_biology : ℕ := 25
  let students_biology : ℕ := girls_biology + boys_biology
  let students_physics : ℕ := 200
  (students_biology : ℚ) / students_physics = 1 / 2 := by
  sorry

end biology_physics_ratio_l1861_186151


namespace digits_of_2_pow_120_l1861_186111

theorem digits_of_2_pow_120 (h : ∃ n : ℕ, 10^60 ≤ 2^200 ∧ 2^200 < 10^61) :
  ∃ m : ℕ, 10^36 ≤ 2^120 ∧ 2^120 < 10^37 :=
sorry

end digits_of_2_pow_120_l1861_186111


namespace cyclist_hiker_catch_up_l1861_186103

/-- Proves that the time the cyclist travels after passing the hiker before stopping
    is equal to the time it takes the hiker to catch up to the cyclist while waiting. -/
theorem cyclist_hiker_catch_up (hiker_speed cyclist_speed : ℝ) (wait_time : ℝ) :
  hiker_speed > 0 →
  cyclist_speed > hiker_speed →
  wait_time > 0 →
  cyclist_speed = 4 * hiker_speed →
  (cyclist_speed / hiker_speed - 1) * wait_time = wait_time :=
by
  sorry

#check cyclist_hiker_catch_up

end cyclist_hiker_catch_up_l1861_186103


namespace polynomial_degree_is_12_l1861_186190

/-- The degree of a polynomial (x^5 + ax^8 + bx^2 + c)(y^3 + dy^2 + e)(z + f) -/
def polynomial_degree (a b c d e f : ℝ) : ℕ :=
  let p1 := fun (x : ℝ) => x^5 + a*x^8 + b*x^2 + c
  let p2 := fun (y : ℝ) => y^3 + d*y^2 + e
  let p3 := fun (z : ℝ) => z + f
  let product := fun (x y z : ℝ) => p1 x * p2 y * p3 z
  12

theorem polynomial_degree_is_12 (a b c d e f : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
    (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) :
    polynomial_degree a b c d e f = 12 := by
  sorry

end polynomial_degree_is_12_l1861_186190


namespace decreasing_function_inequality_l1861_186169

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y, x < y → f x > f y) →  -- f is decreasing on ℝ
  f (3 * a) < f (-2 * a + 10) →
  a > 2 :=
by sorry

end decreasing_function_inequality_l1861_186169


namespace inverse_proportion_point_l1861_186175

theorem inverse_proportion_point : 
  let x : ℝ := 2 * Real.sqrt 2
  let y : ℝ := Real.sqrt 2
  y = 4 / x := by sorry

end inverse_proportion_point_l1861_186175


namespace line_equation_l1861_186137

/-- A line parameterized by (x, y) = (3t + 6, 5t - 7) where t is a real number has the equation y = (5/3)x - 17 -/
theorem line_equation (t : ℝ) :
  let x : ℝ := 3 * t + 6
  let y : ℝ := 5 * t - 7
  y = (5/3) * x - 17 := by sorry

end line_equation_l1861_186137


namespace pure_imaginary_complex_number_l1861_186186

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 2) = (a^2 - 3*a + 2) + Complex.I * (a - 2)) → a = 1 :=
by sorry

end pure_imaginary_complex_number_l1861_186186


namespace no_integer_solutions_l1861_186135

theorem no_integer_solutions : ¬∃ (x y z : ℤ), 
  (x^2 - 2*x*y + 3*y^2 - z^2 = 45) ∧ 
  (-x^2 + 5*y*z + 3*z^2 = 28) ∧ 
  (x^2 - x*y + 9*z^2 = 140) :=
by sorry

end no_integer_solutions_l1861_186135


namespace polynomial_simplification_l1861_186131

theorem polynomial_simplification (p : ℝ) :
  (5 * p^3 - 7 * p^2 + 3 * p + 8) + (-3 * p^3 + 9 * p^2 - 4 * p + 2) =
  2 * p^3 + 2 * p^2 - p + 10 := by
sorry

end polynomial_simplification_l1861_186131


namespace min_value_expression_l1861_186104

theorem min_value_expression (x : ℝ) :
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2021 ≥ 2020 ∧
  ∃ y : ℝ, (y + 1) * (y + 2) * (y + 3) * (y + 4) + 2021 = 2020 :=
by sorry

end min_value_expression_l1861_186104


namespace no_x_squared_term_l1861_186122

/-- Given an algebraic expression (x^2 + mx)(x - 3), if the simplified form does not contain the term x^2, then m = 3 -/
theorem no_x_squared_term (m : ℝ) : 
  (∀ x : ℝ, (x^2 + m*x) * (x - 3) = x^3 + (m - 3)*x^2 - 3*m*x) →
  (m - 3 = 0) →
  m = 3 :=
by sorry

end no_x_squared_term_l1861_186122


namespace integral_proof_l1861_186121

open Real

theorem integral_proof (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -2) :
  let f : ℝ → ℝ := λ x => (x^3 + 6*x^2 + 14*x + 10) / ((x+1)*(x+2)^3)
  let F : ℝ → ℝ := λ x => log (abs (x+1)) - 1 / (x+2)^2
  deriv F x = f x := by sorry

end integral_proof_l1861_186121


namespace equilateral_ABC_l1861_186176

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The incenter of a triangle -/
def incenter (t : Triangle) : Point :=
  sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- Main theorem -/
theorem equilateral_ABC (A B C I X Y Z : Point) :
  let ABC := Triangle.mk A B C
  let BIC := Triangle.mk B I C
  let CIA := Triangle.mk C I A
  let AIB := Triangle.mk A I B
  let XYZ := Triangle.mk X Y Z
  (I = incenter ABC) →
  (X = incenter BIC) →
  (Y = incenter CIA) →
  (Z = incenter AIB) →
  isEquilateral XYZ →
  isEquilateral ABC :=
by
  sorry

end equilateral_ABC_l1861_186176


namespace hypotenuse_ratio_from_area_ratio_l1861_186164

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  hypotenuse : ℝ
  area : ℝ

-- Theorem statement
theorem hypotenuse_ratio_from_area_ratio
  (t1 t2 : IsoscelesRightTriangle)
  (h_area : t2.area = 2 * t1.area) :
  t2.hypotenuse = Real.sqrt 2 * t1.hypotenuse :=
sorry

end hypotenuse_ratio_from_area_ratio_l1861_186164


namespace f_of_one_f_of_a_f_of_f_of_a_l1861_186154

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 3

-- Theorem statements
theorem f_of_one : f 1 = 5 := by sorry

theorem f_of_a (a : ℝ) : f a = 2 * a + 3 := by sorry

theorem f_of_f_of_a (a : ℝ) : f (f a) = 4 * a + 9 := by sorry

end f_of_one_f_of_a_f_of_f_of_a_l1861_186154


namespace x_value_proof_l1861_186117

theorem x_value_proof (x : ℚ) (h : (1/4 : ℚ) - (1/5 : ℚ) + (1/10 : ℚ) = 4/x) : x = 80/3 := by
  sorry

end x_value_proof_l1861_186117


namespace f_increasing_on_negative_reals_l1861_186161

-- Define the function f(x) = -|x|
def f (x : ℝ) : ℝ := -|x|

-- State the theorem
theorem f_increasing_on_negative_reals :
  ∀ (x₁ x₂ : ℝ), x₁ < 0 → x₂ < 0 → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end f_increasing_on_negative_reals_l1861_186161


namespace octagon_area_l1861_186138

-- Define the octagon's vertices
def octagon_vertices : List (ℝ × ℝ) :=
  [(0, 0), (1, 3), (2.5, 4), (4.5, 4), (6, 1), (4.5, -2), (2.5, -3), (1, -3)]

-- Define the function to calculate the area of a polygon given its vertices
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ :=
  sorry

-- Theorem statement
theorem octagon_area :
  polygon_area octagon_vertices = 34 :=
sorry

end octagon_area_l1861_186138


namespace cloud_9_diving_refund_l1861_186177

/-- Cloud 9 Diving Company Cancellation Refund Problem -/
theorem cloud_9_diving_refund (individual_bookings group_bookings total_after_cancellations : ℕ) 
  (h1 : individual_bookings = 12000)
  (h2 : group_bookings = 16000)
  (h3 : total_after_cancellations = 26400) :
  individual_bookings + group_bookings - total_after_cancellations = 1600 := by
  sorry

end cloud_9_diving_refund_l1861_186177


namespace expression_evaluation_l1861_186112

theorem expression_evaluation (x y : ℚ) (hx : x = 2/3) (hy : y = 5/2) :
  (1/3) * x^8 * y^9 = 5^9 / (2 * 3^9) := by
  sorry

end expression_evaluation_l1861_186112


namespace third_to_second_night_ratio_l1861_186165

/-- Represents Billy's sleep pattern over four nights -/
structure SleepPattern where
  first_night : ℝ
  second_night : ℝ
  third_night : ℝ
  fourth_night : ℝ

/-- Calculates the total sleep over four nights -/
def total_sleep (sp : SleepPattern) : ℝ :=
  sp.first_night + sp.second_night + sp.third_night + sp.fourth_night

/-- Theorem stating the ratio of third to second night's sleep -/
theorem third_to_second_night_ratio 
  (sp : SleepPattern)
  (h1 : sp.first_night = 6)
  (h2 : sp.second_night = sp.first_night + 2)
  (h3 : sp.fourth_night = 3 * sp.third_night)
  (h4 : total_sleep sp = 30) :
  sp.third_night / sp.second_night = 1 / 2 := by
  sorry

end third_to_second_night_ratio_l1861_186165


namespace greatest_consecutive_integers_sum_36_l1861_186178

/-- The sum of consecutive integers starting from a given integer -/
def sumConsecutiveIntegers (start : ℤ) (count : ℕ) : ℤ :=
  (count : ℤ) * (2 * start + (count : ℤ) - 1) / 2

/-- The property that the sum of a sequence of consecutive integers is 36 -/
def hasSumThirtySix (start : ℤ) (count : ℕ) : Prop :=
  sumConsecutiveIntegers start count = 36

/-- The theorem stating that 72 is the greatest number of consecutive integers whose sum is 36 -/
theorem greatest_consecutive_integers_sum_36 :
  (∃ start : ℤ, hasSumThirtySix start 72) ∧
  (∀ n : ℕ, n > 72 → ∀ start : ℤ, ¬hasSumThirtySix start n) :=
sorry

end greatest_consecutive_integers_sum_36_l1861_186178


namespace nested_square_root_equality_l1861_186140

theorem nested_square_root_equality : 
  Real.sqrt (49 * Real.sqrt (25 * Real.sqrt 9)) = 5 * Real.sqrt 7 * Real.sqrt (Real.sqrt 3) := by
  sorry

end nested_square_root_equality_l1861_186140


namespace hemisphere_on_cone_surface_area_l1861_186106

/-- The total surface area of a solid figure formed by placing a hemisphere on top of a cone -/
theorem hemisphere_on_cone_surface_area
  (hemisphere_radius : ℝ)
  (cone_base_radius : ℝ)
  (cone_slant_height : ℝ)
  (hemisphere_radius_eq : hemisphere_radius = 5)
  (cone_base_radius_eq : cone_base_radius = 7)
  (cone_slant_height_eq : cone_slant_height = 14) :
  2 * π * hemisphere_radius^2 + π * hemisphere_radius^2 + π * cone_base_radius * cone_slant_height = 173 * π :=
by sorry

end hemisphere_on_cone_surface_area_l1861_186106


namespace tricia_age_correct_l1861_186125

-- Define the ages of each person as natural numbers
def Vincent_age : ℕ := 22
def Rupert_age : ℕ := Vincent_age - 2
def Khloe_age : ℕ := Rupert_age - 10
def Eugene_age : ℕ := 3 * Khloe_age
def Yorick_age : ℕ := 2 * Eugene_age
def Amilia_age : ℕ := Yorick_age / 4
def Tricia_age : ℕ := 5

-- State the theorem
theorem tricia_age_correct : 
  Vincent_age = 22 ∧ 
  Rupert_age = Vincent_age - 2 ∧
  Khloe_age = Rupert_age - 10 ∧
  Khloe_age * 3 = Eugene_age ∧
  Eugene_age * 2 = Yorick_age ∧
  Yorick_age / 4 = Amilia_age ∧
  ∃ (n : ℕ), n * Tricia_age = Amilia_age →
  Tricia_age = 5 :=
by sorry

end tricia_age_correct_l1861_186125


namespace rectangular_field_path_area_and_cost_l1861_186134

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def path_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem rectangular_field_path_area_and_cost 
  (field_length field_width path_width cost_per_unit : ℝ) 
  (h1 : field_length = 85)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5)
  (h4 : cost_per_unit = 2) :
  path_area field_length field_width path_width = 725 ∧ 
  path_cost (path_area field_length field_width path_width) cost_per_unit = 1450 := by
  sorry

#check rectangular_field_path_area_and_cost

end rectangular_field_path_area_and_cost_l1861_186134


namespace prime_divisibility_l1861_186127

theorem prime_divisibility (p : ℕ) (hp : Prime p) (hp2 : p > 2) :
  ∃ k : ℤ, (⌊(2 + Real.sqrt 5)^p⌋ : ℤ) - 2^(p + 1) = k * p := by
  sorry

end prime_divisibility_l1861_186127


namespace raisin_cost_fraction_l1861_186183

/-- Represents the cost ratio of raisins to the total mixture -/
def cost_ratio (raisin_weight : ℚ) (nut_weight : ℚ) (nut_cost_ratio : ℚ) : ℚ :=
  (raisin_weight) / (raisin_weight + nut_weight * nut_cost_ratio)

/-- Theorem stating that the cost of raisins is 3/19 of the total mixture cost -/
theorem raisin_cost_fraction :
  cost_ratio 3 4 4 = 3 / 19 := by
sorry

end raisin_cost_fraction_l1861_186183


namespace orange_balloons_count_l1861_186168

/-- Given the initial number of orange balloons and the number of additional orange balloons found,
    prove that the total number of orange balloons is equal to their sum. -/
theorem orange_balloons_count 
  (initial_orange : ℝ) 
  (found_orange : ℝ) : 
  initial_orange + found_orange = 11 :=
by
  sorry

#check orange_balloons_count 9 2

end orange_balloons_count_l1861_186168


namespace math_problem_time_l1861_186142

/-- Proves that the time to solve each math problem is 2 minutes -/
theorem math_problem_time (
  math_problems : ℕ)
  (social_studies_problems : ℕ)
  (science_problems : ℕ)
  (social_studies_time : ℚ)
  (science_time : ℚ)
  (total_time : ℚ)
  (h1 : math_problems = 15)
  (h2 : social_studies_problems = 6)
  (h3 : science_problems = 10)
  (h4 : social_studies_time = 1/2)
  (h5 : science_time = 3/2)
  (h6 : total_time = 48) :
  ∃ (math_time : ℚ), math_time * math_problems + social_studies_time * social_studies_problems + science_time * science_problems = total_time ∧ math_time = 2 := by
  sorry

#check math_problem_time

end math_problem_time_l1861_186142


namespace magic_square_sum_l1861_186119

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e : ℕ)
  (top_left : ℕ := 30)
  (top_right : ℕ := 27)
  (middle_left : ℕ := 33)
  (bottom_middle : ℕ := 18)
  (sum : ℕ)
  (row_sums : sum = top_left + b + top_right)
  (col_sums : sum = top_left + middle_left + a)
  (diag_sums : sum = top_left + c + e)
  (middle_row : sum = middle_left + c + d)
  (bottom_row : sum = a + bottom_middle + e)

/-- The sum of a and d in the magic square is 38 -/
theorem magic_square_sum (ms : MagicSquare) : ms.a + ms.d = 38 := by
  sorry

end magic_square_sum_l1861_186119


namespace pure_imaginary_fraction_l1861_186158

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ (b : ℝ), (1 - a * Complex.I) / (1 + Complex.I) = b * Complex.I) → a = 1 := by
  sorry

end pure_imaginary_fraction_l1861_186158


namespace divisor_is_six_l1861_186126

def original_number : ℕ := 427398
def subtracted_number : ℕ := 6

theorem divisor_is_six : ∃ (d : ℕ), d > 0 ∧ d = subtracted_number ∧ (original_number - subtracted_number) % d = 0 := by
  sorry

end divisor_is_six_l1861_186126


namespace geometric_sequence_sum_l1861_186136

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  q > 1 →
  (4 * (a 2011)^2 - 8 * (a 2011) + 3 = 0) →
  (4 * (a 2012)^2 - 8 * (a 2012) + 3 = 0) →
  a 2013 + a 2014 = 18 :=
by
  sorry

end geometric_sequence_sum_l1861_186136


namespace soft_drink_pack_size_l1861_186179

/-- The number of cans in a pack of soft drinks -/
def num_cans : ℕ := 11

/-- The cost of a pack of soft drinks in dollars -/
def pack_cost : ℚ := 299/100

/-- The cost of an individual can in dollars -/
def can_cost : ℚ := 1/4

/-- Theorem stating that the number of cans in a pack is 11 -/
theorem soft_drink_pack_size :
  num_cans = ⌊pack_cost / can_cost⌋ := by sorry

end soft_drink_pack_size_l1861_186179


namespace parabolas_intersection_l1861_186113

/-- The first parabola -/
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 7

/-- The second parabola -/
def g (x : ℝ) : ℝ := 2 * x^2 - 5

/-- The intersection points of the two parabolas -/
def intersection_points : Set (ℝ × ℝ) := {(-2, 3), (1/2, -4.5)}

theorem parabolas_intersection :
  ∀ p : ℝ × ℝ, f p.1 = g p.1 ↔ p ∈ intersection_points := by sorry

end parabolas_intersection_l1861_186113


namespace bicycle_distance_l1861_186174

theorem bicycle_distance (front_circ rear_circ : ℚ) (extra_revs : ℕ) : 
  front_circ = 4/3 →
  rear_circ = 3/2 →
  extra_revs = 25 →
  (front_circ * (extra_revs + (rear_circ * extra_revs) / (front_circ - rear_circ))) = 300 := by
  sorry

end bicycle_distance_l1861_186174
