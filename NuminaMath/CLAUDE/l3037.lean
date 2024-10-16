import Mathlib

namespace NUMINAMATH_CALUDE_correct_probabilities_l3037_303771

def ball_probabilities (total_balls : ℕ) (p_red p_black_or_yellow p_yellow_or_green : ℚ) : Prop :=
  let p_black := 1/4
  let p_yellow := 1/6
  let p_green := 1/4
  total_balls = 12 ∧
  p_red = 1/3 ∧
  p_black_or_yellow = 5/12 ∧
  p_yellow_or_green = 5/12 ∧
  p_red + p_black + p_yellow + p_green = 1 ∧
  p_black_or_yellow = p_black + p_yellow ∧
  p_yellow_or_green = p_yellow + p_green

theorem correct_probabilities : 
  ∀ (total_balls : ℕ) (p_red p_black_or_yellow p_yellow_or_green : ℚ),
  ball_probabilities total_balls p_red p_black_or_yellow p_yellow_or_green := by
  sorry

end NUMINAMATH_CALUDE_correct_probabilities_l3037_303771


namespace NUMINAMATH_CALUDE_five_lines_max_sections_l3037_303714

/-- The maximum number of sections created by drawing n line segments through a rectangle,
    given that the first line segment separates the rectangle into 2 sections. -/
def max_sections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 2 + (n - 1) * n / 2

/-- Theorem: The maximum number of sections created by drawing 5 line segments
    through a rectangle is 16, given that the first line segment separates
    the rectangle into 2 sections. -/
theorem five_lines_max_sections :
  max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_lines_max_sections_l3037_303714


namespace NUMINAMATH_CALUDE_orchard_harvest_l3037_303727

/-- Calculates the total mass of fruit harvested in an orchard -/
def total_fruit_mass (apple_trees : ℕ) (apple_yield : ℕ) (peach_trees : ℕ) (peach_yield : ℕ) : ℕ :=
  apple_trees * apple_yield + peach_trees * peach_yield

/-- Theorem stating the total mass of fruit harvested in the specific orchard -/
theorem orchard_harvest :
  total_fruit_mass 30 150 45 65 = 7425 := by
  sorry

end NUMINAMATH_CALUDE_orchard_harvest_l3037_303727


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3037_303768

theorem coin_flip_probability :
  let n : ℕ := 5  -- total number of coins
  let k : ℕ := 3  -- number of specific coins we want to be heads
  let total_outcomes : ℕ := 2^n
  let favorable_outcomes : ℕ := 2^(n - k)
  favorable_outcomes / total_outcomes = (1 : ℚ) / 8 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3037_303768


namespace NUMINAMATH_CALUDE_chord_equation_l3037_303783

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

/-- The midpoint of the chord -/
def P : ℝ × ℝ := (8, 1)

/-- A point lies on the line containing the chord -/
def lies_on_chord_line (x y : ℝ) : Prop := 2*x - y - 15 = 0

theorem chord_equation :
  ∀ A B : ℝ × ℝ,
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  hyperbola x₁ y₁ →
  hyperbola x₂ y₂ →
  (x₁ + x₂) / 2 = P.1 →
  (y₁ + y₂) / 2 = P.2 →
  lies_on_chord_line x₁ y₁ ∧ lies_on_chord_line x₂ y₂ :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l3037_303783


namespace NUMINAMATH_CALUDE_f_properties_l3037_303785

def f (x b c : ℝ) : ℝ := x * abs x + b * x + c

theorem f_properties :
  (∀ x b, f x b 0 = -f (-x) b 0) ∧
  (∀ c, c > 0 → ∃! x, f x 0 c = 0) ∧
  (∀ x b c, f (x - 0) b c - c = -(f (-x - 0) b c - c)) ∧
  (∃ b c, ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x b c = 0 ∧ f y b c = 0 ∧ f z b c = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3037_303785


namespace NUMINAMATH_CALUDE_expression_evaluation_l3037_303775

theorem expression_evaluation : -1^2008 + (-1)^2009 + 1^2010 + (-1)^2011 + 1^2012 = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3037_303775


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l3037_303725

theorem complex_product_pure_imaginary (x : ℝ) : 
  let z₁ : ℂ := 1 - I
  let z₂ : ℂ := -1 - x * I
  (z₁ * z₂).re = 0 → x = -1 := by sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l3037_303725


namespace NUMINAMATH_CALUDE_target_probability_l3037_303717

/-- The probability of hitting the target in a single shot -/
def p : ℝ := 0.8

/-- The probability of missing the target in a single shot -/
def q : ℝ := 1 - p

/-- The probability of hitting the target at least once in two shots -/
def prob_hit_at_least_once_in_two : ℝ := 1 - q^2

theorem target_probability :
  prob_hit_at_least_once_in_two = 0.96 →
  (5 : ℝ) * p^4 * q = 0.4096 :=
sorry

end NUMINAMATH_CALUDE_target_probability_l3037_303717


namespace NUMINAMATH_CALUDE_line_point_at_47_l3037_303707

/-- A line passing through three given points -/
structure Line where
  -- Define the line using two points
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ
  -- Ensure the third point lies on the line
  x3 : ℝ
  y3 : ℝ
  point_on_line : (y3 - y1) / (x3 - x1) = (y2 - y1) / (x2 - x1)

/-- Theorem: For the given line, when x = 47, y = 143 -/
theorem line_point_at_47 (l : Line) 
  (h1 : l.x1 = 2 ∧ l.y1 = 8)
  (h2 : l.x2 = 6 ∧ l.y2 = 20)
  (h3 : l.x3 = 10 ∧ l.y3 = 32) :
  let m := (l.y2 - l.y1) / (l.x2 - l.x1)
  let b := l.y1 - m * l.x1
  m * 47 + b = 143 := by
  sorry

end NUMINAMATH_CALUDE_line_point_at_47_l3037_303707


namespace NUMINAMATH_CALUDE_real_part_of_z_l3037_303788

/-- Given that z = (1+i)(1-2i)(i) where i is the imaginary unit, prove that the real part of z is 3 -/
theorem real_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (1 + i) * (1 - 2*i) * i
  (z.re : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3037_303788


namespace NUMINAMATH_CALUDE_group_payment_l3037_303750

/-- Calculates the total amount paid by a group of moviegoers -/
def total_amount_paid (adult_price child_price : ℚ) (total_people adults : ℕ) : ℚ :=
  adult_price * adults + child_price * (total_people - adults)

/-- Theorem: The group paid $54.50 in total -/
theorem group_payment : total_amount_paid 9.5 6.5 7 3 = 54.5 := by
  sorry

end NUMINAMATH_CALUDE_group_payment_l3037_303750


namespace NUMINAMATH_CALUDE_summer_break_difference_l3037_303786

theorem summer_break_difference (camp_kids : ℕ) (home_kids : ℕ) 
  (h1 : camp_kids = 819058) (h2 : home_kids = 668278) : 
  camp_kids - home_kids = 150780 := by
  sorry

end NUMINAMATH_CALUDE_summer_break_difference_l3037_303786


namespace NUMINAMATH_CALUDE_inequality_solution_inequality_system_solution_l3037_303713

-- Part 1: Inequality solution
theorem inequality_solution (x : ℝ) :
  (2 * x - 1) / 2 ≥ 1 - (x + 1) / 3 ↔ x ≥ 7 / 8 :=
by sorry

-- Part 2: System of inequalities solution
theorem inequality_system_solution (x : ℝ) :
  (-2 * x ≤ -3 ∧ x / 2 < 2) ↔ (3 / 2 ≤ x ∧ x < 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_inequality_system_solution_l3037_303713


namespace NUMINAMATH_CALUDE_problem_solution_l3037_303701

/-- A geometric sequence with the given property -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

/-- The property of the sequence given in the problem -/
def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = 3 * (1/2)^n

theorem problem_solution (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : sequence_property a) : 
  a 5 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3037_303701


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l3037_303757

theorem simplify_radical_expression :
  Real.sqrt (13 + Real.sqrt 48) - Real.sqrt (5 - (2 * Real.sqrt 3 + 1)) + 2 * Real.sqrt (3 + (Real.sqrt 3 - 1)) = Real.sqrt 6 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l3037_303757


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l3037_303790

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) 
  (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l3037_303790


namespace NUMINAMATH_CALUDE_intersection_value_l3037_303764

theorem intersection_value (a : ℝ) : 
  let M : Set ℝ := {a^2, a+1, -3}
  let N : Set ℝ := {a-3, 2*a-1, a^2+1}
  M ∩ N = {-3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_value_l3037_303764


namespace NUMINAMATH_CALUDE_inequality_proof_l3037_303712

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 + b*c) + 1 / (b^3 + c*a) + 1 / (c^3 + a*b) ≤ (a*b + b*c + c*a)^2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3037_303712


namespace NUMINAMATH_CALUDE_lemon_candy_count_l3037_303770

theorem lemon_candy_count (total : ℕ) (caramel : ℕ) (p : ℚ) (lemon : ℕ) : 
  caramel = 3 →
  p = 3 / 7 →
  p = caramel / total →
  lemon = total - caramel →
  lemon = 4 := by
sorry

end NUMINAMATH_CALUDE_lemon_candy_count_l3037_303770


namespace NUMINAMATH_CALUDE_cat_toy_cost_l3037_303791

theorem cat_toy_cost (total_payment change cage_cost : ℚ) 
  (h1 : total_payment = 20)
  (h2 : change = 0.26)
  (h3 : cage_cost = 10.97) :
  total_payment - change - cage_cost = 8.77 := by
  sorry

end NUMINAMATH_CALUDE_cat_toy_cost_l3037_303791


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3037_303731

-- Define a parabola
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

-- Define a line
structure Line where
  m : ℝ
  b : ℝ

-- Define the concept of a directrix
def is_directrix (l : Line) (p : Parabola) : Prop := sorry

-- Define the concept of tangency
def is_tangent (l : Line) (p : Parabola) : Prop := sorry

-- Define the concept of intersection
def intersect (l : Line) (p : Parabola) : Finset ℝ := sorry

-- Main theorem
theorem parabola_line_intersection
  (p : Parabola)
  (l1 l2 : Line)
  (h1 : l1.m ≠ l2.m ∨ l1.b ≠ l2.b) -- lines are distinct
  (h2 : is_directrix l1 p)
  (h3 : ¬ is_tangent l1 p)
  (h4 : ¬ is_tangent l2 p) :
  (intersect l1 p).card + (intersect l2 p).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3037_303731


namespace NUMINAMATH_CALUDE_percent_of_x_l3037_303752

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 5 + x / 25) / x * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_l3037_303752


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l3037_303755

/-- A geometric sequence with specific conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  second_term : a 2 = 6
  sum_condition : 6 * a 1 + a 3 = 30

/-- The general term formula for the geometric sequence -/
def general_term (seq : GeometricSequence) : ℕ → ℝ
| n => (3 * 3^(n - 1) : ℝ)

/-- Alternative general term formula for the geometric sequence -/
def general_term_alt (seq : GeometricSequence) : ℕ → ℝ
| n => (2 * 2^(n - 1) : ℝ)

/-- Theorem stating that one of the general term formulas is correct -/
theorem geometric_sequence_general_term (seq : GeometricSequence) :
  (∀ n, seq.a n = general_term seq n) ∨ (∀ n, seq.a n = general_term_alt seq n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l3037_303755


namespace NUMINAMATH_CALUDE_hoseok_calculation_l3037_303759

theorem hoseok_calculation : ∃ x : ℤ, (x - 7 = 9) ∧ (3 * x = 48) := by
  sorry

end NUMINAMATH_CALUDE_hoseok_calculation_l3037_303759


namespace NUMINAMATH_CALUDE_todd_spending_l3037_303723

/-- The amount Todd spent on the candy bar in cents -/
def candy_cost : ℕ := 14

/-- The amount Todd spent on the box of cookies in cents -/
def cookies_cost : ℕ := 39

/-- The total amount Todd spent in cents -/
def total_spent : ℕ := candy_cost + cookies_cost

theorem todd_spending :
  total_spent = 53 := by sorry

end NUMINAMATH_CALUDE_todd_spending_l3037_303723


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3037_303765

def f (x : ℝ) := -2 * (x + 3)^2 + 1

theorem quadratic_function_properties :
  let opens_downward := ∀ x y : ℝ, f ((x + y) / 2) > (f x + f y) / 2
  let axis_of_symmetry := 3
  let vertex := (3, 1)
  let decreases_after_three := ∀ x₁ x₂ : ℝ, x₁ > 3 → x₂ > x₁ → f x₂ < f x₁
  
  (opens_downward ∧ ¬(f axis_of_symmetry = f (-axis_of_symmetry)) ∧
   ¬(f (vertex.1) = vertex.2) ∧ decreases_after_three) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3037_303765


namespace NUMINAMATH_CALUDE_max_degree_difference_for_special_graph_l3037_303763

/-- A graph with specific properties -/
structure SpecialGraph where
  vertices : ℕ
  edges : ℕ
  disjoint_pairs : ℕ

/-- The maximal degree difference in a graph -/
def max_degree_difference (G : SpecialGraph) : ℕ :=
  sorry

/-- Theorem stating the maximal degree difference for a specific graph -/
theorem max_degree_difference_for_special_graph :
  ∃ (G : SpecialGraph),
    G.vertices = 30 ∧
    G.edges = 105 ∧
    G.disjoint_pairs = 4822 ∧
    max_degree_difference G = 22 :=
  sorry

end NUMINAMATH_CALUDE_max_degree_difference_for_special_graph_l3037_303763


namespace NUMINAMATH_CALUDE_roy_sports_time_l3037_303706

/-- Calculates the total time spent on sports activities for a specific week --/
def total_sports_time (
  basketball_time : ℝ)
  (swimming_time : ℝ)
  (track_time : ℝ)
  (school_days : ℕ)
  (missed_days : ℕ)
  (weekend_soccer : ℝ)
  (weekend_basketball : ℝ)
  (canceled_swimming : ℕ) : ℝ :=
  let school_sports := (basketball_time + swimming_time + track_time) * (school_days - missed_days : ℝ) - 
                       swimming_time * canceled_swimming
  let weekend_sports := weekend_soccer + weekend_basketball
  school_sports + weekend_sports

/-- Theorem stating that Roy's total sports time for the specific week is 13.5 hours --/
theorem roy_sports_time : 
  total_sports_time 1 1.5 1 5 2 1.5 3 1 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_roy_sports_time_l3037_303706


namespace NUMINAMATH_CALUDE_parabola_vertex_l3037_303720

/-- The equation of a parabola in the form y^2 + 4y + 3x + 1 = 0 -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 4*y + 3*x + 1 = 0

/-- The vertex of a parabola -/
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x' y', eq x' y' → y' ≥ y

/-- Theorem: The vertex of the parabola y^2 + 4y + 3x + 1 = 0 is (1, -2) -/
theorem parabola_vertex :
  is_vertex 1 (-2) parabola_equation :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3037_303720


namespace NUMINAMATH_CALUDE_inequality_multiplication_l3037_303767

theorem inequality_multiplication (x y : ℝ) : x < y → 2 * x < 2 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l3037_303767


namespace NUMINAMATH_CALUDE_total_games_calculation_l3037_303760

/-- The number of baseball games played at night -/
def night_games : ℕ := 128

/-- The number of games Joan attended -/
def attended_games : ℕ := 395

/-- The number of games Joan missed -/
def missed_games : ℕ := 469

/-- The total number of baseball games played this year -/
def total_games : ℕ := attended_games + missed_games

theorem total_games_calculation : 
  total_games = attended_games + missed_games := by sorry

end NUMINAMATH_CALUDE_total_games_calculation_l3037_303760


namespace NUMINAMATH_CALUDE_median_of_temperatures_l3037_303716

def temperatures : List ℝ := [19, 21, 25, 22, 19, 22, 21]

def median (l : List ℝ) : ℝ := sorry

theorem median_of_temperatures : median temperatures = 21 := by sorry

end NUMINAMATH_CALUDE_median_of_temperatures_l3037_303716


namespace NUMINAMATH_CALUDE_mixture_weight_l3037_303745

/-- Calculates the weight of a mixture of two brands of vegetable ghee -/
theorem mixture_weight (weight_a weight_b : ℝ) (ratio_a ratio_b total_volume : ℝ) : 
  weight_a = 900 →
  weight_b = 800 →
  ratio_a = 3 →
  ratio_b = 2 →
  total_volume = 4 →
  (((ratio_a / (ratio_a + ratio_b)) * total_volume * weight_a + 
   (ratio_b / (ratio_a + ratio_b)) * total_volume * weight_b) / 1000) = 3.44 := by
  sorry

#check mixture_weight

end NUMINAMATH_CALUDE_mixture_weight_l3037_303745


namespace NUMINAMATH_CALUDE_wholesale_price_calculation_l3037_303740

/-- The wholesale price of a pair of pants -/
def wholesale_price : ℝ := 20

/-- The retail price of a pair of pants -/
def retail_price : ℝ := 36

/-- The markup percentage as a decimal -/
def markup : ℝ := 0.8

theorem wholesale_price_calculation :
  wholesale_price = retail_price / (1 + markup) :=
by sorry

end NUMINAMATH_CALUDE_wholesale_price_calculation_l3037_303740


namespace NUMINAMATH_CALUDE_max_value_of_f_l3037_303721

def f (x : ℝ) : ℝ := -2 * (x + 1)^2 + 3

theorem max_value_of_f :
  ∃ (max : ℝ), max = 3 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3037_303721


namespace NUMINAMATH_CALUDE_solve_for_c_l3037_303778

theorem solve_for_c (h1 : ∀ a b c : ℝ, a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1))
                    (h2 : 6 * 15 * c = 4) : c = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_c_l3037_303778


namespace NUMINAMATH_CALUDE_cost_equation_holds_l3037_303738

/-- Represents the cost equation for notebooks and colored pens --/
def cost_equation (x : ℕ) : Prop :=
  let total_items : ℕ := 20
  let total_cost : ℕ := 50
  let notebook_cost : ℕ := 4
  let pen_cost : ℕ := 2
  2 * (total_items - x) + notebook_cost * x = total_cost

/-- Theorem stating the cost equation holds for the given scenario --/
theorem cost_equation_holds : ∃ x : ℕ, cost_equation x := by sorry

end NUMINAMATH_CALUDE_cost_equation_holds_l3037_303738


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3037_303739

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | x * a - 1 = 0}

-- State the theorem
theorem possible_values_of_a :
  ∀ a : ℝ, (A ∩ B a = B a) → (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3037_303739


namespace NUMINAMATH_CALUDE_largest_inexpressible_number_l3037_303715

def is_expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

def has_enough_coins (a b : ℕ) : Prop :=
  a > 10 ∧ b > 10

theorem largest_inexpressible_number :
  (∀ n : ℕ, n > 19 → n ≤ 50 → is_expressible n) ∧
  ¬(is_expressible 19) ∧
  (∀ a b : ℕ, has_enough_coins a b → ∀ n : ℕ, n ≤ 50 → is_expressible n → ∃ c d : ℕ, n = 5 * c + 6 * d ∧ c ≤ a ∧ d ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_largest_inexpressible_number_l3037_303715


namespace NUMINAMATH_CALUDE_sector_area_l3037_303792

theorem sector_area (n : Real) (r : Real) (h1 : n = 120) (h2 : r = 3) :
  (n * π * r^2) / 360 = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3037_303792


namespace NUMINAMATH_CALUDE_difference_of_squares_302_298_l3037_303799

theorem difference_of_squares_302_298 : 302^2 - 298^2 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_302_298_l3037_303799


namespace NUMINAMATH_CALUDE_video_game_cost_is_60_l3037_303737

/-- Represents the cost of the video game given Bucky's fish-catching earnings --/
def video_game_cost (last_weekend_earnings trout_price bluegill_price total_fish trout_percentage additional_savings : ℝ) : ℝ :=
  let trout_count := trout_percentage * total_fish
  let bluegill_count := total_fish - trout_count
  let sunday_earnings := trout_count * trout_price + bluegill_count * bluegill_price
  last_weekend_earnings + sunday_earnings + additional_savings

/-- Theorem stating the cost of the video game based on given conditions --/
theorem video_game_cost_is_60 :
  video_game_cost 35 5 4 5 0.6 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_video_game_cost_is_60_l3037_303737


namespace NUMINAMATH_CALUDE_y_value_when_x_is_8_l3037_303741

theorem y_value_when_x_is_8 (k : ℝ) :
  (∀ x, (x : ℝ) > 0 → k * x^(1/3) = 3 * Real.sqrt 2 → x = 64) →
  k * 8^(1/3) = (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_y_value_when_x_is_8_l3037_303741


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3037_303735

theorem simplify_and_evaluate : 
  let a : ℚ := -2
  let b : ℚ := 1/5
  2 * a * b^2 - (6 * a^3 * b + 2 * (a * b^2 - 1/2 * a^3 * b)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3037_303735


namespace NUMINAMATH_CALUDE_equation_solution_range_l3037_303754

theorem equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 2 ∧ (x + m) / (x - 2) - 3 = (x - 1) / (2 - x)) ↔ 
  (m ≥ -5 ∧ m ≠ -3) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3037_303754


namespace NUMINAMATH_CALUDE_pool_filling_time_l3037_303772

/-- Represents the volume of the pool -/
def pool_volume : ℝ := 1

/-- Represents the rate at which pipe X fills the pool -/
def rate_X : ℝ := sorry

/-- Represents the rate at which pipe Y fills the pool -/
def rate_Y : ℝ := sorry

/-- Represents the rate at which pipe Z fills the pool -/
def rate_Z : ℝ := sorry

/-- Time taken by pipes X and Y together to fill the pool -/
def time_XY : ℝ := 3

/-- Time taken by pipes X and Z together to fill the pool -/
def time_XZ : ℝ := 6

/-- Time taken by pipes Y and Z together to fill the pool -/
def time_YZ : ℝ := 4.5

theorem pool_filling_time :
  let time_XYZ := pool_volume / (rate_X + rate_Y + rate_Z)
  pool_volume / (rate_X + rate_Y) = time_XY ∧
  pool_volume / (rate_X + rate_Z) = time_XZ ∧
  pool_volume / (rate_Y + rate_Z) = time_YZ →
  (time_XYZ ≥ 2.76 ∧ time_XYZ ≤ 2.78) := by sorry

end NUMINAMATH_CALUDE_pool_filling_time_l3037_303772


namespace NUMINAMATH_CALUDE_probability_x_gt_7y_in_rectangle_l3037_303728

/-- The probability of a point (x,y) satisfying x > 7y in a specific rectangle -/
theorem probability_x_gt_7y_in_rectangle : 
  let rectangle_area := 2009 * 2010
  let triangle_area := (1 / 2) * 2009 * (2009 / 7)
  triangle_area / rectangle_area = 287 / 4020 := by
sorry

end NUMINAMATH_CALUDE_probability_x_gt_7y_in_rectangle_l3037_303728


namespace NUMINAMATH_CALUDE_existence_of_equal_elements_l3037_303794

theorem existence_of_equal_elements
  (p q n : ℕ+)
  (h_sum : p + q < n)
  (x : Fin (n + 1) → ℤ)
  (h_boundary : x 0 = 0 ∧ x n = 0)
  (h_diff : ∀ i : Fin n, x (i + 1) - x i = p ∨ x (i + 1) - x i = -q) :
  ∃ (i j : Fin (n + 1)), i ≠ j ∧ (i, j) ≠ (0, n) ∧ x i = x j :=
by sorry

end NUMINAMATH_CALUDE_existence_of_equal_elements_l3037_303794


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l3037_303711

/-- Three points in a 2D plane are collinear if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem collinear_points_x_value :
  let p : ℝ × ℝ := (1, 1)
  let a : ℝ × ℝ := (2, -4)
  let b : ℝ × ℝ := (x, 9)
  collinear p a b → x = 3 := by
sorry


end NUMINAMATH_CALUDE_collinear_points_x_value_l3037_303711


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l3037_303700

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start ∧ k < start + count → ¬(is_prime k)

theorem smallest_prime_after_six_nonprimes :
  ∃ n : ℕ, 
    (consecutive_nonprimes (n - 6) 6) ∧ 
    (is_prime n) ∧ 
    (∀ m : ℕ, m < n → ¬(consecutive_nonprimes (m - 6) 6 ∧ is_prime m)) ∧
    n = 53 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l3037_303700


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l3037_303789

theorem division_multiplication_problem : 
  let x : ℝ := 7.5
  let y : ℝ := 6
  let z : ℝ := 12
  (x / y) * z = 15 := by sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l3037_303789


namespace NUMINAMATH_CALUDE_exam_score_standard_deviation_l3037_303732

/-- Given an exam with mean score 74 and standard deviation σ,
    prove that if 98 is 3σ above the mean and 58 is k⋅σ below the mean,
    then k = 2 -/
theorem exam_score_standard_deviation (σ : ℝ) (k : ℝ) 
    (h1 : 98 = 74 + 3 * σ)
    (h2 : 58 = 74 - k * σ) : 
    k = 2 := by sorry

end NUMINAMATH_CALUDE_exam_score_standard_deviation_l3037_303732


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3037_303766

/-- The vertex coordinates of the parabola y = x^2 + 2x - 2 are (-1, -3) -/
theorem parabola_vertex_coordinates :
  let f : ℝ → ℝ := λ x => x^2 + 2*x - 2
  ∃ (h : ℝ → ℝ), (∀ x, f x = h (x + 1) - 3) ∧ (∀ x, h x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3037_303766


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_l3037_303798

theorem sqrt_eight_minus_sqrt_two : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_l3037_303798


namespace NUMINAMATH_CALUDE_employed_males_percentage_l3037_303761

/-- Proves that the percentage of the population that are employed males is 80%,
    given that 120% of the population are employed and 33.33333333333333% of employed people are females. -/
theorem employed_males_percentage (total_employed : Real) (female_employed_ratio : Real) :
  total_employed = 120 →
  female_employed_ratio = 100/3 →
  (1 - female_employed_ratio / 100) * total_employed = 80 := by
  sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l3037_303761


namespace NUMINAMATH_CALUDE_square_of_1005_l3037_303722

theorem square_of_1005 : (1005 : ℕ)^2 = 1010025 := by sorry

end NUMINAMATH_CALUDE_square_of_1005_l3037_303722


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l3037_303748

/-- Given a geometric sequence with first term b₁ = 2, 
    the minimum value of 3b₂ + 4b₃ is -9/8 -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) : 
  b₁ = 2 → 
  (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) → 
  (∀ r : ℝ, 3 * b₂ + 4 * b₃ ≥ -9/8) ∧ 
  (∃ r : ℝ, 3 * b₂ + 4 * b₃ = -9/8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l3037_303748


namespace NUMINAMATH_CALUDE_cindys_calculation_l3037_303780

theorem cindys_calculation (x : ℝ) : (x - 12) / 4 = 72 → (x - 5) / 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l3037_303780


namespace NUMINAMATH_CALUDE_smallest_integer_with_divisibility_properties_l3037_303702

theorem smallest_integer_with_divisibility_properties : 
  ∃ (n : ℕ), n > 1 ∧ 
  (∀ (m : ℕ), m > 1 → 
    ((m + 1) % 2 = 0 ∧ 
     (m + 2) % 3 = 0 ∧ 
     (m + 3) % 4 = 0 ∧ 
     (m + 4) % 5 = 0) → m ≥ n) ∧
  (n + 1) % 2 = 0 ∧ 
  (n + 2) % 3 = 0 ∧ 
  (n + 3) % 4 = 0 ∧ 
  (n + 4) % 5 = 0 ∧
  n = 61 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_divisibility_properties_l3037_303702


namespace NUMINAMATH_CALUDE_circle_focus_at_center_l3037_303736

/-- An ellipse with equal major and minor axes is a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The focus of a circle is at its center -/
def Circle.focus (c : Circle) : ℝ × ℝ := c.center

theorem circle_focus_at_center (h_center : ℝ × ℝ) (h_radius : ℝ) :
  let c : Circle := { center := h_center, radius := h_radius }
  c.focus = c.center := by sorry

end NUMINAMATH_CALUDE_circle_focus_at_center_l3037_303736


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l3037_303749

def triangle_inequality (f g : ℝ → ℝ) (A B : ℝ) : Prop :=
  f (Real.cos A) * g (Real.sin B) > f (Real.sin B) * g (Real.cos A)

theorem triangle_inequality_theorem 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g)
  (hg_pos : ∀ x, g x > 0)
  (h_deriv : ∀ x, (deriv f x) * (g x) - (f x) * (deriv g x) > 0)
  (A B C : ℝ)
  (h_obtuse : C > Real.pi / 2)
  (h_triangle : A + B + C = Real.pi) :
  triangle_inequality f g A B :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l3037_303749


namespace NUMINAMATH_CALUDE_receipts_change_l3037_303705

theorem receipts_change 
  (original_price : ℝ) 
  (original_sales : ℝ) 
  (price_reduction_rate : ℝ) 
  (sales_increase_rate : ℝ) 
  (h1 : price_reduction_rate = 0.3) 
  (h2 : sales_increase_rate = 0.5) : 
  let new_price := original_price * (1 - price_reduction_rate)
  let new_sales := original_sales * (1 + sales_increase_rate)
  let original_receipts := original_price * original_sales
  let new_receipts := new_price * new_sales
  (new_receipts - original_receipts) / original_receipts = 0.05 := by
sorry

end NUMINAMATH_CALUDE_receipts_change_l3037_303705


namespace NUMINAMATH_CALUDE_eulers_theorem_l3037_303703

/-- A convex polyhedron with f faces, p vertices, and a edges -/
structure ConvexPolyhedron where
  f : ℕ  -- number of faces
  p : ℕ  -- number of vertices
  a : ℕ  -- number of edges

/-- Euler's theorem for convex polyhedra -/
theorem eulers_theorem (poly : ConvexPolyhedron) : poly.f + poly.p - poly.a = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_theorem_l3037_303703


namespace NUMINAMATH_CALUDE_magic_square_sum_l3037_303756

/-- Represents a 3x3 magic square with some known and unknown values -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  sum : ℕ
  sum_eq_row1 : sum = 30 + e + 18
  sum_eq_row2 : sum = 15 + c + d
  sum_eq_row3 : sum = a + 27 + b
  sum_eq_col1 : sum = 30 + 15 + a
  sum_eq_col2 : sum = e + c + 27
  sum_eq_col3 : sum = 18 + d + b
  sum_eq_diag1 : sum = 30 + c + b
  sum_eq_diag2 : sum = a + c + 18

theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 47 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l3037_303756


namespace NUMINAMATH_CALUDE_min_area_triangle_l3037_303747

/-- Given a line mx + ny - 1 = 0 that intersects the x-axis at A and y-axis at B, 
    with distance √3 from the origin, the minimum area of triangle AOB is 3. -/
theorem min_area_triangle (m n : ℝ) 
  (h1 : m^2 + n^2 = 1/3)  -- Distance from origin to line is √3
  (h2 : m ≠ 0 ∧ n ≠ 0)   -- Line intersects both axes
  : (∀ S : ℝ, S = (1 / (2 * |m * n|)) → S ≥ 3) ∧ 
    (∃ S : ℝ, S = (1 / (2 * |m * n|)) ∧ S = 3) :=
by sorry


end NUMINAMATH_CALUDE_min_area_triangle_l3037_303747


namespace NUMINAMATH_CALUDE_no_integer_solutions_cube_equation_l3037_303709

theorem no_integer_solutions_cube_equation :
  ¬ ∃ (x y z : ℤ), x^3 + y^3 = 9*z + 5 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_cube_equation_l3037_303709


namespace NUMINAMATH_CALUDE_new_students_count_l3037_303743

/-- The number of new students who joined Hendrix's class -/
def new_students : ℕ :=
  let initial_students : ℕ := 160
  let final_students : ℕ := 120
  let transfer_ratio : ℚ := 1/3
  let total_after_join : ℕ := final_students * 3 / 2
  total_after_join - initial_students

theorem new_students_count : new_students = 20 := by sorry

end NUMINAMATH_CALUDE_new_students_count_l3037_303743


namespace NUMINAMATH_CALUDE_no_integer_in_interval_l3037_303730

theorem no_integer_in_interval (n : ℕ) : ¬∃ k : ℤ, (n : ℝ) * Real.sqrt 2 - 1 / (3 * (n : ℝ)) < (k : ℝ) ∧ (k : ℝ) < (n : ℝ) * Real.sqrt 2 + 1 / (3 * (n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_in_interval_l3037_303730


namespace NUMINAMATH_CALUDE_slopes_negative_reciprocals_min_area_ANB_l3037_303744

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define points M and N
def M : ℝ × ℝ := (1, 0)
def N : ℝ × ℝ := (-1, 0)

-- Define a line passing through M
def line_through_M (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define points A and B as intersections of the line and parabola
def point_A (k : ℝ) : ℝ × ℝ := sorry
def point_B (k : ℝ) : ℝ × ℝ := sorry

-- Define slopes of NA and NB
def slope_NA (k : ℝ) : ℝ := sorry
def slope_NB (k : ℝ) : ℝ := sorry

-- Define area of triangle ANB
def area_ANB (k : ℝ) : ℝ := sorry

theorem slopes_negative_reciprocals :
  ∀ k : ℝ, k ≠ 0 → slope_NA k * slope_NB k = -1 :=
sorry

theorem min_area_ANB :
  ∃ min_area : ℝ, min_area = 4 ∧ ∀ k : ℝ, k ≠ 0 → area_ANB k ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_slopes_negative_reciprocals_min_area_ANB_l3037_303744


namespace NUMINAMATH_CALUDE_tetrahedron_volume_zero_l3037_303782

-- Define the arithmetic progression
def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k, k < 11 → a (k + 1) = a k + d

-- Define the vertices of the tetrahedron
def tetrahedron_vertices (a : ℕ → ℝ) : Fin 4 → ℝ × ℝ × ℝ
| 0 => (a 1 ^ 2, a 2 ^ 2, a 3 ^ 2)
| 1 => (a 4 ^ 2, a 5 ^ 2, a 6 ^ 2)
| 2 => (a 7 ^ 2, a 8 ^ 2, a 9 ^ 2)
| 3 => (a 10 ^ 2, a 11 ^ 2, a 12 ^ 2)

-- Define the volume of a tetrahedron
def tetrahedron_volume (vertices : Fin 4 → ℝ × ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem tetrahedron_volume_zero (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_progression a d →
  tetrahedron_volume (tetrahedron_vertices a) = 0 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_zero_l3037_303782


namespace NUMINAMATH_CALUDE_sum_of_digits_up_to_billion_l3037_303708

/-- Sum of digits function for a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := sorry

/-- The main theorem: sum of digits of all numbers from 1 to 1 billion -/
theorem sum_of_digits_up_to_billion :
  sumOfDigitsUpTo 1000000000 = 40500000001 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_up_to_billion_l3037_303708


namespace NUMINAMATH_CALUDE_derived_figure_total_length_l3037_303762

/-- Represents a shape with perpendicular adjacent sides -/
structure NewShape where
  sides : ℕ

/-- Represents the derived figure created from the new shape -/
structure DerivedFigure where
  left_vertical : ℕ
  right_vertical : ℕ
  lower_horizontal : ℕ
  extra_top : ℕ

/-- Creates a derived figure from a new shape -/
def create_derived_figure (s : NewShape) : DerivedFigure :=
  { left_vertical := 12
  , right_vertical := 9
  , lower_horizontal := 7
  , extra_top := 2 }

/-- Calculates the total length of segments in the derived figure -/
def total_length (d : DerivedFigure) : ℕ :=
  d.left_vertical + d.right_vertical + d.lower_horizontal + d.extra_top

/-- Theorem stating that the total length of segments in the derived figure is 30 units -/
theorem derived_figure_total_length (s : NewShape) :
  total_length (create_derived_figure s) = 30 := by
  sorry

end NUMINAMATH_CALUDE_derived_figure_total_length_l3037_303762


namespace NUMINAMATH_CALUDE_AB_length_l3037_303776

-- Define the points and lengths
variable (A B C D E F G : ℝ)

-- Define the midpoint relationships
axiom C_midpoint : C = (A + B) / 2
axiom D_midpoint : D = (A + C) / 2
axiom E_midpoint : E = (A + D) / 2
axiom F_midpoint : F = (A + E) / 2
axiom G_midpoint : G = (A + F) / 2

-- Given condition
axiom AG_length : G - A = 5

-- Theorem to prove
theorem AB_length : B - A = 160 := by
  sorry

end NUMINAMATH_CALUDE_AB_length_l3037_303776


namespace NUMINAMATH_CALUDE_f_inequality_l3037_303758

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem f_inequality (x : ℝ) (h : 0 < x ∧ x < 1) : f x < f (x^2) ∧ f (x^2) < (f x)^2 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l3037_303758


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l3037_303781

theorem subset_implies_a_range (a : ℝ) : 
  let M := {x : ℝ | (x - 1) * (x - 2) < 0}
  let N := {x : ℝ | x < a}
  (M ⊆ N) → a ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l3037_303781


namespace NUMINAMATH_CALUDE_negation_of_implication_l3037_303777

theorem negation_of_implication (a : ℝ) :
  ¬(a > -3 → a > -6) ↔ (a ≤ -3 → a ≤ -6) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3037_303777


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l3037_303733

/-- Given a triangle with sides 6, 8, and 10 units, formed by the centers of
    three mutually externally tangent circles, the sum of the areas of these
    circles is 56π. -/
theorem sum_of_circle_areas (r s t : ℝ) : 
  r + s = 6 →
  r + t = 8 →
  s + t = 10 →
  π * (r^2 + s^2 + t^2) = 56 * π :=
by sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l3037_303733


namespace NUMINAMATH_CALUDE_anya_hair_growth_l3037_303746

/-- Calculates the additional hairs Anya needs to grow in a week -/
def additional_hairs_needed (washes_per_week : ℕ) (hairs_lost_per_wash : ℕ) 
  (brushings_per_week : ℕ) (growth_rate : ℕ) (growth_period : ℕ) : ℕ :=
  let hairs_lost_washing := washes_per_week * hairs_lost_per_wash
  let hairs_lost_brushing := brushings_per_week * (hairs_lost_per_wash / 2)
  let total_hair_loss := hairs_lost_washing + hairs_lost_brushing
  let growth_periods := 7 / growth_period
  let total_hair_growth := growth_periods * growth_rate
  total_hair_loss - total_hair_growth + 1

/-- Theorem stating that Anya needs to grow 63 additional hairs -/
theorem anya_hair_growth : 
  additional_hairs_needed 5 32 7 70 2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_anya_hair_growth_l3037_303746


namespace NUMINAMATH_CALUDE_probability_even_8_sided_die_l3037_303793

/-- A fair 8-sided die -/
def fair_8_sided_die : Finset ℕ := Finset.range 8

/-- The set of even outcomes on the die -/
def even_outcomes : Finset ℕ := Finset.filter (λ x => x % 2 = 0) fair_8_sided_die

/-- The probability of an event occurring when rolling the die -/
def probability (event : Finset ℕ) : ℚ :=
  (event.card : ℚ) / (fair_8_sided_die.card : ℚ)

theorem probability_even_8_sided_die :
  probability even_outcomes = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_even_8_sided_die_l3037_303793


namespace NUMINAMATH_CALUDE_line_points_theorem_l3037_303742

-- Define the line L with slope 2 passing through (3, 5)
def L (x y : ℝ) : Prop := y - 5 = 2 * (x - 3)

-- Define the points
def P1 : ℝ × ℝ := (3, 5)
def P2 (x2 : ℝ) : ℝ × ℝ := (x2, 7)
def P3 (y3 : ℝ) : ℝ × ℝ := (-1, y3)

theorem line_points_theorem (x2 y3 : ℝ) :
  L P1.1 P1.2 ∧ L (P2 x2).1 (P2 x2).2 ∧ L (P3 y3).1 (P3 y3).2 →
  x2 = 4 ∧ y3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_line_points_theorem_l3037_303742


namespace NUMINAMATH_CALUDE_complex_number_magnitude_product_l3037_303726

theorem complex_number_magnitude_product (z₁ z₂ : ℂ) : 
  Complex.abs (z₁ * z₂) = Complex.abs z₁ * Complex.abs z₂ := by sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_product_l3037_303726


namespace NUMINAMATH_CALUDE_vector_loop_closure_l3037_303795

variable {V : Type*} [AddCommGroup V]

theorem vector_loop_closure (A B C : V) :
  (B - A) - (B - C) + (A - C) = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_loop_closure_l3037_303795


namespace NUMINAMATH_CALUDE_standard_deviation_proof_l3037_303719

/-- The average age of job applicants -/
def average_age : ℝ := 31

/-- The number of different ages in the acceptable range -/
def different_ages : ℕ := 19

/-- The standard deviation of applicants' ages -/
def standard_deviation : ℝ := 9

/-- Theorem stating that the standard deviation is correct given the problem conditions -/
theorem standard_deviation_proof : 
  (average_age + standard_deviation) - (average_age - standard_deviation) = different_ages - 1 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_proof_l3037_303719


namespace NUMINAMATH_CALUDE_joan_eggs_count_l3037_303774

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Joan bought -/
def dozens_bought : ℕ := 6

/-- Theorem: Joan bought 72 eggs -/
theorem joan_eggs_count : dozens_bought * eggs_per_dozen = 72 := by
  sorry

end NUMINAMATH_CALUDE_joan_eggs_count_l3037_303774


namespace NUMINAMATH_CALUDE_det_sum_of_matrices_l3037_303734

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 6; 2, 3]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; 1, 0]

theorem det_sum_of_matrices : Matrix.det (A + B) = -3 := by sorry

end NUMINAMATH_CALUDE_det_sum_of_matrices_l3037_303734


namespace NUMINAMATH_CALUDE_series_sum_l3037_303729

theorem series_sum : 
  let a : ℕ → ℝ := λ n => n / 5^n
  let S := ∑' n, a n
  S = 5/16 := by
sorry

end NUMINAMATH_CALUDE_series_sum_l3037_303729


namespace NUMINAMATH_CALUDE_expression_simplification_l3037_303787

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (3 * x + 3)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3037_303787


namespace NUMINAMATH_CALUDE_inequality_proof_l3037_303797

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + a + 1) * (b^2 + b + 1) * (c^2 + c + 1) / (a * b * c) ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3037_303797


namespace NUMINAMATH_CALUDE_system_solution_unique_l3037_303751

theorem system_solution_unique :
  ∃! (x y z : ℝ),
    x^2 - 2*y + 1 = 0 ∧
    y^2 - 4*z + 7 = 0 ∧
    z^2 + 2*x - 2 = 0 ∧
    x = -1 ∧ y = 1 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3037_303751


namespace NUMINAMATH_CALUDE_yearly_income_calculation_l3037_303718

/-- Calculates the simple interest for a given principal, rate, and time (in years) -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ := 1) : ℚ :=
  principal * rate * time / 100

theorem yearly_income_calculation (totalAmount : ℚ) (part1 : ℚ) (rate1 : ℚ) (rate2 : ℚ) 
  (h1 : totalAmount = 2600)
  (h2 : part1 = 1600)
  (h3 : rate1 = 5)
  (h4 : rate2 = 6) :
  simpleInterest part1 rate1 + simpleInterest (totalAmount - part1) rate2 = 140 := by
  sorry

#eval simpleInterest 1600 5 + simpleInterest 1000 6

end NUMINAMATH_CALUDE_yearly_income_calculation_l3037_303718


namespace NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_line_l3037_303796

/-- The equation of a line passing through the center of a circle and parallel to another line -/
theorem line_through_circle_center_parallel_to_line :
  let circle : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 1}
  let center : ℝ × ℝ := (2, 0)
  let parallel_line : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 + 1 = 0}
  let result_line : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 - 4 = 0}
  (center ∈ circle) →
  (∀ p ∈ result_line, ∃ q ∈ parallel_line, (p.2 - q.2) / (p.1 - q.1) = (center.2 - q.2) / (center.1 - q.1)) →
  (center ∈ result_line) :=
by
  sorry

end NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_line_l3037_303796


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_greater_than_neg_four_l3037_303753

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + (a+2)*x + 1 = 0}
def B : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem intersection_empty_iff_a_greater_than_neg_four (a : ℝ) :
  A a ∩ B = ∅ ↔ a > -4 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_greater_than_neg_four_l3037_303753


namespace NUMINAMATH_CALUDE_max_value_inequality_l3037_303704

theorem max_value_inequality (k : ℝ) : 
  (∀ x : ℝ, |x^2 - 4*x + k| + |x - 3| ≤ 5) ∧ 
  (∃ x : ℝ, x = 3 ∧ |x^2 - 4*x + k| + |x - 3| = 5) ∧
  (∀ x : ℝ, x > 3 → |x^2 - 4*x + k| + |x - 3| > 5) →
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3037_303704


namespace NUMINAMATH_CALUDE_max_principals_is_three_l3037_303769

/-- Represents the duration of a principal's term in years -/
def term_length : ℕ := 4

/-- Represents the period of interest in years -/
def period_length : ℕ := 9

/-- Calculates the maximum number of principals that can serve during the period -/
def max_principals : ℕ := 
  (period_length + term_length - 1) / term_length

/-- Theorem stating that the maximum number of principals is 3 -/
theorem max_principals_is_three : max_principals = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_principals_is_three_l3037_303769


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3037_303773

def M : Set ℝ := {x | (x + 1) * (x - 3) ≤ 0}
def N : Set ℝ := {x | 1 < x ∧ x < 4}

theorem intersection_of_M_and_N : M ∩ N = {x | 1 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3037_303773


namespace NUMINAMATH_CALUDE_max_regions_quadratic_trinomials_l3037_303710

/-- The maximum number of regions into which the coordinate plane can be divided by n quadratic trinomials -/
def max_regions (n : ℕ) : ℕ := n^2 + 1

/-- Theorem stating that the maximum number of regions created by n quadratic trinomials is n^2 + 1 -/
theorem max_regions_quadratic_trinomials (n : ℕ) :
  max_regions n = n^2 + 1 := by sorry

end NUMINAMATH_CALUDE_max_regions_quadratic_trinomials_l3037_303710


namespace NUMINAMATH_CALUDE_difference_of_cubes_l3037_303724

theorem difference_of_cubes (y : ℝ) : 
  512 * y^3 - 27 = (8*y - 3) * (64*y^2 + 24*y + 9) ∧ 
  (8 + (-3) + 64 + 24 + 9 = 102) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_cubes_l3037_303724


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l3037_303779

theorem integer_root_of_cubic (b c : ℚ) :
  (∃ x : ℝ, x^3 + b*x + c = 0 ∧ x = 5 - Real.sqrt 21) →
  (∃ n : ℤ, n^3 + b*n + c = 0) →
  (∃ n : ℤ, n^3 + b*n + c = 0 ∧ n = -10) := by
sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l3037_303779


namespace NUMINAMATH_CALUDE_estimated_y_at_x_100_l3037_303784

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 1.43 * x + 257

-- Theorem statement
theorem estimated_y_at_x_100 :
  regression_equation 100 = 400 := by
  sorry

end NUMINAMATH_CALUDE_estimated_y_at_x_100_l3037_303784
