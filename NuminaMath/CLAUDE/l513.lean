import Mathlib

namespace polynomial_inequality_l513_51360

theorem polynomial_inequality (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1/2) →
  (∀ x : ℝ, |x| ≥ 1 → |a * x^2 + b * x + c| ≤ x^2 - 1/2) := by
  sorry

end polynomial_inequality_l513_51360


namespace concave_iff_m_nonneg_l513_51369

/-- A function f is concave on a set A if for any x₁, x₂ ∈ A,
    f((x₁ + x₂)/2) ≤ (1/2)[f(x₁) + f(x₂)] -/
def IsConcave (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f ((x₁ + x₂) / 2) ≤ (f x₁ + f x₂) / 2

/-- The function f(x) = mx² + x -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x

theorem concave_iff_m_nonneg (m : ℝ) :
  IsConcave (f m) ↔ m ≥ 0 := by sorry

end concave_iff_m_nonneg_l513_51369


namespace problem_solution_l513_51339

theorem problem_solution : 
  (Real.sqrt 6 + Real.sqrt 8 * Real.sqrt 12 = 5 * Real.sqrt 6) ∧ 
  (Real.sqrt 4 - Real.sqrt 2 / (Real.sqrt 2 + 1) = Real.sqrt 2) := by
  sorry

end problem_solution_l513_51339


namespace simplify_trig_expression_l513_51304

theorem simplify_trig_expression (x : ℝ) :
  (3 + 3 * Real.sin x - 3 * Real.cos x) / (3 + 3 * Real.sin x + 3 * Real.cos x) = Real.tan (x / 2) :=
by sorry

end simplify_trig_expression_l513_51304


namespace band_repertoire_size_l513_51390

def prove_band_repertoire (first_set second_set encore third_and_fourth_avg : ℕ) : Prop :=
  let total_songs := first_set + second_set + encore + 2 * third_and_fourth_avg
  total_songs = 30

theorem band_repertoire_size :
  prove_band_repertoire 5 7 2 8 := by
  sorry

end band_repertoire_size_l513_51390


namespace range_of_a_range_of_m_l513_51302

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * |x - 1| - a

-- Theorem 1
theorem range_of_a (a : ℝ) :
  (∃ x, f a x - 2 * |x - 7| ≤ 0) → a ≥ -12 := by
  sorry

-- Theorem 2
theorem range_of_m (m : ℝ) :
  (∀ x, f 1 x + |x + 7| ≥ m) → m ≤ 7 := by
  sorry

end range_of_a_range_of_m_l513_51302


namespace inscribed_square_side_length_l513_51331

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (hypotenuse : ℝ)
  (is_right : side1 = 5 ∧ side2 = 12 ∧ hypotenuse = 13)

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) :=
  (side_length : ℝ)
  (is_inscribed : True)  -- We assume the square is properly inscribed

/-- The side length of the inscribed square is 780/169 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 780 / 169 :=
sorry

end inscribed_square_side_length_l513_51331


namespace min_value_theorem_l513_51318

theorem min_value_theorem (a b : ℝ) (h : a * b = 1) :
  4 * a^2 + 9 * b^2 ≥ 12 := by sorry

end min_value_theorem_l513_51318


namespace problem_solution_l513_51340

theorem problem_solution : 
  let x := ((12 ^ 5) * (6 ^ 4)) / ((3 ^ 2) * (36 ^ 2)) + (Real.sqrt 9 * Real.log 27)
  ∃ ε > 0, |x - 27657.887510597983| < ε := by
sorry

end problem_solution_l513_51340


namespace product_upper_bound_l513_51337

theorem product_upper_bound (x : ℝ) (h : x ∈ Set.Icc 0 1) : x * (1 - x) ≤ (1 : ℝ) / 4 := by
  sorry

end product_upper_bound_l513_51337


namespace problem_1_l513_51380

theorem problem_1 : 
  Real.sqrt 48 / Real.sqrt 3 - 4 * Real.sqrt (1/5) * Real.sqrt 30 + (2 * Real.sqrt 2 + Real.sqrt 3)^2 = 15 := by
  sorry

end problem_1_l513_51380


namespace alloy_composition_theorem_l513_51393

/-- Represents the composition of an alloy -/
structure AlloyComposition where
  copper : ℝ
  tin : ℝ
  zinc : ℝ
  sum_to_one : copper + tin + zinc = 1

/-- The conditions given in the problem -/
def satisfies_conditions (c : AlloyComposition) : Prop :=
  c.copper - c.tin = 1/10 ∧ c.tin - c.zinc = 3/10

/-- The theorem to be proved -/
theorem alloy_composition_theorem :
  ∃ (c : AlloyComposition),
    satisfies_conditions c ∧
    c.copper = 0.5 ∧ c.tin = 0.4 ∧ c.zinc = 0.1 := by
  sorry

end alloy_composition_theorem_l513_51393


namespace sqrt_of_square_neg_l513_51309

theorem sqrt_of_square_neg (a : ℝ) (h : a < 0) : Real.sqrt (a ^ 2) = -a := by sorry

end sqrt_of_square_neg_l513_51309


namespace derek_same_color_probability_l513_51315

/-- Represents the number of marbles of each color -/
structure MarbleDistribution :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)

/-- Represents the number of marbles drawn by each person -/
structure DrawingProcess :=
  (david : ℕ)
  (dana : ℕ)
  (derek : ℕ)

/-- Calculates the probability of Derek getting at least 2 marbles of the same color -/
def probability_same_color (dist : MarbleDistribution) (process : DrawingProcess) : ℚ :=
  sorry

theorem derek_same_color_probability :
  let initial_distribution : MarbleDistribution := ⟨3, 2, 3⟩
  let drawing_process : DrawingProcess := ⟨2, 2, 3⟩
  probability_same_color initial_distribution drawing_process = 19 / 210 :=
sorry

end derek_same_color_probability_l513_51315


namespace digit_equation_solution_l513_51377

theorem digit_equation_solution :
  ∀ (A M C : ℕ),
  (A ≤ 9 ∧ M ≤ 9 ∧ C ≤ 9) →
  (10 * A^2 + 10 * M + C) * (A + M^2 + C^2) = 1050 →
  A = 2 := by
sorry

end digit_equation_solution_l513_51377


namespace sin_2theta_plus_pi_4_l513_51355

theorem sin_2theta_plus_pi_4 (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin (2 * θ + Real.pi / 4) = Real.sqrt 2 / 10 := by
  sorry

end sin_2theta_plus_pi_4_l513_51355


namespace sin_alpha_abs_value_l513_51351

/-- Theorem: If point P(3a, 4a) lies on the terminal side of angle α, where a ≠ 0, then |sin α| = 4/5 -/
theorem sin_alpha_abs_value (a : ℝ) (α : ℝ) (ha : a ≠ 0) :
  let P : ℝ × ℝ := (3 * a, 4 * a)
  (P.1 = 3 * a ∧ P.2 = 4 * a) → |Real.sin α| = 4 / 5 := by
  sorry

end sin_alpha_abs_value_l513_51351


namespace smallest_an_l513_51335

theorem smallest_an (n : ℕ+) (x : ℝ) :
  (((x^(2^(n.val+1)) + 1) / 2) ^ (1 / (2^n.val))) ≤ 2^(n.val-1) * (x-1)^2 + x :=
sorry

end smallest_an_l513_51335


namespace circle_equation_l513_51356

/-- A circle C with given properties -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (center_in_first_quadrant : center.1 > 0 ∧ center.2 > 0)
  (tangent_to_line : |4 * center.1 - 3 * center.2| = 5 * radius)
  (tangent_to_x_axis : center.2 = radius)
  (radius_is_one : radius = 1)

/-- The standard equation of a circle -/
def standard_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Theorem: The standard equation of the circle C is (x-2)^2 + (y-1)^2 = 1 -/
theorem circle_equation (c : Circle) :
  ∀ x y : ℝ, standard_equation c x y ↔ (x - 2)^2 + (y - 1)^2 = 1 :=
sorry

end circle_equation_l513_51356


namespace expand_expression_l513_51394

theorem expand_expression (x y z : ℝ) : 
  (x + 10 + y) * (2 * z + 10) = 2 * x * z + 2 * y * z + 10 * x + 10 * y + 20 * z + 100 := by
  sorry

end expand_expression_l513_51394


namespace digit_of_fraction_l513_51364

/-- The fraction we're considering -/
def f : ℚ := 66 / 1110

/-- The index of the digit we're looking for (0-indexed) -/
def n : ℕ := 221

/-- The function that returns the nth digit after the decimal point
    in the decimal representation of a rational number -/
noncomputable def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem digit_of_fraction :
  nth_digit_after_decimal f n = 5 := by sorry

end digit_of_fraction_l513_51364


namespace inequality_proof_l513_51319

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) :
  x^2 * y^2 + |x^2 - y^2| ≤ π/2 := by
  sorry

end inequality_proof_l513_51319


namespace det_specific_matrix_l513_51388

theorem det_specific_matrix : 
  Matrix.det !![2, 0, 4; 3, -1, 5; 1, 2, 3] = 2 := by
  sorry

end det_specific_matrix_l513_51388


namespace inequality_solution_set_l513_51316

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, x^2 + x - m > 0 ↔ x < -3 ∨ x > 2) → m = 6 := by
sorry

end inequality_solution_set_l513_51316


namespace circle_C2_equation_l513_51387

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

-- Define circle C1
def circle_C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define circle C2
def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

-- Define symmetry relation
def symmetric_point (x y x' y' : ℝ) : Prop :=
  line_of_symmetry ((x + x') / 2) ((y + y') / 2) ∧
  (x - x')^2 + (y - y')^2 = 2 * ((x - y - 1)^2)

-- Theorem statement
theorem circle_C2_equation :
  ∀ x y : ℝ, circle_C2 x y ↔
  ∃ x' y' : ℝ, circle_C1 x' y' ∧ symmetric_point x y x' y' :=
sorry

end circle_C2_equation_l513_51387


namespace range_of_x_l513_51345

def is_monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f y ≤ f x

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem range_of_x (f : ℝ → ℝ) 
  (h1 : is_monotone_decreasing f)
  (h2 : is_odd_function f)
  (h3 : f 1 = -1)
  (h4 : ∀ x, -1 ≤ f (x - 2) ∧ f (x - 2) ≤ 1) :
  ∀ x, -1 ≤ f (x - 2) ∧ f (x - 2) ≤ 1 → 1 ≤ x ∧ x ≤ 3 :=
sorry

end range_of_x_l513_51345


namespace regular_polygon_with_20_degree_exterior_angle_l513_51362

theorem regular_polygon_with_20_degree_exterior_angle (n : ℕ) : 
  n > 2 → (360 : ℝ) / n = 20 → n = 18 := by
  sorry

end regular_polygon_with_20_degree_exterior_angle_l513_51362


namespace floor_abs_negative_real_l513_51317

theorem floor_abs_negative_real : ⌊|(-25.7 : ℝ)|⌋ = 25 := by sorry

end floor_abs_negative_real_l513_51317


namespace property_P_implies_m_range_l513_51376

open Real

/-- Property P(a) for a function f -/
def has_property_P (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ h : ℝ → ℝ, (∀ x > 1, h x > 0) ∧
    (∀ x > 1, deriv f x = h x * (x^2 - a*x + 1))

theorem property_P_implies_m_range
  (g : ℝ → ℝ) (hg : has_property_P g 2)
  (x₁ x₂ : ℝ) (hx : 1 < x₁ ∧ x₁ < x₂)
  (m : ℝ) (α β : ℝ)
  (hα : α = m*x₁ + (1-m)*x₂)
  (hβ : β = (1-m)*x₁ + m*x₂)
  (hαβ : α > 1 ∧ β > 1)
  (hineq : |g α - g β| < |g x₁ - g x₂|) :
  0 < m ∧ m < 1 :=
sorry

end property_P_implies_m_range_l513_51376


namespace class_size_l513_51300

/-- The number of chocolate bars Gerald brings --/
def gerald_bars : ℕ := 7

/-- The number of squares in each chocolate bar --/
def squares_per_bar : ℕ := 8

/-- The number of additional bars the teacher brings for each of Gerald's bars --/
def teacher_multiplier : ℕ := 2

/-- The number of squares each student gets --/
def squares_per_student : ℕ := 7

/-- The total number of chocolate bars --/
def total_bars : ℕ := gerald_bars + gerald_bars * teacher_multiplier

/-- The total number of chocolate squares --/
def total_squares : ℕ := total_bars * squares_per_bar

/-- The number of students in the class --/
def num_students : ℕ := total_squares / squares_per_student

theorem class_size : num_students = 24 := by
  sorry

end class_size_l513_51300


namespace equidistant_point_on_x_axis_l513_51349

theorem equidistant_point_on_x_axis : ∃ x : ℝ, 
  (x^2 + 6*x + 9 = x^2 + 25) ∧ (x = 8/3) := by
  sorry

end equidistant_point_on_x_axis_l513_51349


namespace peter_has_320_dollars_l513_51379

-- Define the friends' money amounts
def john_money : ℝ := 160
def peter_money : ℝ := 2 * john_money
def quincy_money : ℝ := peter_money + 20
def andrew_money : ℝ := 1.15 * quincy_money

-- Define the total money and expenses
def total_money : ℝ := john_money + peter_money + quincy_money + andrew_money
def item_cost : ℝ := 1200
def money_left : ℝ := 11

-- Theorem to prove
theorem peter_has_320_dollars :
  peter_money = 320 ∧
  john_money + peter_money + quincy_money + andrew_money = item_cost + money_left :=
by sorry

end peter_has_320_dollars_l513_51379


namespace james_two_point_shots_l513_51375

/-- Represents the number of 2-point shots scored by James -/
def two_point_shots : ℕ := sorry

/-- Represents the number of 3-point shots scored by James -/
def three_point_shots : ℕ := 13

/-- Represents the total points scored by James -/
def total_points : ℕ := 79

/-- Theorem stating that James scored 20 two-point shots -/
theorem james_two_point_shots : 
  two_point_shots = 20 ∧ 
  2 * two_point_shots + 3 * three_point_shots = total_points := by
  sorry

end james_two_point_shots_l513_51375


namespace cross_country_race_winning_scores_l513_51311

/-- Represents a cross-country race with two teams -/
structure CrossCountryRace where
  /-- The number of players in each team -/
  players_per_team : Nat
  /-- The total number of players in the race -/
  total_players : Nat
  /-- The sum of all possible scores in the race -/
  total_score : Nat

/-- Calculates the maximum possible score for the winning team -/
def max_winning_score (race : CrossCountryRace) : Nat :=
  race.total_score / 2

/-- Calculates the minimum possible score for any team -/
def min_team_score (race : CrossCountryRace) : Nat :=
  List.sum (List.range race.players_per_team)

/-- The number of possible scores for the winning team -/
def winning_score_count (race : CrossCountryRace) : Nat :=
  max_winning_score race - min_team_score race + 1

/-- Theorem stating the number of possible scores for the winning team in a specific cross-country race -/
theorem cross_country_race_winning_scores :
  ∃ (race : CrossCountryRace),
    race.players_per_team = 5 ∧
    race.total_players = 10 ∧
    race.total_score = (race.total_players * (race.total_players + 1)) / 2 ∧
    winning_score_count race = 13 := by
  sorry


end cross_country_race_winning_scores_l513_51311


namespace original_equals_scientific_l513_51381

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  mantissa_bounds : 1 ≤ mantissa ∧ mantissa < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 274000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { mantissa := 2.74
  , exponent := 8
  , mantissa_bounds := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.mantissa * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end original_equals_scientific_l513_51381


namespace least_sum_of_equal_multiples_l513_51353

theorem least_sum_of_equal_multiples (x y z : ℕ+) (h : (2 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (8 : ℕ) * z.val) :
  x.val + y.val + z.val ≥ 33 ∧ ∃ (a b c : ℕ+), (2 : ℕ) * a.val = (5 : ℕ) * b.val ∧ (5 : ℕ) * b.val = (8 : ℕ) * c.val ∧ a.val + b.val + c.val = 33 :=
by
  sorry

#check least_sum_of_equal_multiples

end least_sum_of_equal_multiples_l513_51353


namespace min_value_theorem_l513_51389

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + r

-- Define the conditions of the problem
def problem_conditions (a : ℕ → ℝ) : Prop :=
  arithmetic_sequence a ∧
  (∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + r) ∧
  a 2018 = a 2017 + 2 * a 2016 ∧
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ a m * a n = 16 * (a 1)^2

-- State the theorem
theorem min_value_theorem (a : ℕ → ℝ) :
  problem_conditions a →
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ 4/m + 1/n ≥ 5/3 ∧
  (∀ k l : ℕ, k > 0 → l > 0 → 4/k + 1/l ≥ 4/m + 1/n) :=
sorry

end min_value_theorem_l513_51389


namespace max_plus_min_of_f_l513_51397

def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem max_plus_min_of_f : 
  ∃ (m n : ℝ), (∀ x, f x ≤ m) ∧ (∃ x₁, f x₁ = m) ∧ 
               (∀ x, n ≤ f x) ∧ (∃ x₂, f x₂ = n) ∧ 
               m + n = 2 := by
  sorry

end max_plus_min_of_f_l513_51397


namespace bricklayer_electrician_problem_l513_51391

theorem bricklayer_electrician_problem :
  ∀ (bricklayer_rate electrician_rate total_pay bricklayer_hours : ℝ),
    bricklayer_rate = 12 →
    electrician_rate = 16 →
    total_pay = 1350 →
    bricklayer_hours = 67.5 →
    ∃ (electrician_hours : ℝ),
      electrician_hours = (total_pay - bricklayer_rate * bricklayer_hours) / electrician_rate ∧
      bricklayer_hours + electrician_hours = 101.25 :=
by sorry

end bricklayer_electrician_problem_l513_51391


namespace circle_C_equation_line_l_equation_l513_51338

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the line x - y + 1 = 0
def line_1 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the x-axis
def x_axis (y : ℝ) : Prop := y = 0

-- Define the line x + y + 3 = 0
def line_2 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the circle x^2 + (y - 3)^2 = 4
def circle_C2 (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x = -1 ∨ 4*x - 3*y + 4 = 0

-- Theorem 1
theorem circle_C_equation : 
  ∀ x y : ℝ, 
  (∃ x₀, line_1 x₀ 0 ∧ x_axis 0) → 
  (∀ x₁ y₁, line_2 x₁ y₁ → ∃ t, circle_C (x₁ + t) (y₁ + t) ∧ ¬(∃ s ≠ t, circle_C (x₁ + s) (y₁ + s))) →
  circle_C x y :=
sorry

-- Theorem 2
theorem line_l_equation :
  ∀ x y : ℝ,
  circle_C2 x y →
  (∃ x₀ y₀, x₀ = -1 ∧ y₀ = 0 ∧ line_l x₀ y₀) →
  (∃ p q : ℝ × ℝ, circle_C2 p.1 p.2 ∧ circle_C2 q.1 q.2 ∧ line_l p.1 p.2 ∧ line_l q.1 q.2 ∧ (p.1 - q.1)^2 + (p.2 - q.2)^2 = 12) →
  line_l x y :=
sorry

end circle_C_equation_line_l_equation_l513_51338


namespace intersection_line_parabola_l513_51332

/-- The line y = kx - 2 intersects the parabola y² = 8x at two points A and B,
    and the x-coordinate of the midpoint of AB is 2. Then k = 2. -/
theorem intersection_line_parabola (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧
    (∀ x y, (x, y) = A ∨ (x, y) = B → y = k * x - 2 ∧ y^2 = 8 * x) ∧
    (A.1 + B.1) / 2 = 2) →
  k = 2 := by
sorry


end intersection_line_parabola_l513_51332


namespace max_small_boxes_in_large_box_l513_51396

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Converts meters to centimeters -/
def metersToCentimeters (m : ℕ) : ℕ := m * 100

/-- The dimensions of the large wooden box in meters -/
def largeBoxDimensionsMeters : BoxDimensions := {
  length := 8,
  width := 10,
  height := 6
}

/-- The dimensions of the large wooden box in centimeters -/
def largeBoxDimensionsCm : BoxDimensions := {
  length := metersToCentimeters largeBoxDimensionsMeters.length,
  width := metersToCentimeters largeBoxDimensionsMeters.width,
  height := metersToCentimeters largeBoxDimensionsMeters.height
}

/-- The dimensions of the small rectangular box in centimeters -/
def smallBoxDimensions : BoxDimensions := {
  length := 4,
  width := 5,
  height := 6
}

/-- Theorem: The maximum number of small boxes that can fit in the large box is 4,000,000 -/
theorem max_small_boxes_in_large_box :
  (boxVolume largeBoxDimensionsCm) / (boxVolume smallBoxDimensions) = 4000000 := by
  sorry

end max_small_boxes_in_large_box_l513_51396


namespace three_digit_number_is_142_l513_51323

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Converts a repeating decimal of the form 0.xyxy̅xy to a fraction -/
def repeating_decimal_xy (x y : Digit) : ℚ :=
  (10 * x.val + y.val : ℚ) / 99

/-- Converts a repeating decimal of the form 0.xyzxyz̅xyz to a fraction -/
def repeating_decimal_xyz (x y z : Digit) : ℚ :=
  (100 * x.val + 10 * y.val + z.val : ℚ) / 999

/-- The main theorem stating that the three-digit number xyz is 142 -/
theorem three_digit_number_is_142 :
  ∃ (x y z : Digit),
    repeating_decimal_xy x y + repeating_decimal_xyz x y z = 39 / 41 ∧
    x.val = 1 ∧ y.val = 4 ∧ z.val = 2 := by
  sorry

end three_digit_number_is_142_l513_51323


namespace fraction_equality_l513_51333

theorem fraction_equality (A B : ℤ) : 
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ -5 ∧ x ≠ 2 → 
    (A / (x + 2) + B / (x^2 - 4*x - 5) = (x^2 + x + 7) / (x^3 + 6*x^2 - 13*x - 10))) → 
  B / A = -1 := by
sorry

end fraction_equality_l513_51333


namespace download_rate_proof_l513_51314

/-- Proves that the download rate for the first 60 megabytes is 5 megabytes per second -/
theorem download_rate_proof (file_size : ℝ) (first_part_size : ℝ) (second_part_rate : ℝ) (total_time : ℝ)
  (h1 : file_size = 90)
  (h2 : first_part_size = 60)
  (h3 : second_part_rate = 10)
  (h4 : total_time = 15)
  (h5 : file_size = first_part_size + (file_size - first_part_size))
  (h6 : total_time = first_part_size / R + (file_size - first_part_size) / second_part_rate) :
  R = 5 := by
  sorry

#check download_rate_proof

end download_rate_proof_l513_51314


namespace lorenzo_board_test_l513_51363

/-- The number of boards Lorenzo tested -/
def boards_tested : ℕ := 120

/-- The total number of thumbtacks Lorenzo started with -/
def total_thumbtacks : ℕ := 450

/-- The number of cans of thumbtacks -/
def number_of_cans : ℕ := 3

/-- The number of thumbtacks remaining in each can at the end of the day -/
def remaining_thumbtacks_per_can : ℕ := 30

/-- The number of thumbtacks used per board -/
def thumbtacks_per_board : ℕ := 3

theorem lorenzo_board_test :
  boards_tested = (total_thumbtacks - number_of_cans * remaining_thumbtacks_per_can) / thumbtacks_per_board :=
by sorry

end lorenzo_board_test_l513_51363


namespace inequality_condition_l513_51305

theorem inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c :=
by sorry

end inequality_condition_l513_51305


namespace exists_perfect_square_with_digit_sum_2011_l513_51346

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a perfect square with sum of digits 2011 -/
theorem exists_perfect_square_with_digit_sum_2011 : 
  ∃ n : ℕ, sum_of_digits (n^2) = 2011 := by
sorry

end exists_perfect_square_with_digit_sum_2011_l513_51346


namespace division_multiplication_equality_l513_51306

theorem division_multiplication_equality : (180 / 6) * 3 = 90 := by
  sorry

end division_multiplication_equality_l513_51306


namespace card_width_is_15_l513_51303

/-- A rectangular card with a given perimeter and width-length relationship -/
structure RectangularCard where
  length : ℝ
  width : ℝ
  perimeter_eq : length * 2 + width * 2 = 46
  width_length_rel : width = length + 7

/-- The width of the rectangular card is 15 cm -/
theorem card_width_is_15 (card : RectangularCard) : card.width = 15 := by
  sorry

end card_width_is_15_l513_51303


namespace point_in_third_quadrant_l513_51385

theorem point_in_third_quadrant (x y : ℝ) (h1 : x + y < 0) (h2 : x * y > 0) : x < 0 ∧ y < 0 := by
  sorry

end point_in_third_quadrant_l513_51385


namespace max_large_sculptures_l513_51354

theorem max_large_sculptures (total_blocks : ℕ) (small_sculptures large_sculptures : ℕ) : 
  total_blocks = 30 →
  small_sculptures > large_sculptures →
  small_sculptures + 3 * large_sculptures + (small_sculptures + large_sculptures) / 2 ≤ total_blocks →
  large_sculptures ≤ 4 :=
by sorry

end max_large_sculptures_l513_51354


namespace factorization_of_4x_squared_plus_x_l513_51301

theorem factorization_of_4x_squared_plus_x (x : ℝ) : 4 * x^2 + x = x * (4 * x + 1) := by
  sorry

end factorization_of_4x_squared_plus_x_l513_51301


namespace cricket_bat_profit_percentage_l513_51350

/-- Calculates the profit percentage for a cricket bat sale --/
theorem cricket_bat_profit_percentage 
  (selling_price : ℝ) 
  (initial_profit : ℝ) 
  (tax_rate : ℝ) 
  (discount_rate : ℝ) 
  (h1 : selling_price = 850)
  (h2 : initial_profit = 255)
  (h3 : tax_rate = 0.07)
  (h4 : discount_rate = 0.05) : 
  ∃ (profit_percentage : ℝ), abs (profit_percentage - 25.71) < 0.01 :=
by
  sorry

end cricket_bat_profit_percentage_l513_51350


namespace prop_a_necessary_not_sufficient_l513_51321

theorem prop_a_necessary_not_sufficient :
  (∃ a : ℝ, a < 2 ∧ a^2 ≥ 4) ∧
  (∀ a : ℝ, a^2 < 4 → a < 2) :=
by sorry

end prop_a_necessary_not_sufficient_l513_51321


namespace consecutive_non_prime_powers_l513_51398

/-- A number is a prime power if it can be expressed as p^k where p is prime and k ≥ 1 -/
def IsPrimePower (n : ℕ) : Prop :=
  ∃ (p k : ℕ), Prime p ∧ k ≥ 1 ∧ n = p^k

theorem consecutive_non_prime_powers (N : ℕ) (h : N > 0) :
  ∃ (M : ℤ), ∀ (i : ℕ), i < N → ¬IsPrimePower (Int.toNat (M + i)) :=
sorry

end consecutive_non_prime_powers_l513_51398


namespace triangle_tangent_relation_l513_51328

theorem triangle_tangent_relation (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < π / 2) ∧
  (0 < B) ∧ (B < π / 2) ∧
  (0 < C) ∧ (C < π / 2) ∧
  (A + B + C = π) ∧
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  (a / Real.sin A = b / Real.sin B) ∧
  (b / Real.sin B = c / Real.sin C) ∧
  (c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) ∧
  (Real.tan A * Real.tan B = Real.tan A * Real.tan C + Real.tan C * Real.tan B) →
  (a^2 + b^2) / c^2 = 3 := by
sorry

end triangle_tangent_relation_l513_51328


namespace quadratic_two_distinct_roots_l513_51374

theorem quadratic_two_distinct_roots (a : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 + 2*a*x₁ + a^2 - 1 = 0 ∧ 
  x₂^2 + 2*a*x₂ + a^2 - 1 = 0 :=
sorry

end quadratic_two_distinct_roots_l513_51374


namespace max_value_a_sqrt_1_plus_b_sq_l513_51359

theorem max_value_a_sqrt_1_plus_b_sq (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (heq : a^2 / 2 + b^2 = 4) :
  ∃ (max : ℝ), max = (5 * Real.sqrt 2) / 2 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → x^2 / 2 + y^2 = 4 → 
  x * Real.sqrt (1 + y^2) ≤ max :=
by sorry

end max_value_a_sqrt_1_plus_b_sq_l513_51359


namespace line_through_points_l513_51348

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the line passing through two given points -/
def line_equation (p₀ p₁ : Point) : ℝ → Prop :=
  fun y => y = p₀.y

/-- The theorem states that the line equation y = 2 passes through the given points -/
theorem line_through_points :
  let p₀ : Point := ⟨1, 2⟩
  let p₁ : Point := ⟨3, 2⟩
  let eq := line_equation p₀ p₁
  (eq 2) ∧ (p₀.y = 2) ∧ (p₁.y = 2) := by sorry

end line_through_points_l513_51348


namespace complex_inequalities_l513_51370

theorem complex_inequalities :
  (∀ z w : ℂ, Complex.abs z + Complex.abs w ≤ Complex.abs (z + w) + Complex.abs (z - w)) ∧
  (∀ z₁ z₂ z₃ z₄ : ℂ, 
    Complex.abs z₁ + Complex.abs z₂ + Complex.abs z₃ + Complex.abs z₄ ≤
    Complex.abs (z₁ + z₂) + Complex.abs (z₁ + z₃) + Complex.abs (z₁ + z₄) +
    Complex.abs (z₂ + z₃) + Complex.abs (z₂ + z₄) + Complex.abs (z₃ + z₄)) := by
  sorry

end complex_inequalities_l513_51370


namespace cone_surface_area_l513_51373

/-- The surface area of a cone, given its lateral surface properties -/
theorem cone_surface_area (r : Real) (arc_length : Real) : 
  r = 4 → arc_length = 4 * Real.pi → 
  (π * (arc_length / (2 * π))^2) + (1/2 * r * arc_length) = 12 * π := by
sorry

end cone_surface_area_l513_51373


namespace negation_of_forall_geq_zero_l513_51361

theorem negation_of_forall_geq_zero :
  (¬ ∀ x : ℝ, 2 * x + 4 ≥ 0) ↔ (∃ x : ℝ, 2 * x + 4 < 0) := by
  sorry

end negation_of_forall_geq_zero_l513_51361


namespace solution_systems_l513_51368

-- System a
def system_a (x y : ℝ) : Prop :=
  x + y + x*y = 5 ∧ x*y*(x + y) = 6

-- System b
def system_b (x y : ℝ) : Prop :=
  x^3 + y^3 + 2*x*y = 4 ∧ x^2 - x*y + y^2 = 1

theorem solution_systems :
  (∃ x y : ℝ, system_a x y ∧ ((x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2))) ∧
  (∃ x y : ℝ, system_b x y ∧ x = 1 ∧ y = 1) := by
  sorry

end solution_systems_l513_51368


namespace f_min_value_inequality_property_l513_51378

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Theorem for the minimum value of f
theorem f_min_value :
  (∀ x : ℝ, f x ≥ 4) ∧ (∃ x : ℝ, f x = 4) := by sorry

-- Theorem for the inequality
theorem inequality_property (a b x : ℝ) (ha : |a| < 2) (hb : |b| < 2) :
  |a + b| + |a - b| < f x := by sorry

end f_min_value_inequality_property_l513_51378


namespace delta_phi_equation_l513_51341

def δ (x : ℝ) : ℝ := 3 * x + 8

def φ (x : ℝ) : ℝ := 8 * x + 7

theorem delta_phi_equation (x : ℝ) : δ (φ x) = 7 ↔ x = -11/12 := by sorry

end delta_phi_equation_l513_51341


namespace montoya_family_food_budget_l513_51308

theorem montoya_family_food_budget (grocery_fraction eating_out_fraction : ℝ) 
  (h1 : grocery_fraction = 0.6)
  (h2 : eating_out_fraction = 0.2) :
  grocery_fraction + eating_out_fraction = 0.8 := by
  sorry

end montoya_family_food_budget_l513_51308


namespace complement_of_M_in_U_l513_51312

universe u

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def M : Finset ℕ := {1, 3, 5}

theorem complement_of_M_in_U :
  (U \ M) = {2, 4, 6} := by sorry

end complement_of_M_in_U_l513_51312


namespace floor_times_self_eq_90_l513_51325

theorem floor_times_self_eq_90 (x : ℝ) (h1 : x > 0) (h2 : ⌊x⌋ * x = 90) : x = 10 := by
  sorry

end floor_times_self_eq_90_l513_51325


namespace one_of_each_color_probability_l513_51395

/-- The probability of selecting one marble of each color from a bag with 3 red, 3 blue, and 3 green marbles -/
theorem one_of_each_color_probability : 
  let total_marbles : ℕ := 3 + 3 + 3
  let marbles_per_color : ℕ := 3
  let selected_marbles : ℕ := 3
  (marbles_per_color ^ selected_marbles : ℚ) / (Nat.choose total_marbles selected_marbles) = 9 / 28 :=
by sorry

end one_of_each_color_probability_l513_51395


namespace det_cofactor_matrix_cube_l513_51310

/-- For a 4x4 matrix A, the determinant of its cofactor matrix B is equal to the cube of the determinant of A. -/
theorem det_cofactor_matrix_cube (A : Matrix (Fin 4) (Fin 4) ℝ) :
  let d := Matrix.det A
  let B := Matrix.adjugate A
  Matrix.det B = d^3 := by sorry

end det_cofactor_matrix_cube_l513_51310


namespace money_difference_l513_51327

/-- The amount of money Gwen received from her dad -/
def money_from_dad : ℕ := 5

/-- The amount of money Gwen received from her mom -/
def money_from_mom : ℕ := 7

/-- The difference between the amount Gwen received from her mom and her dad -/
def difference : ℕ := money_from_mom - money_from_dad

theorem money_difference : difference = 2 := by
  sorry

end money_difference_l513_51327


namespace complement_union_theorem_l513_51383

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {0, 1, 2}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 1, 2, 3} := by
  sorry

end complement_union_theorem_l513_51383


namespace shape_count_theorem_l513_51324

/-- Represents the count of shapes in a box -/
structure ShapeCount where
  triangles : ℕ
  squares : ℕ
  circles : ℕ

/-- Checks if a ShapeCount satisfies the given conditions -/
def isValidShapeCount (sc : ShapeCount) : Prop :=
  sc.triangles + sc.squares + sc.circles = 24 ∧
  sc.triangles = 7 * sc.squares

/-- The set of all possible valid shape counts -/
def validShapeCounts : Set ShapeCount :=
  { sc | isValidShapeCount sc }

/-- The theorem stating the only possible combinations -/
theorem shape_count_theorem :
  validShapeCounts = {
    ⟨0, 0, 24⟩,
    ⟨7, 1, 16⟩,
    ⟨14, 2, 8⟩,
    ⟨21, 3, 0⟩
  } := by sorry

end shape_count_theorem_l513_51324


namespace semi_annual_annuity_payment_l513_51366

/-- Calculates the semi-annual annuity payment given the following conditions:
  * Initial annual payment of 2500 HUF
  * Payment duration of 15 years
  * No collection for first 5 years
  * Convert to semi-annual annuity lasting 20 years, starting at beginning of 6th year
  * Annual interest rate of 4.75%
-/
def calculate_semi_annual_annuity (
  initial_payment : ℝ
  ) (payment_duration : ℕ
  ) (no_collection_years : ℕ
  ) (annuity_duration : ℕ
  ) (annual_interest_rate : ℝ
  ) : ℝ :=
  sorry

/-- The semi-annual annuity payment is approximately 2134.43 HUF -/
theorem semi_annual_annuity_payment :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |calculate_semi_annual_annuity 2500 15 5 20 0.0475 - 2134.43| < ε :=
sorry

end semi_annual_annuity_payment_l513_51366


namespace exterior_angle_regular_hexagon_l513_51372

theorem exterior_angle_regular_hexagon :
  let n : ℕ := 6  -- Number of sides in a hexagon
  let sum_interior_angles : ℝ := 180 * (n - 2)  -- Sum of interior angles formula
  let interior_angle : ℝ := sum_interior_angles / n  -- Each interior angle in a regular polygon
  let exterior_angle : ℝ := 180 - interior_angle  -- Exterior angle is supplementary to interior angle
  exterior_angle = 60 := by sorry

end exterior_angle_regular_hexagon_l513_51372


namespace sequence_problem_l513_51313

/-- The sequence function F that generates the nth term of the sequence --/
def F : ℕ → ℚ := sorry

/-- The sum of the first n natural numbers --/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that F(16) = 1/6 and F(4952) = 2/99 --/
theorem sequence_problem :
  F 16 = 1 / 6 ∧ F 4952 = 2 / 99 := by sorry

end sequence_problem_l513_51313


namespace mildred_oranges_l513_51382

/-- The number of oranges Mildred ends up with after a series of operations -/
def final_oranges (initial : ℕ) : ℕ :=
  let from_father := 3 * initial
  let after_father := initial + from_father
  let after_sister := after_father - 174
  2 * after_sister

/-- Theorem stating that given an initial collection of 215 oranges, 
    Mildred ends up with 1372 oranges after the described operations -/
theorem mildred_oranges : final_oranges 215 = 1372 := by
  sorry

end mildred_oranges_l513_51382


namespace total_passengers_in_hour_l513_51334

/-- Calculates the total number of different passengers stepping on and off trains at a station within an hour -/
def total_passengers (train_frequency : ℕ) (passengers_leaving : ℕ) (passengers_boarding : ℕ) : ℕ :=
  let trains_per_hour := 60 / train_frequency
  let passengers_per_train := passengers_leaving + passengers_boarding
  trains_per_hour * passengers_per_train

/-- Proves that given the specified conditions, the total number of different passengers in an hour is 6240 -/
theorem total_passengers_in_hour :
  total_passengers 5 200 320 = 6240 := by
  sorry

end total_passengers_in_hour_l513_51334


namespace article_cost_l513_51307

/-- The cost of an article given specific selling price conditions -/
theorem article_cost (selling_price_high : ℝ) (selling_price_low : ℝ) 
  (price_difference : ℝ) (gain_difference_percent : ℝ) :
  selling_price_high = 350 →
  selling_price_low = 340 →
  price_difference = selling_price_high - selling_price_low →
  gain_difference_percent = 5 →
  price_difference = (gain_difference_percent / 100) * 200 →
  200 = price_difference / (gain_difference_percent / 100) :=
by sorry

end article_cost_l513_51307


namespace simplify_and_evaluate_l513_51352

theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  (1 + 1 / (x + 1)) / ((x^2 + 4*x + 4) / (x + 1)) = -1 := by
  sorry

end simplify_and_evaluate_l513_51352


namespace profit_percentage_l513_51365

theorem profit_percentage (selling_price cost_price : ℝ) 
  (h : cost_price = 0.92 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100 / 92 - 1) * 100 := by
  sorry

end profit_percentage_l513_51365


namespace first_part_segments_second_part_segments_l513_51347

/-- Number of segments after cutting loops in a Chinese knot --/
def segments_after_cutting (loops : ℕ) (wings : ℕ := 1) : ℕ :=
  (loops * 2 * wings + wings) / wings

/-- Theorem for the first part of the problem --/
theorem first_part_segments : segments_after_cutting 5 = 6 := by sorry

/-- Theorem for the second part of the problem --/
theorem second_part_segments : segments_after_cutting 7 2 = 15 := by sorry

end first_part_segments_second_part_segments_l513_51347


namespace julia_tag_game_l513_51322

theorem julia_tag_game (monday_kids tuesday_kids : ℕ) : 
  monday_kids = 22 → 
  monday_kids = tuesday_kids + 8 → 
  tuesday_kids = 14 := by
sorry

end julia_tag_game_l513_51322


namespace linear_equation_solve_l513_51342

theorem linear_equation_solve (x y : ℝ) :
  2 * x - 7 * y = 5 → y = (2 * x - 5) / 7 := by
  sorry

end linear_equation_solve_l513_51342


namespace inequality_equivalence_l513_51357

theorem inequality_equivalence (x : ℝ) : x - 1 > 0 ↔ x > 1 := by
  sorry

end inequality_equivalence_l513_51357


namespace credit_card_balance_l513_51371

theorem credit_card_balance (B : ℝ) : 
  (1.44 * B + 24 = 96) → B = 50 := by
sorry

end credit_card_balance_l513_51371


namespace fraction_order_l513_51392

theorem fraction_order (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hac : a < c) (hbd : b > d) :
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d ∧
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < (a + c) / (b - d) ∧
  (c - a) / (b + d) < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d ∧
  (c - a) / (b + d) < (a + c) / (b + d) ∧ (a + c) / (b + d) < (a + c) / (b - d) ∧
  (c - a) / (b + d) < (c - a) / (b - d) ∧ (c - a) / (b - d) < (a + c) / (b - d) :=
by sorry

end fraction_order_l513_51392


namespace factor_quadratic_l513_51336

theorem factor_quadratic (x t : ℝ) : 
  (x - t) ∣ (10 * x^2 + 23 * x - 7) ↔ 
  t = (-23 + Real.sqrt 809) / 20 ∨ t = (-23 - Real.sqrt 809) / 20 := by
  sorry

end factor_quadratic_l513_51336


namespace probability_increasing_maxima_correct_l513_51399

/-- The probability that the maximum numbers in each row of a triangular array
    are in strictly increasing order. -/
def probability_increasing_maxima (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / (n + 1).factorial

/-- Theorem stating that the probability of increasing maxima in a triangular array
    with n rows is equal to 2^n / (n+1)! -/
theorem probability_increasing_maxima_correct (n : ℕ) :
  let array_size := n * (n + 1) / 2
  probability_increasing_maxima n =
    (2 ^ n : ℚ) / (n + 1).factorial :=
by sorry

end probability_increasing_maxima_correct_l513_51399


namespace circumradius_inradius_ratio_irrational_l513_51344

-- Define a lattice point
def LatticePoint := ℤ × ℤ

-- Define a triangle with lattice points as vertices
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

-- Define a square-free natural number
def SquareFree (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m * m ∣ n → m = 1

-- Define the property that one side of the triangle has length √n
def HasSqrtNSide (t : LatticeTriangle) (n : ℕ) : Prop :=
  SquareFree n ∧
  (((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 : ℚ) = n ∨
   ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2 : ℚ) = n ∨
   ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2 : ℚ) = n)

-- Define the circumradius and inradius of a triangle
noncomputable def circumradius (t : LatticeTriangle) : ℝ := sorry
noncomputable def inradius (t : LatticeTriangle) : ℝ := sorry

-- The main theorem
theorem circumradius_inradius_ratio_irrational (t : LatticeTriangle) (n : ℕ) :
  HasSqrtNSide t n → ¬ (∃ q : ℚ, (circumradius t / inradius t : ℝ) = q) :=
sorry

end circumradius_inradius_ratio_irrational_l513_51344


namespace census_suitability_l513_51329

/-- Represents a survey --/
structure Survey where
  description : String
  population_size : Nat
  ease_of_survey : Bool

/-- Defines when a survey is suitable for a census --/
def suitable_for_census (s : Survey) : Prop :=
  s.population_size < 1000 ∧ s.ease_of_survey

/-- Theorem stating the condition for a survey to be suitable for a census --/
theorem census_suitability (s : Survey) :
  suitable_for_census s ↔ s.population_size < 1000 ∧ s.ease_of_survey := by sorry

end census_suitability_l513_51329


namespace janet_action_figures_l513_51384

/-- Calculates the final number of action figures Janet has --/
def final_action_figures (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  let after_selling := initial - sold
  let after_buying := after_selling + bought
  let brothers_collection := 2 * after_buying
  after_buying + brothers_collection

/-- Theorem stating that Janet ends up with 24 action figures --/
theorem janet_action_figures :
  final_action_figures 10 6 4 = 24 := by
  sorry

end janet_action_figures_l513_51384


namespace max_large_chips_l513_51386

/-- The smallest composite number -/
def smallest_composite : ℕ := 4

/-- Represents the problem of finding the maximum number of large chips -/
def chip_problem (total : ℕ) (small : ℕ) (large : ℕ) : Prop :=
  total = 60 ∧
  small + large = total ∧
  ∃ c : ℕ, c ≥ smallest_composite ∧ small = large + c

/-- The theorem stating the maximum number of large chips -/
theorem max_large_chips :
  ∀ total small large,
  chip_problem total small large →
  large ≤ 28 :=
sorry

end max_large_chips_l513_51386


namespace remainder_evaluation_l513_51358

-- Define the remainder function
def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

-- State the theorem
theorem remainder_evaluation :
  rem (-1/7 : ℚ) (1/3 : ℚ) = 4/21 := by
  sorry

end remainder_evaluation_l513_51358


namespace largest_product_bound_l513_51343

theorem largest_product_bound (a : Fin 1985 → ℕ) 
  (h_perm : Function.Bijective a) 
  (h_range : ∀ i, a i ∈ Finset.range 1986) : 
  (Finset.range 1985).sup (λ k => (k + 1) * a k) ≥ 993^2 := by
  sorry

end largest_product_bound_l513_51343


namespace root_sum_theorem_l513_51330

theorem root_sum_theorem (p q r s : ℂ) : 
  p^4 - 15*p^3 + 35*p^2 - 27*p + 9 = 0 →
  q^4 - 15*q^3 + 35*q^2 - 27*q + 9 = 0 →
  r^4 - 15*r^3 + 35*r^2 - 27*r + 9 = 0 →
  s^4 - 15*s^3 + 35*s^2 - 27*s + 9 = 0 →
  p / (1/p + q*r) + q / (1/q + r*s) + r / (1/r + s*p) + s / (1/s + p*q) = 155/123 := by
sorry

end root_sum_theorem_l513_51330


namespace alpha_range_l513_51320

theorem alpha_range (α : Real) (h1 : 0 ≤ α) (h2 : α ≤ 2 * Real.pi) 
  (h3 : Real.sin α > Real.sqrt 3 * Real.cos α) : 
  Real.pi / 3 < α ∧ α < 4 * Real.pi / 3 := by
  sorry

end alpha_range_l513_51320


namespace min_brown_eyes_and_lunch_box_l513_51367

theorem min_brown_eyes_and_lunch_box 
  (total_students : ℕ) 
  (brown_eyes : ℕ) 
  (lunch_box : ℕ) 
  (h1 : total_students = 25) 
  (h2 : brown_eyes = 15) 
  (h3 : lunch_box = 18) :
  (brown_eyes + lunch_box - total_students : ℕ) ≥ 8 := by
  sorry

end min_brown_eyes_and_lunch_box_l513_51367


namespace import_tax_threshold_l513_51326

/-- The amount in excess of which the import tax was applied -/
def X : ℝ := 1000

/-- The total value of the item -/
def total_value : ℝ := 2580

/-- The import tax rate -/
def tax_rate : ℝ := 0.07

/-- The amount of import tax paid -/
def tax_paid : ℝ := 110.60

/-- Theorem stating that X is the correct amount in excess of which the import tax was applied -/
theorem import_tax_threshold (X total_value tax_rate tax_paid : ℝ) 
  (h1 : total_value = 2580)
  (h2 : tax_rate = 0.07)
  (h3 : tax_paid = 110.60) :
  X = 1000 ∧ tax_rate * (total_value - X) = tax_paid :=
by sorry

end import_tax_threshold_l513_51326
