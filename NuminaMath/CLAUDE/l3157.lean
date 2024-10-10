import Mathlib

namespace reciprocal_sum_theorem_l3157_315750

theorem reciprocal_sum_theorem :
  (∀ (a b c : ℕ+), (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ ≠ 9/11) ∧
  (∀ (a b c : ℕ+), (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ > 41/42 →
    (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ ≥ 1) := by sorry

end reciprocal_sum_theorem_l3157_315750


namespace fraction_denominator_problem_l3157_315716

theorem fraction_denominator_problem (y x : ℝ) (h1 : y > 0) 
  (h2 : (1 * y) / x + (3 * y) / 10 = 0.35 * y) : x = 20 := by
  sorry

end fraction_denominator_problem_l3157_315716


namespace smallest_gcd_of_20m_25n_l3157_315765

theorem smallest_gcd_of_20m_25n (m n : ℕ+) (h : Nat.gcd m.val n.val = 18) :
  ∃ (m₀ n₀ : ℕ+), Nat.gcd m₀.val n₀.val = 18 ∧
    Nat.gcd (20 * m₀.val) (25 * n₀.val) = 90 ∧
    ∀ (m' n' : ℕ+), Nat.gcd m'.val n'.val = 18 →
      Nat.gcd (20 * m'.val) (25 * n'.val) ≥ 90 := by
  sorry

end smallest_gcd_of_20m_25n_l3157_315765


namespace arithmetic_sequence_sum_l3157_315757

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℚ := sorry

/-- The first term of the arithmetic sequence -/
def a₁ : ℚ := sorry

/-- The common difference of the arithmetic sequence -/
def d : ℚ := sorry

/-- Properties of the arithmetic sequence -/
axiom sum_formula (n : ℕ) : S n = n * a₁ + (n * (n - 1) / 2) * d

/-- Given conditions -/
axiom condition_1 : S 10 = 16
axiom condition_2 : S 100 - S 90 = 24

/-- Theorem to prove -/
theorem arithmetic_sequence_sum : S 100 = 200 := by sorry

end arithmetic_sequence_sum_l3157_315757


namespace weight_11_25m_l3157_315778

/-- Represents the weight of a uniform rod given its length -/
def rod_weight (length : ℝ) : ℝ := sorry

/-- The rod is uniform, meaning its weight is proportional to its length -/
axiom rod_uniform (l₁ l₂ : ℝ) : l₁ * rod_weight l₂ = l₂ * rod_weight l₁

/-- The weight of 6 meters of the rod is 22.8 kg -/
axiom weight_6m : rod_weight 6 = 22.8

/-- Theorem: If 6 m of a uniform rod weighs 22.8 kg, then 11.25 m weighs 42.75 kg -/
theorem weight_11_25m : rod_weight 11.25 = 42.75 := by sorry

end weight_11_25m_l3157_315778


namespace multiplication_table_odd_fraction_l3157_315740

theorem multiplication_table_odd_fraction :
  let table_size : ℕ := 16
  let is_odd (n : ℕ) := n % 2 = 1
  let total_products := table_size * table_size
  let odd_products := (table_size / 2) * (table_size / 2)
  odd_products / total_products = (1 : ℚ) / 4 := by
sorry

end multiplication_table_odd_fraction_l3157_315740


namespace circle_equation_l3157_315731

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line 4x + 3y = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4 * p.1 + 3 * p.2 = 0}

-- Define the y-axis
def YAxis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0}

-- Theorem statement
theorem circle_equation :
  ∀ (C : Set (ℝ × ℝ)) (center : ℝ × ℝ),
    -- Conditions
    C = Circle center 1 →  -- Radius is 1
    center.1 < 0 ∧ center.2 > 0 →  -- Center is in second quadrant
    ∃ (p : ℝ × ℝ), p ∈ C ∩ Line →  -- Tangent to 4x + 3y = 0
    ∃ (q : ℝ × ℝ), q ∈ C ∩ YAxis →  -- Tangent to y-axis
    -- Conclusion
    C = {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 3)^2 = 1} :=
by sorry

end circle_equation_l3157_315731


namespace complex_multiplication_problem_l3157_315775

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Definition of the complex number multiplication -/
def complex_mult (a b c d : ℝ) : ℂ := Complex.mk (a * c - b * d) (a * d + b * c)

/-- The problem statement -/
theorem complex_multiplication_problem :
  complex_mult 4 (-3) 4 3 = 25 := by sorry

end complex_multiplication_problem_l3157_315775


namespace adams_shelves_l3157_315706

/-- The number of action figures that can fit on each shelf -/
def action_figures_per_shelf : ℕ := 11

/-- The total number of action figures that can be held by all shelves -/
def total_action_figures : ℕ := 44

/-- The number of shelves in Adam's room -/
def number_of_shelves : ℕ := total_action_figures / action_figures_per_shelf

/-- Theorem stating that the number of shelves in Adam's room is 4 -/
theorem adams_shelves : number_of_shelves = 4 := by
  sorry

end adams_shelves_l3157_315706


namespace cube_sum_over_product_l3157_315736

theorem cube_sum_over_product (x y z : ℝ) :
  ((x - y)^3 + (y - z)^3 + (z - x)^3) / (15 * (x - y) * (y - z) * (z - x)) = 1/5 :=
by sorry

end cube_sum_over_product_l3157_315736


namespace remainder_5432876543_mod_101_l3157_315714

theorem remainder_5432876543_mod_101 : 5432876543 % 101 = 79 := by
  sorry

end remainder_5432876543_mod_101_l3157_315714


namespace hyperbola_eccentricity_l3157_315709

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let C : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (c, 0)
  let A : ℝ × ℝ := (x₀, y₀)
  (∃ x₀ y₀, C (x₀, y₀) ∧ (x₀ * b)^2 = (y₀ * a)^2) →  -- A is on the asymptote
  (x₀^2 + y₀^2 = c^2 / 4) →  -- A is on the circle with diameter OF
  (Real.cos (π/6) * c = b) →  -- ∠AFO = π/6
  c / a = 2 :=  -- eccentricity is 2
by
  sorry


end hyperbola_eccentricity_l3157_315709


namespace union_of_A_and_B_l3157_315767

def A : Set Int := {-1, 1, 2}
def B : Set Int := {-2, -1, 0}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1, 2} := by sorry

end union_of_A_and_B_l3157_315767


namespace division_equality_l3157_315792

theorem division_equality : 815472 / 6630 = 123 := by
  sorry

end division_equality_l3157_315792


namespace pen_rubber_length_difference_l3157_315710

/-- Given a rubber, pen, and pencil with certain length relationships,
    prove that the pen is 3 cm longer than the rubber. -/
theorem pen_rubber_length_difference :
  ∀ (rubber_length pen_length pencil_length : ℝ),
    pencil_length = 12 →
    pen_length = pencil_length - 2 →
    rubber_length + pen_length + pencil_length = 29 →
    pen_length - rubber_length = 3 :=
by sorry

end pen_rubber_length_difference_l3157_315710


namespace total_candy_pieces_l3157_315776

def chocolate_boxes : ℕ := 2
def caramel_boxes : ℕ := 5
def pieces_per_box : ℕ := 4

theorem total_candy_pieces : 
  (chocolate_boxes + caramel_boxes) * pieces_per_box = 28 := by
  sorry

end total_candy_pieces_l3157_315776


namespace infinite_solutions_condition_l3157_315743

theorem infinite_solutions_condition (c : ℝ) : 
  (∀ x, 5 * (3 * x - c) = 3 * (5 * x + 20)) ↔ c = -12 := by sorry

end infinite_solutions_condition_l3157_315743


namespace smallest_gcd_yz_l3157_315742

theorem smallest_gcd_yz (x y z : ℕ+) (h1 : Nat.gcd x y = 210) (h2 : Nat.gcd x z = 770) :
  ∃ (y' z' : ℕ+), Nat.gcd x y' = 210 ∧ Nat.gcd x z' = 770 ∧ Nat.gcd y' z' = 10 ∧
  ∀ (y'' z'' : ℕ+), Nat.gcd x y'' = 210 → Nat.gcd x z'' = 770 → Nat.gcd y'' z'' ≥ 10 :=
by sorry

end smallest_gcd_yz_l3157_315742


namespace parabola_focus_theorem_l3157_315763

/-- Parabola with equation y² = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- Point on the parabola -/
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * c.p * x

/-- Circle tangent to y-axis and intersecting MF -/
structure TangentCircle (c : Parabola) (m : PointOnParabola c) where
  a : ℝ × ℝ  -- Point A
  tangent_to_y_axis : sorry
  intersects_mf : sorry

/-- Theorem: Given the conditions, p = 2 -/
theorem parabola_focus_theorem (c : Parabola) 
    (m : PointOnParabola c)
    (h_m : m.y = 2 * Real.sqrt 2)
    (circle : TangentCircle c m)
    (h_ratio : (Real.sqrt ((m.x - circle.a.1)^2 + (m.y - circle.a.2)^2)) / 
               (Real.sqrt ((c.p - circle.a.1)^2 + circle.a.2^2)) = 2) :
  c.p = 2 := by
  sorry

end parabola_focus_theorem_l3157_315763


namespace part_one_part_two_l3157_315702

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → f a x ≤ 4) → 
  -1 ≤ a ∧ a ≤ 2 :=
sorry

-- Part II
theorem part_two (a : ℝ) :
  (∃ x : ℝ, f a (x - a) - f a (x + a) ≤ 2 * a - 1) → 
  a ≥ 1/4 :=
sorry

end part_one_part_two_l3157_315702


namespace range_of_m_l3157_315752

/-- Set A is defined as the set of real numbers x where -3 ≤ x ≤ 4 -/
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}

/-- Set B is defined as the set of real numbers x where 1 < x < m, and m > 1 -/
def B (m : ℝ) : Set ℝ := {x : ℝ | 1 < x ∧ x < m}

/-- The theorem states that if B is a subset of A and m > 1, then 1 < m ≤ 4 -/
theorem range_of_m (m : ℝ) (h1 : B m ⊆ A) (h2 : 1 < m) : 1 < m ∧ m ≤ 4 := by
  sorry

end range_of_m_l3157_315752


namespace candy_distribution_l3157_315760

theorem candy_distribution (total_candies : ℕ) (sour_percentage : ℚ) (num_people : ℕ) : 
  total_candies = 300 → 
  sour_percentage = 40 / 100 → 
  num_people = 3 → 
  (total_candies - (sour_percentage * total_candies).floor) / num_people = 60 := by
  sorry

end candy_distribution_l3157_315760


namespace polynomial_ratio_l3157_315770

/-- Given a polynomial ax^4 + bx^3 + cx^2 + dx + e = 0 with roots 1, 2, 3, and 4,
    prove that c/e = 35/24 -/
theorem polynomial_ratio (a b c d e : ℝ) (h : ∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) :
  c / e = 35 / 24 := by
  sorry

end polynomial_ratio_l3157_315770


namespace fraction_equality_l3157_315777

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) : 
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -4 / 9 := by
  sorry

end fraction_equality_l3157_315777


namespace binomial_expansion_sum_l3157_315784

theorem binomial_expansion_sum (n : ℕ) : 
  (∃ p q : ℕ, p = (3 + 1)^n ∧ q = 2^n ∧ p + q = 272) → n = 4 := by
  sorry

end binomial_expansion_sum_l3157_315784


namespace lower_price_calculation_l3157_315790

/-- The lower selling price of an article -/
def lower_price : ℚ := 348

/-- The higher selling price of an article -/
def higher_price : ℚ := 350

/-- The cost price of the article -/
def cost_price : ℚ := 40

/-- The percentage difference in profit between the two selling prices -/
def profit_difference_percentage : ℚ := 5 / 100

theorem lower_price_calculation :
  (higher_price - cost_price) = (lower_price - cost_price) + profit_difference_percentage * cost_price :=
by sorry

end lower_price_calculation_l3157_315790


namespace no_strictly_increasing_sequence_with_addition_property_l3157_315700

theorem no_strictly_increasing_sequence_with_addition_property :
  ¬ ∃ (a : ℕ → ℕ), 
    (∀ n m : ℕ, a (n * m) = a n + a m) ∧ 
    (∀ n : ℕ, a n < a (n + 1)) :=
by sorry

end no_strictly_increasing_sequence_with_addition_property_l3157_315700


namespace counterexample_exists_l3157_315707

theorem counterexample_exists : ∃ n : ℕ, 
  Nat.Prime n ∧ Even n ∧ ¬(Nat.Prime (n + 2)) := by
sorry

end counterexample_exists_l3157_315707


namespace other_number_is_99_l3157_315701

/-- Given two positive integers with specific HCF and LCM, prove one is 99 when the other is 48 -/
theorem other_number_is_99 (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 396) (h3 : a = 48) :
  b = 99 := by
  sorry

end other_number_is_99_l3157_315701


namespace inscribed_sphere_surface_area_l3157_315726

/-- The surface area of a sphere inscribed in a triangular pyramid with all edges of length a -/
theorem inscribed_sphere_surface_area (a : ℝ) (h : a > 0) :
  ∃ (r : ℝ), r > 0 ∧ r = a / (2 * Real.sqrt 6) ∧ 
  4 * Real.pi * r^2 = Real.pi * a^2 / 6 :=
sorry

end inscribed_sphere_surface_area_l3157_315726


namespace average_weight_of_all_girls_l3157_315766

theorem average_weight_of_all_girls (group1_count : ℕ) (group1_avg : ℝ) 
  (group2_count : ℕ) (group2_avg : ℝ) : 
  group1_count = 16 → 
  group1_avg = 50.25 → 
  group2_count = 8 → 
  group2_avg = 45.15 → 
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  let total_count := group1_count + group2_count
  (total_weight / total_count) = 48.55 := by
sorry

end average_weight_of_all_girls_l3157_315766


namespace bug_return_probability_l3157_315759

/-- Represents the probability of the bug being at its starting vertex after n moves -/
def P (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (1 - P (n - 1)) / 2

/-- The main theorem stating the probability of returning to the starting vertex on the 12th move -/
theorem bug_return_probability : P 12 = 683 / 2048 := by
  sorry

end bug_return_probability_l3157_315759


namespace square_2007_position_l3157_315799

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | DABC
  | CBAD
  | DCBA

-- Define the transformations
def rotate90Clockwise : SquarePosition → SquarePosition
  | SquarePosition.ABCD => SquarePosition.DABC
  | SquarePosition.DABC => SquarePosition.CBAD
  | SquarePosition.CBAD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

def reflectVertical : SquarePosition → SquarePosition
  | SquarePosition.ABCD => SquarePosition.DCBA
  | SquarePosition.DABC => SquarePosition.CBAD
  | SquarePosition.CBAD => SquarePosition.DABC
  | SquarePosition.DCBA => SquarePosition.ABCD

-- Define the transformation sequence
def transformSquare : Nat → SquarePosition → SquarePosition
  | 0, pos => pos
  | n + 1, pos => 
    if n % 2 == 0 
    then transformSquare n (rotate90Clockwise pos)
    else transformSquare n (reflectVertical pos)

-- Theorem to prove
theorem square_2007_position : 
  transformSquare 2007 SquarePosition.ABCD = SquarePosition.CBAD := by
  sorry

end square_2007_position_l3157_315799


namespace jellybean_count_jellybean_problem_l3157_315764

theorem jellybean_count (normal_class_size : ℕ) (absent_children : ℕ) 
  (jellybeans_per_child : ℕ) (remaining_jellybeans : ℕ) : ℕ :=
  let present_children := normal_class_size - absent_children
  let eaten_jellybeans := present_children * jellybeans_per_child
  eaten_jellybeans + remaining_jellybeans

theorem jellybean_problem : 
  jellybean_count 24 2 3 34 = 100 := by
  sorry

end jellybean_count_jellybean_problem_l3157_315764


namespace games_lost_l3157_315785

theorem games_lost (total_games won_games : ℕ) 
  (h1 : total_games = 16) 
  (h2 : won_games = 12) : 
  total_games - won_games = 4 := by
sorry

end games_lost_l3157_315785


namespace sufficient_not_necessary_l3157_315755

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 → a^2 > 1) ∧ (∃ a, a^2 > 1 ∧ ¬(a > 1)) :=
sorry

end sufficient_not_necessary_l3157_315755


namespace train_bridge_crossing_time_l3157_315780

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 150)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 225) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

#check train_bridge_crossing_time

end train_bridge_crossing_time_l3157_315780


namespace barbara_candies_l3157_315795

/-- The number of candies Barbara bought -/
def candies_bought (initial : ℕ) (total : ℕ) : ℕ := total - initial

theorem barbara_candies :
  candies_bought 9 27 = 18 :=
by sorry

end barbara_candies_l3157_315795


namespace seventh_term_of_geometric_sequence_l3157_315768

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometricSequenceTerm (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

theorem seventh_term_of_geometric_sequence :
  let a₁ : ℚ := 3
  let a₂ : ℚ := -1/2
  let r : ℚ := a₂ / a₁
  let a₇ : ℚ := geometricSequenceTerm a₁ r 7
  a₇ = 1/15552 :=
by sorry

end seventh_term_of_geometric_sequence_l3157_315768


namespace group_size_proof_l3157_315774

/-- Proves that the number of members in a group is 54, given the conditions of the problem -/
theorem group_size_proof (n : ℕ) : 
  (n : ℚ) * n = 2916 → n = 54 := by
  sorry

end group_size_proof_l3157_315774


namespace log_relation_l3157_315753

theorem log_relation (y : ℝ) (k : ℝ) : 
  (Real.log 4 / Real.log 8 = y) → 
  (Real.log 81 / Real.log 2 = k * y) → 
  k = 6 := by
sorry

end log_relation_l3157_315753


namespace ascent_speed_l3157_315727

/-- Given a journey with ascent and descent, calculate the average speed during ascent -/
theorem ascent_speed
  (total_time : ℝ)
  (overall_speed : ℝ)
  (ascent_time : ℝ)
  (h_total_time : total_time = 6)
  (h_overall_speed : overall_speed = 3.5)
  (h_ascent_time : ascent_time = 4)
  (h_equal_distance : ∀ d : ℝ, d = overall_speed * total_time / 2) :
  ∃ (ascent_speed : ℝ), ascent_speed = 2.625 ∧ ascent_speed = (overall_speed * total_time / 2) / ascent_time :=
by sorry

end ascent_speed_l3157_315727


namespace unique_digit_solution_l3157_315738

theorem unique_digit_solution :
  ∃! (digits : Fin 5 → Nat),
    (∀ i, digits i ≠ 0 ∧ digits i ≤ 9) ∧
    (digits 0 + digits 1 = (digits 2 + digits 3 + digits 4) / 7) ∧
    (digits 0 + digits 3 = (digits 1 + digits 2 + digits 4) / 5) := by
  sorry

end unique_digit_solution_l3157_315738


namespace crescent_area_implies_square_area_l3157_315773

/-- Given a square with side length s, the area of 8 "crescent" shapes formed by
    semicircles on its sides and the sides of its inscribed square (formed by
    connecting midpoints) is equal to πs². If this area is 5 square centimeters,
    then the area of the original square is 10 square centimeters. -/
theorem crescent_area_implies_square_area :
  ∀ s : ℝ,
  s > 0 →
  π * s^2 = 5 →
  s^2 = 10 :=
by sorry

end crescent_area_implies_square_area_l3157_315773


namespace figure_squares_l3157_315704

-- Define the sequence function
def f (n : ℕ) : ℕ := 2 * n^2 + 2 * n + 1

-- State the theorem
theorem figure_squares (n : ℕ) : 
  f 0 = 1 ∧ f 1 = 5 ∧ f 2 = 13 ∧ f 3 = 25 → f 100 = 20201 := by
  sorry


end figure_squares_l3157_315704


namespace factorization_problems_l3157_315718

variable (m x y : ℝ)

theorem factorization_problems :
  (mx^2 - m*y = m*(x^2 - y)) ∧
  (2*x^2 - 8*x + 8 = 2*(x-2)^2) ∧
  (x^2*(2*x-1) + y^2*(1-2*x) = (2*x-1)*(x+y)*(x-y)) := by
  sorry

end factorization_problems_l3157_315718


namespace richter_frequency_ratio_l3157_315782

/-- Represents the energy released for a given Richter scale reading -/
def energy_released (x : ℝ) : ℝ := sorry

/-- The Richter scale property: a reading of x - 1 indicates one-tenth the released energy as x -/
axiom richter_scale_property (x : ℝ) : energy_released (x - 1) = (1 / 10) * energy_released x

/-- The frequency corresponding to a given Richter scale reading -/
def frequency (x : ℝ) : ℝ := sorry

/-- Theorem stating the relationship between frequencies for Richter scale readings 5 and 3 -/
theorem richter_frequency_ratio : frequency 5 = 100 * frequency 3 := by sorry

end richter_frequency_ratio_l3157_315782


namespace solution_for_a_l3157_315728

theorem solution_for_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (eq1 : a + 1/b = 5) (eq2 : b + 1/a = 10) : 
  a = (5 + Real.sqrt 23) / 2 ∨ a = (5 - Real.sqrt 23) / 2 := by
  sorry

end solution_for_a_l3157_315728


namespace taxi_growth_equation_l3157_315748

def initial_taxis : ℕ := 11720
def final_taxis : ℕ := 13116
def years : ℕ := 2

theorem taxi_growth_equation (x : ℝ) : 
  (initial_taxis : ℝ) * (1 + x)^years = final_taxis ↔ 
  x = ((final_taxis : ℝ) / initial_taxis)^(1 / years : ℝ) - 1 :=
by sorry

end taxi_growth_equation_l3157_315748


namespace stating_probability_same_district_l3157_315772

/-- Represents the four districts available for housing applications. -/
inductive District : Type
  | A
  | B
  | C
  | D

/-- The number of districts available. -/
def num_districts : ℕ := 4

/-- Represents an application scenario for two applicants. -/
def ApplicationScenario : Type := District × District

/-- The total number of possible application scenarios for two applicants. -/
def total_scenarios : ℕ := num_districts * num_districts

/-- Predicate to check if two applicants applied for the same district. -/
def same_district (scenario : ApplicationScenario) : Prop :=
  scenario.1 = scenario.2

/-- The number of scenarios where two applicants apply for the same district. -/
def num_same_district_scenarios : ℕ := num_districts

/-- 
Theorem stating that the probability of two applicants choosing the same district
is 1/4, given that there are four equally likely choices for each applicant.
-/
theorem probability_same_district :
  (num_same_district_scenarios : ℚ) / total_scenarios = 1 / 4 := by
  sorry


end stating_probability_same_district_l3157_315772


namespace special_word_count_l3157_315798

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering --/
def word_length : ℕ := 5

/-- 
  Counts the number of five-letter words where:
  - The first and last letters are the same
  - The second and fourth letters are the same
--/
def count_special_words : ℕ := alphabet_size ^ 3

/-- 
  Theorem: The number of five-letter words with the given properties
  is equal to the cube of the alphabet size.
--/
theorem special_word_count :
  count_special_words = alphabet_size ^ 3 := by
  sorry

end special_word_count_l3157_315798


namespace five_students_four_lectures_l3157_315721

/-- The number of ways students can choose lectures -/
def lecture_choices (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: 5 students choosing from 4 lectures results in 4^5 choices -/
theorem five_students_four_lectures :
  lecture_choices 5 4 = 4^5 := by
  sorry

end five_students_four_lectures_l3157_315721


namespace codes_lost_l3157_315786

/-- The number of digits in each code -/
def code_length : Nat := 5

/-- The number of possible digits (0 to 9) -/
def digit_options : Nat := 10

/-- The number of non-zero digits (1 to 9) -/
def nonzero_digit_options : Nat := 9

/-- The number of codes with leading zeros allowed -/
def codes_with_leading_zeros : Nat := digit_options ^ code_length

/-- The number of codes without leading zeros -/
def codes_without_leading_zeros : Nat := nonzero_digit_options * (digit_options ^ (code_length - 1))

theorem codes_lost (code_length : Nat) (digit_options : Nat) (nonzero_digit_options : Nat) 
  (codes_with_leading_zeros : Nat) (codes_without_leading_zeros : Nat) :
  codes_with_leading_zeros - codes_without_leading_zeros = 10000 := by
  sorry

end codes_lost_l3157_315786


namespace weekly_toy_production_l3157_315787

/-- A factory produces toys with the following conditions:
  * Workers work 5 days a week
  * Workers produce the same number of toys every day
  * Workers produce 1100 toys each day
-/
def toy_factory (days_per_week : ℕ) (toys_per_day : ℕ) : Prop :=
  days_per_week = 5 ∧ toys_per_day = 1100

/-- The number of toys produced in a week -/
def weekly_production (days_per_week : ℕ) (toys_per_day : ℕ) : ℕ :=
  days_per_week * toys_per_day

/-- Theorem: Under the given conditions, the factory produces 5500 toys in a week -/
theorem weekly_toy_production :
  ∀ (days_per_week toys_per_day : ℕ),
    toy_factory days_per_week toys_per_day →
    weekly_production days_per_week toys_per_day = 5500 :=
by
  sorry

end weekly_toy_production_l3157_315787


namespace all_terms_are_perfect_squares_l3157_315751

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 14 * a (n + 1) - a n - 4

theorem all_terms_are_perfect_squares :
  ∀ n : ℕ, ∃ s : ℤ, a n = s^2 := by
  sorry

end all_terms_are_perfect_squares_l3157_315751


namespace quadrilateral_area_theorem_l3157_315732

/-- Represents a quadrilateral ABCD with given side lengths and angles -/
structure Quadrilateral :=
  (AB BC CD DA : ℝ)
  (angleA angleD : ℝ)

/-- Calculates the area of the quadrilateral ABCD -/
def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating that for the given quadrilateral, its area is (47√3)/4 -/
theorem quadrilateral_area_theorem (q : Quadrilateral) 
  (h1 : q.AB = 5) 
  (h2 : q.BC = 7) 
  (h3 : q.CD = 3) 
  (h4 : q.DA = 4) 
  (h5 : q.angleA = 2 * π / 3) 
  (h6 : q.angleD = 2 * π / 3) : 
  area q = (47 * Real.sqrt 3) / 4 :=
sorry

end quadrilateral_area_theorem_l3157_315732


namespace unique_prime_in_sequence_l3157_315761

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def number (A : ℕ) : ℕ := 205100 + A

theorem unique_prime_in_sequence :
  ∃! A : ℕ, A < 10 ∧ is_prime (number A) ∧ number A = 205103 := by sorry

end unique_prime_in_sequence_l3157_315761


namespace shaded_area_is_eleven_l3157_315779

/-- Given a grid with rectangles of dimensions 2x3, 3x4, and 4x5, and two unshaded right-angled triangles
    with dimensions (base 12, height 4) and (base 3, height 2), the shaded area is 11. -/
theorem shaded_area_is_eleven :
  let grid_area := 2 * 3 + 3 * 4 + 4 * 5
  let triangle1_area := (12 * 4) / 2
  let triangle2_area := (3 * 2) / 2
  let shaded_area := grid_area - triangle1_area - triangle2_area
  shaded_area = 11 := by
sorry


end shaded_area_is_eleven_l3157_315779


namespace randy_blocks_problem_l3157_315720

theorem randy_blocks_problem (blocks_used blocks_left : ℕ) 
  (h1 : blocks_used = 36)
  (h2 : blocks_left = 23) :
  blocks_used + blocks_left = 59 := by
  sorry

end randy_blocks_problem_l3157_315720


namespace parking_savings_l3157_315749

-- Define the weekly and monthly rental rates
def weekly_rate : ℕ := 10
def monthly_rate : ℕ := 42

-- Define the number of weeks and months in a year
def weeks_per_year : ℕ := 52
def months_per_year : ℕ := 12

-- Define the yearly cost for weekly and monthly rentals
def yearly_cost_weekly : ℕ := weekly_rate * weeks_per_year
def yearly_cost_monthly : ℕ := monthly_rate * months_per_year

-- Theorem: The difference in yearly cost between weekly and monthly rentals is $16
theorem parking_savings : yearly_cost_weekly - yearly_cost_monthly = 16 := by
  sorry

end parking_savings_l3157_315749


namespace leftover_eggs_l3157_315712

/-- Given that there are 119 eggs to be packaged into cartons of 12 eggs each,
    prove that the number of eggs left over is 11. -/
theorem leftover_eggs : Int.mod 119 12 = 11 := by
  sorry

end leftover_eggs_l3157_315712


namespace sum_first_100_odd_integers_l3157_315708

/-- The sum of the first n positive odd integers -/
def sumFirstNOddIntegers (n : ℕ) : ℕ :=
  n * n

theorem sum_first_100_odd_integers :
  sumFirstNOddIntegers 100 = 10000 := by
  sorry

end sum_first_100_odd_integers_l3157_315708


namespace didi_fundraiser_total_l3157_315734

/-- Calculates the total amount raised by Didi for her local soup kitchen --/
theorem didi_fundraiser_total (num_cakes : ℕ) (slices_per_cake : ℕ) (price_per_slice : ℚ) 
  (donation1 : ℚ) (donation2 : ℚ) (donation3 : ℚ) (donation4 : ℚ) :
  num_cakes = 20 →
  slices_per_cake = 12 →
  price_per_slice = 1 →
  donation1 = 3/4 →
  donation2 = 1/2 →
  donation3 = 1/4 →
  donation4 = 1/10 →
  (num_cakes * slices_per_cake * price_per_slice) + 
  (num_cakes * slices_per_cake * (donation1 + donation2 + donation3 + donation4)) = 624 := by
sorry

end didi_fundraiser_total_l3157_315734


namespace janet_total_miles_l3157_315703

/-- Represents Janet's running schedule for a week -/
structure WeekSchedule where
  days : ℕ
  milesPerDay : ℕ

/-- Calculates the total miles run in a week -/
def weeklyMiles (schedule : WeekSchedule) : ℕ :=
  schedule.days * schedule.milesPerDay

/-- Janet's running schedule for three weeks -/
def janetSchedule : List WeekSchedule :=
  [{ days := 5, milesPerDay := 8 },
   { days := 4, milesPerDay := 10 },
   { days := 3, milesPerDay := 6 }]

/-- Theorem: Janet ran a total of 98 miles over the three weeks -/
theorem janet_total_miles :
  (janetSchedule.map weeklyMiles).sum = 98 := by
  sorry

end janet_total_miles_l3157_315703


namespace car_speed_ratio_l3157_315730

-- Define the variables and constants
variable (v : ℝ) -- speed of car A
variable (k : ℝ) -- speed multiplier for car B
variable (AB CD AD : ℝ) -- distances

-- Define the theorem
theorem car_speed_ratio (h1 : k > 1) (h2 : AD = AB / 2) (h3 : CD / AD = 1 / 2) : k = 2 := by
  sorry

end car_speed_ratio_l3157_315730


namespace projection_vector_l3157_315754

/-- Given two lines k and n in 2D space, prove that the vector (-6, 9) satisfies the conditions for the projection of DC onto the normal of line n. -/
theorem projection_vector : ∃ (w1 w2 : ℝ), w1 = -6 ∧ w2 = 9 ∧ w1 + w2 = 3 ∧ 
  ∃ (t s : ℝ),
    let k := λ t : ℝ => (2 + 3*t, 3 + 2*t)
    let n := λ s : ℝ => (1 + 3*s, 5 + 2*s)
    let C := k t
    let D := n s
    let normal_n := (-2, 3)
    ∃ (c : ℝ), (w1, w2) = c • normal_n :=
by sorry

end projection_vector_l3157_315754


namespace stratified_sampling_problem_l3157_315705

theorem stratified_sampling_problem (total : ℕ) (sample_size : ℕ) 
  (stratum_A : ℕ) (stratum_B : ℕ) (h1 : total = 1200) (h2 : sample_size = 120) 
  (h3 : stratum_A = 380) (h4 : stratum_B = 420) : 
  let stratum_C := total - stratum_A - stratum_B
  (sample_size * stratum_C) / total = 40 := by
sorry

end stratified_sampling_problem_l3157_315705


namespace jerry_shelf_items_after_changes_l3157_315762

/-- Represents the items on Jerry's shelf -/
structure ShelfItems where
  action_figures : ℕ
  books : ℕ
  video_games : ℕ

/-- Calculates the total number of items on the shelf -/
def total_items (items : ShelfItems) : ℕ :=
  items.action_figures + items.books + items.video_games

/-- Represents the changes made to the shelf items -/
structure ItemChanges where
  action_figures_added : ℕ
  books_removed : ℕ
  video_games_added : ℕ

/-- Applies changes to the shelf items -/
def apply_changes (items : ShelfItems) (changes : ItemChanges) : ShelfItems where
  action_figures := items.action_figures + changes.action_figures_added
  books := items.books - changes.books_removed
  video_games := items.video_games + changes.video_games_added

theorem jerry_shelf_items_after_changes :
  let initial_items : ShelfItems := ⟨4, 22, 10⟩
  let changes : ItemChanges := ⟨6, 5, 3⟩
  let final_items := apply_changes initial_items changes
  total_items final_items = 40 := by
  sorry

end jerry_shelf_items_after_changes_l3157_315762


namespace garden_area_l3157_315791

theorem garden_area (total_posts : ℕ) (post_spacing : ℝ) (longer_side_ratio : ℕ) :
  total_posts = 24 →
  post_spacing = 3 →
  longer_side_ratio = 3 →
  ∃ (short_side_posts long_side_posts : ℕ),
    short_side_posts > 1 ∧
    long_side_posts > 1 ∧
    long_side_posts = longer_side_ratio * short_side_posts ∧
    total_posts = 2 * short_side_posts + 2 * long_side_posts - 4 ∧
    (short_side_posts - 1) * post_spacing * (long_side_posts - 1) * post_spacing = 297 :=
by sorry

end garden_area_l3157_315791


namespace geometric_progression_middle_term_l3157_315789

theorem geometric_progression_middle_term 
  (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_geometric : b^2 = a * c) 
  (h_a : a = 5 + 2 * Real.sqrt 6) 
  (h_c : c = 5 - 2 * Real.sqrt 6) : 
  b = 1 := by
sorry

end geometric_progression_middle_term_l3157_315789


namespace polynomial_remainder_l3157_315735

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^4 - 18 * x^3 + 6 * x^2 - 4 * x + 30) % (2 * x - 4) = 30 := by
  sorry

end polynomial_remainder_l3157_315735


namespace range_of_a_l3157_315769

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x > 1 then Real.exp x - a * x^2 + x - 1 else sorry

-- State the theorem
theorem range_of_a :
  (∀ x, f a (-x) = -(f a x)) →  -- f is odd
  (∀ m : ℝ, m ≠ 0 → f a (1/m) * f a m = 1) →  -- property for non-zero m
  (∀ x, x > 1 → f a x = Real.exp x - a * x^2 + x - 1) →  -- definition for x > 1
  (∀ y : ℝ, ∃ x, f a x = y) →  -- range of f is R
  (∀ x, (x - 2) * Real.exp x - x + 4 > 0) →  -- given inequality
  a ∈ Set.Icc (Real.exp 1 - 1) ((Real.exp 2 + 1) / 4) :=
by sorry

end range_of_a_l3157_315769


namespace abc_encodes_to_57_l3157_315793

/-- Represents the set of characters used in the encoding -/
inductive EncodingChar : Type
  | A | B | C | D

/-- Represents a base 4 number as a list of EncodingChar -/
def Base4Number := List EncodingChar

/-- Converts a Base4Number to its decimal (base 10) representation -/
def toDecimal (n : Base4Number) : ℕ :=
  sorry

/-- Checks if three Base4Numbers are consecutive encodings -/
def areConsecutiveEncodings (a b c : Base4Number) : Prop :=
  sorry

/-- Main theorem: Given the conditions, ABC encodes to 57 in base 10 -/
theorem abc_encodes_to_57 
  (h : areConsecutiveEncodings 
    [EncodingChar.B, EncodingChar.C, EncodingChar.D]
    [EncodingChar.B, EncodingChar.C, EncodingChar.C]
    [EncodingChar.B, EncodingChar.D, EncodingChar.A]) :
  toDecimal [EncodingChar.A, EncodingChar.B, EncodingChar.C] = 57 := by
  sorry

end abc_encodes_to_57_l3157_315793


namespace zero_in_interval_l3157_315771

/-- Given two positive real numbers a and b where a > b > 0 and |log a| = |log b|,
    there exists an x in the interval (-1, 0) such that a^x + x - b = 0 -/
theorem zero_in_interval (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : |Real.log a| = |Real.log b|) :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ a^x + x - b = 0 := by
  sorry

end zero_in_interval_l3157_315771


namespace gravel_cost_calculation_l3157_315788

/-- The cost of gravel in dollars per cubic foot -/
def gravel_cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The volume of gravel in cubic yards -/
def gravel_volume_cubic_yards : ℝ := 3

/-- The cost of the given volume of gravel in dollars -/
def total_cost : ℝ := gravel_volume_cubic_yards * cubic_feet_per_cubic_yard * gravel_cost_per_cubic_foot

theorem gravel_cost_calculation : total_cost = 648 := by
  sorry

end gravel_cost_calculation_l3157_315788


namespace x_div_y_value_l3157_315733

theorem x_div_y_value (x y : ℝ) (h1 : |x| = 4) (h2 : |y| = 2) (h3 : x < y) :
  x / y = -2 := by
  sorry

end x_div_y_value_l3157_315733


namespace square_eq_product_sum_seven_l3157_315747

theorem square_eq_product_sum_seven (a b : ℕ) : a^2 = b * (b + 7) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end square_eq_product_sum_seven_l3157_315747


namespace second_month_sale_l3157_315715

/-- Proves that the sale in the second month is 10500 given the conditions of the problem -/
theorem second_month_sale (sales : Fin 6 → ℕ)
  (h1 : sales 0 = 2500)
  (h3 : sales 2 = 9855)
  (h4 : sales 3 = 7230)
  (h5 : sales 4 = 7000)
  (h6 : sales 5 = 11915)
  (avg : (sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = 7500) :
  sales 1 = 10500 := by
  sorry

#check second_month_sale

end second_month_sale_l3157_315715


namespace max_value_expression_l3157_315713

theorem max_value_expression (x y z : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y) (h₃ : 0 ≤ z) (h₄ : x^2 + y^2 + z^2 = 1) :
  3 * x * y * Real.sqrt 3 + 9 * y * z ≤ Real.sqrt 255 :=
by sorry

end max_value_expression_l3157_315713


namespace inequality_system_solution_l3157_315739

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x + 2 < 2*m ∧ x - m < 0) ↔ x < 2*m - 2) → m ≤ 2 := by
  sorry

end inequality_system_solution_l3157_315739


namespace warm_up_puzzle_time_l3157_315723

/-- Represents the time taken for the warm-up puzzle in minutes -/
def warm_up_time : ℝ := 10

/-- Represents the total number of puzzles solved -/
def total_puzzles : ℕ := 3

/-- Represents the total time spent solving all puzzles in minutes -/
def total_time : ℝ := 70

/-- Represents the time multiplier for the longer puzzles compared to the warm-up puzzle -/
def longer_puzzle_multiplier : ℝ := 3

/-- Represents the number of longer puzzles solved -/
def longer_puzzles : ℕ := 2

theorem warm_up_puzzle_time :
  warm_up_time * (1 + longer_puzzle_multiplier * longer_puzzles) = total_time :=
by sorry

end warm_up_puzzle_time_l3157_315723


namespace cordelia_hair_bleaching_l3157_315744

/-- The time it takes to bleach Cordelia's hair. -/
def bleaching_time : ℝ := 3

/-- The total time for the hair coloring process. -/
def total_time : ℝ := 9

/-- The relationship between dyeing time and bleaching time. -/
def dyeing_time (b : ℝ) : ℝ := 2 * b

theorem cordelia_hair_bleaching :
  bleaching_time + dyeing_time bleaching_time = total_time ∧
  bleaching_time = 3 := by
sorry

end cordelia_hair_bleaching_l3157_315744


namespace divisible_by_twelve_l3157_315719

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def last_two_digits (n : ℕ) : ℕ := n % 100

def six_digit_number (a b c d e f : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f

theorem divisible_by_twelve (square : ℕ) :
  is_divisible_by (six_digit_number 4 8 6 3 square 5) 12 ↔ square = 1 :=
sorry

end divisible_by_twelve_l3157_315719


namespace alex_coin_distribution_l3157_315745

def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let total_needed := (num_friends * (num_friends + 1)) / 2
  if total_needed > initial_coins then
    total_needed - initial_coins
  else
    0

theorem alex_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 90) :
  min_additional_coins num_friends initial_coins = 30 := by
  sorry

#eval min_additional_coins 15 90

end alex_coin_distribution_l3157_315745


namespace tv_price_decrease_l3157_315796

theorem tv_price_decrease (x : ℝ) : 
  (1 - x / 100) * (1 + 55 / 100) = 1 + 24 / 100 → x = 20 := by
  sorry

end tv_price_decrease_l3157_315796


namespace largest_sum_is_1803_l3157_315756

/-- The set of digits to be used -/
def digits : Finset Nat := {1, 2, 3, 7, 8, 9}

/-- A function that computes the sum of two 3-digit numbers -/
def sum_3digit (a b c d e f : Nat) : Nat :=
  100 * (a + d) + 10 * (b + e) + (c + f)

/-- The theorem stating that 1803 is the largest possible sum -/
theorem largest_sum_is_1803 :
  ∀ a b c d e f : Nat,
    a ∈ digits → b ∈ digits → c ∈ digits →
    d ∈ digits → e ∈ digits → f ∈ digits →
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f →
    b ≠ c → b ≠ d → b ≠ e → b ≠ f →
    c ≠ d → c ≠ e → c ≠ f →
    d ≠ e → d ≠ f →
    e ≠ f →
    sum_3digit a b c d e f ≤ 1803 :=
by
  sorry

end largest_sum_is_1803_l3157_315756


namespace repeating_decimal_ratio_l3157_315797

/-- Represents a repeating decimal with a 3-digit repetend -/
def RepeatingDecimal (whole : ℕ) (repetend : ℕ) : ℚ :=
  whole + (repetend : ℚ) / 999

theorem repeating_decimal_ratio : 
  (RepeatingDecimal 0 833) / (RepeatingDecimal 1 666) = 1 / 2 := by
  sorry

end repeating_decimal_ratio_l3157_315797


namespace second_class_average_l3157_315729

theorem second_class_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) : 
  n₁ = 30 →
  n₂ = 50 →
  avg₁ = 50 →
  avg_total = 56.25 →
  (n₁ * avg₁ + n₂ * (n₁ * avg₁ + n₂ * avg_total - n₁ * avg₁) / n₂) / (n₁ + n₂) = avg_total →
  (n₁ * avg₁ + n₂ * avg_total - n₁ * avg₁) / n₂ = 60 := by
sorry

end second_class_average_l3157_315729


namespace scientific_notation_3650000_l3157_315741

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Function to convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_3650000 :
  toScientificNotation 3650000 = ScientificNotation.mk 3.65 6 sorry := by
  sorry

end scientific_notation_3650000_l3157_315741


namespace partial_fraction_decomposition_l3157_315794

theorem partial_fraction_decomposition (a b c : ℤ) 
  (h1 : (1 : ℚ) / 2015 = a / 5 + b / 13 + c / 31)
  (h2 : 0 ≤ a ∧ a < 5)
  (h3 : 0 ≤ b ∧ b < 13) :
  a + b = 14 := by sorry

end partial_fraction_decomposition_l3157_315794


namespace exponent_transform_to_one_l3157_315717

theorem exponent_transform_to_one (a : ℝ) (h : a > 0) :
  ∀ x : ℝ, a^x = 1 ↔ x = 0 := by
sorry

end exponent_transform_to_one_l3157_315717


namespace stream_rate_l3157_315725

/-- The rate of a stream given boat speed and downstream travel information -/
theorem stream_rate (boat_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  boat_speed = 16 →
  distance = 168 →
  time = 8 →
  (boat_speed + (distance / time - boat_speed)) * time = distance →
  distance / time - boat_speed = 5 :=
by
  sorry

end stream_rate_l3157_315725


namespace elderly_sample_size_l3157_315781

/-- Given a population with elderly people, prove the number of elderly to be sampled -/
theorem elderly_sample_size
  (total_population : ℕ)
  (elderly_population : ℕ)
  (sample_size : ℕ)
  (h1 : total_population = 180)
  (h2 : elderly_population = 30)
  (h3 : sample_size = 36)
  : (elderly_population * sample_size) / total_population = 6 := by
  sorry

end elderly_sample_size_l3157_315781


namespace apples_given_to_neighbor_l3157_315711

theorem apples_given_to_neighbor (initial_apples remaining_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : remaining_apples = 39) :
  initial_apples - remaining_apples = 88 := by
  sorry

end apples_given_to_neighbor_l3157_315711


namespace vector_problems_l3157_315758

def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

theorem vector_problems (x : ℝ) :
  (∃ k : ℝ, a x = k • b x → ‖a x - b x‖ = 2 ∨ ‖a x - b x‖ = 2 * Real.sqrt 5) ∧
  (0 < (a x).1 * (b x).1 + (a x).2 * (b x).2 → x ∈ Set.Ioo (-1) 0 ∪ Set.Ioo 0 3) ∧
  (‖a x‖ = 2 → ∃ c : ℝ × ℝ, ‖c‖ = 1 ∧ (a x).1 * c.1 + (a x).2 * c.2 = 0 ∧
    ((c.1 = Real.sqrt 3 / 2 ∧ c.2 = -1/2) ∨
     (c.1 = -Real.sqrt 3 / 2 ∧ c.2 = 1/2) ∨
     (c.1 = Real.sqrt 3 / 2 ∧ c.2 = 1/2) ∨
     (c.1 = -Real.sqrt 3 / 2 ∧ c.2 = -1/2))) :=
by sorry


end vector_problems_l3157_315758


namespace special_permutations_l3157_315724

def word_length : ℕ := 7
def num_vowels : ℕ := 3
def num_consonants : ℕ := 4

theorem special_permutations :
  (word_length.choose num_vowels) * (num_consonants.factorial) = 840 := by
  sorry

end special_permutations_l3157_315724


namespace polygon_with_150_degree_interior_angles_has_12_sides_l3157_315737

theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    interior_angle = 150 →
    (n : ℝ) * (180 - interior_angle) = 360 →
    n = 12 := by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l3157_315737


namespace quadratic_inequality_solution_set_l3157_315746

/-- Given that the solution set of ax^2 + x + b > 0 with respect to x is (-1, 2), prove that a + b = 1 -/
theorem quadratic_inequality_solution_set (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + x + b > 0 ↔ -1 < x ∧ x < 2) → a + b = 1 := by
  sorry

end quadratic_inequality_solution_set_l3157_315746


namespace friends_behind_yuna_l3157_315722

theorem friends_behind_yuna (total_friends : ℕ) (friends_in_front : ℕ) : 
  total_friends = 6 → friends_in_front = 2 → total_friends - friends_in_front = 4 := by
  sorry

end friends_behind_yuna_l3157_315722


namespace marias_trip_l3157_315783

theorem marias_trip (total_distance : ℝ) (h1 : total_distance = 360) : 
  let first_stop := total_distance / 2
  let remaining_after_first := total_distance - first_stop
  let second_stop := remaining_after_first / 4
  let distance_after_second := remaining_after_first - second_stop
  distance_after_second = 135 := by
sorry

end marias_trip_l3157_315783
