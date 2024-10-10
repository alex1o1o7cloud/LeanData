import Mathlib

namespace empty_solution_set_implies_a_range_l1487_148739

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x - a > -3) → a ∈ Set.Ioo (-6) 2 := by
  sorry

end empty_solution_set_implies_a_range_l1487_148739


namespace max_profit_price_l1487_148791

/-- The cost of one item in yuan -/
def cost : ℝ := 30

/-- The number of items sold as a function of price -/
def itemsSold (x : ℝ) : ℝ := 200 - x

/-- The profit function -/
def profit (x : ℝ) : ℝ := (x - cost) * (itemsSold x)

/-- Theorem: The price that maximizes profit is 115 yuan -/
theorem max_profit_price : 
  ∃ (x : ℝ), x = 115 ∧ ∀ (y : ℝ), profit y ≤ profit x :=
sorry

end max_profit_price_l1487_148791


namespace distinct_prime_factors_count_l1487_148750

/-- Represents the number of accent options for each letter in "cesontoiseaux" --/
def accentOptions : List Nat := [2, 5, 5, 1, 1, 3, 3, 1, 1, 2, 3, 1, 4]

/-- The number of ways to split 12 letters into 3 words --/
def wordSplitOptions : Nat := 66

/-- Calculates the total number of possible phrases --/
def totalPhrases : Nat :=
  wordSplitOptions * (accentOptions.foldl (·*·) 1)

/-- Theorem stating that the number of distinct prime factors of totalPhrases is 4 --/
theorem distinct_prime_factors_count :
  (Nat.factors totalPhrases).toFinset.card = 4 := by sorry

end distinct_prime_factors_count_l1487_148750


namespace complement_of_complement_is_A_l1487_148721

-- Define the universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- Define the complement of A in U
def C_UA : Set ℕ := {5, 7}

-- Define set A
def A : Set ℕ := {1, 3, 9}

-- Theorem statement
theorem complement_of_complement_is_A :
  A = U \ C_UA :=
by sorry

end complement_of_complement_is_A_l1487_148721


namespace extreme_value_cubic_l1487_148774

/-- Given a cubic function f(x) = x^3 + ax^2 + bx with an extreme value of -2 at x = 1,
    prove that a + 2b = -6 -/
theorem extreme_value_cubic (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + b*x
  (f 1 = -2) ∧ (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1 ∨ f x ≤ f 1) →
  a + 2*b = -6 := by
sorry

end extreme_value_cubic_l1487_148774


namespace triangle_side_length_l1487_148730

/-- 
Given a triangle ABC where:
- a, b, c are sides opposite to angles A, B, C respectively
- A = 2π/3
- b = √2
- Area of triangle ABC is √3
Prove that a = √14
-/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 2 * Real.pi / 3 →
  b = Real.sqrt 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a = Real.sqrt 14 := by
  sorry


end triangle_side_length_l1487_148730


namespace mrs_petersons_change_l1487_148708

theorem mrs_petersons_change (number_of_tumblers : ℕ) (price_per_tumbler : ℕ) (number_of_bills : ℕ) (bill_value : ℕ) : 
  number_of_tumblers = 10 →
  price_per_tumbler = 45 →
  number_of_bills = 5 →
  bill_value = 100 →
  (number_of_bills * bill_value) - (number_of_tumblers * price_per_tumbler) = 50 :=
by
  sorry

#check mrs_petersons_change

end mrs_petersons_change_l1487_148708


namespace complex_sum_powers_of_i_l1487_148700

theorem complex_sum_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ i + i^2 + i^3 + i^4 = 0 :=
by sorry

end complex_sum_powers_of_i_l1487_148700


namespace tangent_line_polar_equation_l1487_148799

/-- Given a circle in polar form ρ = 4sinθ and a point (2√2, π/4),
    the polar equation of the tangent line passing through this point is ρcosθ = 2 -/
theorem tangent_line_polar_equation
  (ρ θ : ℝ) 
  (circle_eq : ρ = 4 * Real.sin θ) 
  (point : (ρ, θ) = (2 * Real.sqrt 2, Real.pi / 4)) :
  ∃ (k : ℝ), ρ * Real.cos θ = k ∧ k = 2 :=
sorry

end tangent_line_polar_equation_l1487_148799


namespace equality_of_pairs_l1487_148789

theorem equality_of_pairs (a b x y : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0)
  (h_sum : a + b + x + y < 2)
  (h_eq1 : a + b^2 = x + y^2)
  (h_eq2 : a^2 + b = x^2 + y) :
  a = x ∧ b = y := by
  sorry

end equality_of_pairs_l1487_148789


namespace intersection_of_A_and_B_l1487_148712

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_of_A_and_B_l1487_148712


namespace x_is_perfect_square_l1487_148784

theorem x_is_perfect_square (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x > y)
  (h_div : (x^2019 + x + y^2) % (x*y) = 0) : 
  ∃ (n : ℕ), x = n^2 := by
sorry

end x_is_perfect_square_l1487_148784


namespace power_of_two_equality_l1487_148741

theorem power_of_two_equality : (2^8)^5 = 2^8 * 2^32 := by sorry

end power_of_two_equality_l1487_148741


namespace polynomial_equality_l1487_148728

/-- Given a polynomial function q(x) satisfying the equation
    q(x) + (x^6 + 2x^4 + 5x^2 + 8x) = (3x^4 + 18x^3 + 20x^2 + 5x + 2),
    prove that q(x) = -x^6 + x^4 + 18x^3 + 15x^2 - 3x + 2 -/
theorem polynomial_equality (q : ℝ → ℝ) :
  (∀ x, q x + (x^6 + 2*x^4 + 5*x^2 + 8*x) = (3*x^4 + 18*x^3 + 20*x^2 + 5*x + 2)) →
  (∀ x, q x = -x^6 + x^4 + 18*x^3 + 15*x^2 - 3*x + 2) :=
by
  sorry

end polynomial_equality_l1487_148728


namespace diophantine_equation_solutions_l1487_148797

theorem diophantine_equation_solutions :
  ∀ (a b : ℕ), 2017^a = b^6 - 32*b + 1 ↔ (a = 0 ∧ b = 0) ∨ (a = 0 ∧ b = 2) :=
by sorry

end diophantine_equation_solutions_l1487_148797


namespace corral_area_ratio_l1487_148757

/-- The ratio of areas between four small square corrals and one large square corral -/
theorem corral_area_ratio (s : ℝ) (h : s > 0) : 
  (4 * s^2) / ((4 * s)^2) = 1 / 4 := by
  sorry

#check corral_area_ratio

end corral_area_ratio_l1487_148757


namespace pen_pencil_cost_ratio_l1487_148729

/-- Given a pen and pencil with a total cost of $6, where the pen costs $4,
    prove that the ratio of the cost of the pen to the cost of the pencil is 4:1. -/
theorem pen_pencil_cost_ratio :
  ∀ (pen_cost pencil_cost : ℚ),
  pen_cost + pencil_cost = 6 →
  pen_cost = 4 →
  pen_cost / pencil_cost = 4 := by
sorry

end pen_pencil_cost_ratio_l1487_148729


namespace rotary_club_eggs_l1487_148756

/-- Calculates the total number of eggs needed for the Rotary Club's Omelet Breakfast --/
def total_eggs_needed (small_children : ℕ) (older_children : ℕ) (adults : ℕ) (seniors : ℕ) 
  (waste_percent : ℚ) (extra_omelets : ℕ) (eggs_per_extra_omelet : ℚ) : ℕ :=
  let eggs_for_tickets := small_children + 2 * older_children + 3 * adults + 4 * seniors
  let waste_eggs := ⌈(eggs_for_tickets : ℚ) * waste_percent⌉
  let extra_omelet_eggs := ⌈(extra_omelets : ℚ) * eggs_per_extra_omelet⌉
  eggs_for_tickets + waste_eggs.toNat + extra_omelet_eggs.toNat

/-- Theorem stating the total number of eggs needed for the Rotary Club's Omelet Breakfast --/
theorem rotary_club_eggs : 
  total_eggs_needed 53 35 75 37 (3/100) 25 (5/2) = 574 := by
  sorry

end rotary_club_eggs_l1487_148756


namespace difference_of_squares_example_l1487_148737

theorem difference_of_squares_example : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end difference_of_squares_example_l1487_148737


namespace solution_value_l1487_148705

-- Define the equations
def equation1 (m x : ℝ) : Prop := (m + 3) * x^(|m| - 2) + 6 * m = 0
def equation2 (n x : ℝ) : Prop := n * x - 5 = x * (3 - n)

-- Define the linearity condition for equation1
def equation1_is_linear (m : ℝ) : Prop := |m| - 2 = 0

-- Define the main theorem
theorem solution_value (m n x : ℝ) :
  (∀ y : ℝ, equation1 m y ↔ equation2 n y) →
  equation1_is_linear m →
  (m + x)^2000 * (-m^2 * n + x * n^2) + 1 = 1 :=
sorry

end solution_value_l1487_148705


namespace boris_climbs_needed_l1487_148751

def hugo_elevation : ℕ := 10000
def boris_elevation : ℕ := hugo_elevation - 2500
def hugo_climbs : ℕ := 3

theorem boris_climbs_needed : 
  (hugo_elevation * hugo_climbs) / boris_elevation = 4 := by sorry

end boris_climbs_needed_l1487_148751


namespace rectangle_area_diagonal_l1487_148776

/-- Given a rectangle with length to width ratio of 5:4 and diagonal d, 
    its area A can be expressed as A = (20/41)d^2 -/
theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) : 
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ l / w = 5 / 4 ∧ l^2 + w^2 = d^2 ∧ l * w = (20/41) * d^2 := by
  sorry

end rectangle_area_diagonal_l1487_148776


namespace root_ratio_sum_l1487_148752

theorem root_ratio_sum (k₁ k₂ : ℝ) : 
  (∃ a b : ℝ, 3 * a^2 - (3 - k₁) * a + 7 = 0 ∧ 
              3 * b^2 - (3 - k₁) * b + 7 = 0 ∧ 
              a / b + b / a = 9 / 7) ∧
  (∃ a b : ℝ, 3 * a^2 - (3 - k₂) * a + 7 = 0 ∧ 
              3 * b^2 - (3 - k₂) * b + 7 = 0 ∧ 
              a / b + b / a = 9 / 7) →
  k₁ / k₂ + k₂ / k₁ = -20 / 7 := by
sorry

end root_ratio_sum_l1487_148752


namespace mean_median_difference_l1487_148715

/-- Represents the frequency distribution of days missed by students -/
def frequency_distribution : List (Nat × Nat) := [
  (0, 4),  -- 4 students missed 0 days
  (1, 2),  -- 2 students missed 1 day
  (2, 5),  -- 5 students missed 2 days
  (3, 2),  -- 2 students missed 3 days
  (4, 1),  -- 1 student missed 4 days
  (5, 3),  -- 3 students missed 5 days
  (6, 1)   -- 1 student missed 6 days
]

/-- Calculate the median of the distribution -/
def median (dist : List (Nat × Nat)) : Nat :=
  sorry

/-- Calculate the mean of the distribution -/
def mean (dist : List (Nat × Nat)) : Rat :=
  sorry

/-- The total number of students -/
def total_students : Nat := frequency_distribution.map (·.2) |>.sum

theorem mean_median_difference :
  mean frequency_distribution - median frequency_distribution = 0 ∧ total_students = 18 := by
  sorry

end mean_median_difference_l1487_148715


namespace intersection_of_A_and_B_l1487_148749

def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | -4 < x ∧ x < 1 ∨ 3 < x ∧ x < 4} := by sorry

end intersection_of_A_and_B_l1487_148749


namespace n_value_l1487_148769

theorem n_value (n : ℝ) (h1 : n > 0) (h2 : Real.sqrt (4 * n^2) = 64) : n = 32 := by
  sorry

end n_value_l1487_148769


namespace reflection_of_P_l1487_148764

/-- Given a point P in a Cartesian coordinate system, 
    return its coordinates with respect to the origin -/
def reflect_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), -(p.2))

/-- Theorem: The reflection of point P(2,1) across the origin is (-2,-1) -/
theorem reflection_of_P : reflect_point (2, 1) = (-2, -1) := by
  sorry

end reflection_of_P_l1487_148764


namespace train_distance_l1487_148780

theorem train_distance (x : ℝ) :
  (x > 0) →
  (x / 40 + 2*x / 20 = (x + 2*x) / 48) →
  (x + 2*x = 6) :=
by sorry

end train_distance_l1487_148780


namespace partial_fraction_decomposition_l1487_148759

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), A = 75 / 16 ∧ B = 21 / 16 ∧
  ∀ (x : ℚ), x ≠ 12 → x ≠ -4 →
    (6 * x + 3) / (x^2 - 8*x - 48) = A / (x - 12) + B / (x + 4) := by
  sorry

end partial_fraction_decomposition_l1487_148759


namespace right_triangle_hypotenuse_l1487_148725

/-- In a right triangle with one angle of 30° and the side opposite to this angle
    having length 6, the length of the hypotenuse is 12. -/
theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  a = 6 →  -- Length of the side opposite to 30° angle
  Real.cos (30 * π / 180) = b / c →  -- Cosine of 30° in terms of adjacent side and hypotenuse
  c = 12 := by
sorry

end right_triangle_hypotenuse_l1487_148725


namespace age_solution_l1487_148794

/-- The age equation as described in the problem -/
def age_equation (x : ℝ) : Prop :=
  3 * (x + 3) - 3 * (x - 3) = x

/-- Theorem stating that 18 is the solution to the age equation -/
theorem age_solution : ∃ x : ℝ, age_equation x ∧ x = 18 := by
  sorry

end age_solution_l1487_148794


namespace max_value_of_vector_sum_l1487_148717

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem max_value_of_vector_sum (a b c : V) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (hc : ‖c‖ = 3) :
  ∃ (max_value : ℝ), max_value = 94 ∧
    ∀ (x y z : V), ‖x‖ = 1 → ‖y‖ = 2 → ‖z‖ = 3 →
      ‖x + 2•y‖^2 + ‖y + 2•z‖^2 + ‖z + 2•x‖^2 ≤ max_value :=
by
  sorry

end max_value_of_vector_sum_l1487_148717


namespace intersection_equals_two_to_infinity_l1487_148762

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log ((1 - x) / x)}
def N : Set ℝ := {y | ∃ x, y = x^2 + 2*x + 3}

-- Define the complement of M in ℝ
def M_complement : Set ℝ := {x | x ∉ M}

-- Define the set [2, +∞)
def two_to_infinity : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem intersection_equals_two_to_infinity : (M_complement ∩ N) = two_to_infinity := by
  sorry

end intersection_equals_two_to_infinity_l1487_148762


namespace negative_324_same_terminal_side_as_36_l1487_148724

/-- Two angles have the same terminal side if their difference is a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

/-- The angle -324° has the same terminal side as 36° -/
theorem negative_324_same_terminal_side_as_36 :
  same_terminal_side 36 (-324) := by
  sorry

end negative_324_same_terminal_side_as_36_l1487_148724


namespace monotonic_decreasing_interval_of_f_l1487_148754

noncomputable def f (x : ℝ) : ℝ := -2 * x + x^3

theorem monotonic_decreasing_interval_of_f :
  ∀ x : ℝ, (x > -Real.sqrt 6 / 3 ∧ x < Real.sqrt 6 / 3) ↔ 
    StrictMonoOn f (Set.Ioo (-Real.sqrt 6 / 3) (Real.sqrt 6 / 3)) := by
  sorry

end monotonic_decreasing_interval_of_f_l1487_148754


namespace tangent_length_fq_l1487_148722

-- Define the triangle
structure RightTriangle where
  de : ℝ
  df : ℝ
  ef : ℝ
  right_angle_at_e : de^2 + ef^2 = df^2

-- Define the circle
structure TangentCircle where
  center_on_de : Bool
  tangent_to_df : Bool
  tangent_to_ef : Bool

-- Theorem statement
theorem tangent_length_fq 
  (t : RightTriangle) 
  (c : TangentCircle) 
  (h1 : t.de = 7) 
  (h2 : t.df = Real.sqrt 85) 
  (h3 : c.center_on_de = true) 
  (h4 : c.tangent_to_df = true) 
  (h5 : c.tangent_to_ef = true) : 
  ∃ q : ℝ, q = 6 ∧ q = t.ef := by
  sorry

end tangent_length_fq_l1487_148722


namespace lines_parallel_lines_perpendicular_l1487_148770

/-- Two lines in the plane --/
structure Lines where
  a : ℝ
  l1 : ℝ → ℝ → ℝ := λ x y => a * x + 2 * y + 6
  l2 : ℝ → ℝ → ℝ := λ x y => x + (a - 1) * y + a^2 - 1

/-- The lines are parallel iff a = -1 --/
theorem lines_parallel (lines : Lines) : 
  (∃ k : ℝ, ∀ x y : ℝ, lines.l1 x y = k * lines.l2 x y) ↔ lines.a = -1 :=
sorry

/-- The lines are perpendicular iff a = 2/3 --/
theorem lines_perpendicular (lines : Lines) :
  (∀ x1 y1 x2 y2 : ℝ, 
    (lines.l1 x1 y1 = 0 ∧ lines.l1 x2 y2 = 0) → 
    (lines.l2 x1 y1 = 0 ∧ lines.l2 x2 y2 = 0) → 
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) * 
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) = 
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))^2) 
  ↔ lines.a = 2/3 :=
sorry

end lines_parallel_lines_perpendicular_l1487_148770


namespace f_value_at_5_l1487_148714

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 - b * x + 2

theorem f_value_at_5 (a b : ℝ) :
  f a b (-5) = 17 → f a b 5 = -13 := by
  sorry

end f_value_at_5_l1487_148714


namespace min_cards_for_four_of_a_kind_standard_deck_l1487_148763

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : Nat)
  (num_ranks : Nat)
  (cards_per_rank : Nat)
  (num_jokers : Nat)

/-- Calculates the minimum number of cards needed to guarantee "a four of a kind" -/
def min_cards_for_four_of_a_kind (d : Deck) : Nat :=
  d.num_jokers + (d.num_ranks * (d.cards_per_rank - 1)) + 1

/-- Theorem stating the minimum number of cards needed for "a four of a kind" in a standard deck -/
theorem min_cards_for_four_of_a_kind_standard_deck :
  let standard_deck : Deck := {
    total_cards := 52,
    num_ranks := 13,
    cards_per_rank := 4,
    num_jokers := 2
  }
  min_cards_for_four_of_a_kind standard_deck = 42 := by
  sorry

end min_cards_for_four_of_a_kind_standard_deck_l1487_148763


namespace solution_set_when_a_is_one_range_of_a_l1487_148748

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x + a| - x - 2

-- Theorem for part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 0} = {x : ℝ | x < 0 ∨ x > 2} := by sorry

-- Theorem for part II
theorem range_of_a (a : ℝ) (h : a > -1) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-a) 1 ∧ f a x₀ ≤ 0) →
  a ∈ Set.Ioo (-1) 2 := by sorry

end solution_set_when_a_is_one_range_of_a_l1487_148748


namespace ln_inequality_condition_l1487_148732

theorem ln_inequality_condition (x : ℝ) :
  (∀ x, (Real.log x < 0 → x < 1)) ∧
  (∃ x, x < 1 ∧ Real.log x ≥ 0) :=
by sorry

end ln_inequality_condition_l1487_148732


namespace trigonometric_expression_equality_l1487_148740

theorem trigonometric_expression_equality :
  let sin30 := (1 : ℝ) / 2
  let cos30 := Real.sqrt 3 / 2
  let tan60 := Real.sqrt 3
  2 * sin30 + cos30 * tan60 = 5 / 2 := by
  sorry

end trigonometric_expression_equality_l1487_148740


namespace meat_voters_count_l1487_148723

/-- The number of students who voted for veggies -/
def veggies_votes : ℕ := 337

/-- The total number of students who voted -/
def total_votes : ℕ := 672

/-- The number of students who voted for meat -/
def meat_votes : ℕ := total_votes - veggies_votes

theorem meat_voters_count : meat_votes = 335 := by
  sorry

end meat_voters_count_l1487_148723


namespace average_difference_l1487_148738

theorem average_difference (x : ℝ) : 
  (10 + 30 + 50) / 3 = ((20 + 40 + x) / 3) + 8 → x = 6 := by
  sorry

end average_difference_l1487_148738


namespace g_equals_g_l1487_148760

/-- Two triangles are similar isosceles triangles with vertex A and angle α -/
def similarIsoscelesA (t1 t2 : Set (ℝ × ℝ)) (A : ℝ × ℝ) (α : ℝ) : Prop :=
  sorry

/-- Two triangles are similar isosceles triangles with angle π - α at the vertex -/
def similarIsoscelesVertex (t1 t2 : Set (ℝ × ℝ)) (α : ℝ) : Prop :=
  sorry

/-- The theorem stating that G = G' given the conditions -/
theorem g_equals_g' (A K L M N G G' : ℝ × ℝ) (α : ℝ) 
    (h1 : similarIsoscelesA {A, K, L} {A, M, N} A α)
    (h2 : similarIsoscelesVertex {G, N, K} {G', L, M} α) :
    G = G' :=
  sorry

end g_equals_g_l1487_148760


namespace flag_designs_count_l1487_148779

/-- The number of colors available for the flag. -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag. -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs. -/
def total_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the total number of possible flag designs is 27. -/
theorem flag_designs_count : total_flag_designs = 27 := by
  sorry

end flag_designs_count_l1487_148779


namespace triangle_8_6_4_l1487_148746

/-- A triangle can be formed if the sum of any two sides is greater than the third side,
    and the difference between any two sides is less than the third side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  a - b < c ∧ b - c < a ∧ c - a < b

/-- Prove that line segments of lengths 8, 6, and 4 can form a triangle. -/
theorem triangle_8_6_4 : can_form_triangle 8 6 4 := by
  sorry

end triangle_8_6_4_l1487_148746


namespace nail_pierces_one_shape_l1487_148771

/-- Represents a shape that can be placed on a rectangular surface --/
structure Shape where
  area : ℝ
  -- Other properties of the shape could be added here

/-- Represents a rectangular box --/
structure Box where
  length : ℝ
  width : ℝ
  center : ℝ × ℝ

/-- Represents the placement of a shape on the box's bottom --/
structure Placement where
  shape : Shape
  position : ℝ × ℝ

/-- Checks if two placements completely cover the box's bottom --/
def covers (b : Box) (p1 p2 : Placement) : Prop := sorry

/-- Checks if a point is inside a placed shape --/
def pointInPlacement (point : ℝ × ℝ) (p : Placement) : Prop := sorry

/-- Main theorem: It's possible to arrange two identical shapes to cover a box's bottom
    such that the center point is in only one of the shapes --/
theorem nail_pierces_one_shape (b : Box) (s : Shape) :
  ∃ (p1 p2 : Placement),
    p1.shape = s ∧ p2.shape = s ∧
    covers b p1 p2 ∧
    (pointInPlacement b.center p1 ↔ ¬pointInPlacement b.center p2) := sorry

end nail_pierces_one_shape_l1487_148771


namespace hyperbola_eccentricity_l1487_148768

/-- The eccentricity of a hyperbola with equation x²/4 - y²/2 = 1 is √6/2 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = Real.sqrt 6 / 2 ∧
  ∀ x y : ℝ, x^2 / 4 - y^2 / 2 = 1 → 
  e = Real.sqrt ((x / 2)^2 + (y / Real.sqrt 2)^2) / 2 := by
  sorry

end hyperbola_eccentricity_l1487_148768


namespace largest_square_tile_l1487_148735

theorem largest_square_tile (a b : ℕ) (ha : a = 72) (hb : b = 90) :
  ∃ (s : ℕ), s = Nat.gcd a b ∧ 
  s * (a / s) = a ∧ 
  s * (b / s) = b ∧
  ∀ (t : ℕ), t * (a / t) = a → t * (b / t) = b → t ≤ s :=
sorry

end largest_square_tile_l1487_148735


namespace B_power_99_is_identity_l1487_148790

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_99_is_identity :
  B ^ 99 = (1 : Matrix (Fin 3) (Fin 3) ℚ) := by
  sorry

end B_power_99_is_identity_l1487_148790


namespace large_triangle_perimeter_l1487_148782

/-- An isosceles triangle with two sides of length 12 and one side of length 14 -/
structure SmallTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : side1 = side2 ∧ side1 = 12 ∧ side3 = 14

/-- A triangle similar to the small triangle with longest side 42 -/
structure LargeTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  similar_to_small : ∃ (k : ℝ), side1 = k * 12 ∧ side2 = k * 12 ∧ side3 = k * 14
  longest_side : side3 = 42

/-- The perimeter of the large triangle is 114 -/
theorem large_triangle_perimeter (small : SmallTriangle) (large : LargeTriangle) :
  large.side1 + large.side2 + large.side3 = 114 := by
  sorry

end large_triangle_perimeter_l1487_148782


namespace cost_price_calculation_l1487_148767

/-- Proves that if an article is sold for Rs. 400 with a 60% profit, its cost price is Rs. 250. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 400 →
  profit_percentage = 60 →
  selling_price = (1 + profit_percentage / 100) * 250 := by
  sorry

end cost_price_calculation_l1487_148767


namespace subset_count_divisible_by_prime_l1487_148753

theorem subset_count_divisible_by_prime (p : Nat) (hp : Nat.Prime p) (hp_odd : Odd p) :
  let S := Finset.range (2 * p)
  (Finset.filter (fun A : Finset Nat =>
    A.card = p ∧ (A.sum id) % p = 0) (Finset.powerset S)).card =
  (1 / p) * (Nat.choose (2 * p) p - 2) + 2 := by
  sorry

end subset_count_divisible_by_prime_l1487_148753


namespace ellipse_eccentricity_l1487_148778

/-- Given an ellipse and hyperbola with common foci F₁ and F₂, intersecting at point P -/
structure EllipseHyperbolaIntersection where
  /-- The eccentricity of the ellipse -/
  e₁ : ℝ
  /-- The eccentricity of the hyperbola -/
  e₂ : ℝ
  /-- Angle F₁PF₂ -/
  angle_F₁PF₂ : ℝ
  /-- 0 < e₁ < 1 (eccentricity of ellipse) -/
  h₁ : 0 < e₁ ∧ e₁ < 1
  /-- e₂ > 1 (eccentricity of hyperbola) -/
  h₂ : e₂ > 1
  /-- cos ∠F₁PF₂ = 3/5 -/
  h₃ : Real.cos angle_F₁PF₂ = 3/5
  /-- e₂ = 2e₁ -/
  h₄ : e₂ = 2 * e₁

/-- The eccentricity of the ellipse is √10/5 -/
theorem ellipse_eccentricity (eh : EllipseHyperbolaIntersection) : eh.e₁ = Real.sqrt 10 / 5 := by
  sorry

end ellipse_eccentricity_l1487_148778


namespace parity_of_expression_l1487_148781

theorem parity_of_expression (o n c : ℤ) 
  (ho : Odd o) (hc : Odd c) : Even (o^2 + n*o + c) := by
  sorry

end parity_of_expression_l1487_148781


namespace difference_divisible_by_nine_l1487_148706

theorem difference_divisible_by_nine (a b : ℤ) : 
  ∃ k : ℤ, (3 * a + 2)^2 - (3 * b + 2)^2 = 9 * k := by
  sorry

end difference_divisible_by_nine_l1487_148706


namespace rectangle_area_l1487_148786

theorem rectangle_area (square_area : ℝ) (rect_length rect_width : ℝ) : 
  square_area = 36 →
  4 * square_area.sqrt = 2 * (rect_length + rect_width) →
  rect_length = 3 * rect_width →
  rect_length * rect_width = 27 := by
sorry

end rectangle_area_l1487_148786


namespace adult_tickets_count_l1487_148709

theorem adult_tickets_count
  (adult_price : ℝ)
  (child_price : ℝ)
  (total_tickets : ℕ)
  (total_cost : ℝ)
  (h1 : adult_price = 5.5)
  (h2 : child_price = 3.5)
  (h3 : total_tickets = 21)
  (h4 : total_cost = 83.5) :
  ∃ (adult_count : ℕ) (child_count : ℕ),
    adult_count + child_count = total_tickets ∧
    adult_count * adult_price + child_count * child_price = total_cost ∧
    adult_count = 5 :=
by sorry

end adult_tickets_count_l1487_148709


namespace triangle_area_l1487_148736

/-- Given a point A(a, 0) where a > 0, a line with 30° inclination tangent to circle O: x^2 + y^2 = r^2 
    at point B, and |AB| = √3, prove that the area of triangle OAB is √3/2 -/
theorem triangle_area (a r : ℝ) (ha : a > 0) (hr : r > 0) : 
  let A : ℝ × ℝ := (a, 0)
  let O : ℝ × ℝ := (0, 0)
  let line_slope : ℝ := Real.sqrt 3 / 3
  let circle (x y : ℝ) := x^2 + y^2 = r^2
  let tangent_line (x y : ℝ) := y = line_slope * (x - a)
  ∃ (B : ℝ × ℝ), 
    circle B.1 B.2 ∧ 
    tangent_line B.1 B.2 ∧ 
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = 3 →
    (1/2 : ℝ) * r * Real.sqrt 3 = Real.sqrt 3 / 2 :=
by sorry

end triangle_area_l1487_148736


namespace matrix_equality_l1487_148718

theorem matrix_equality (X Y : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : X + Y = X * Y)
  (h2 : X * Y = ![![16/3, 2], ![-10/3, 10/3]]) :
  Y * X = ![![16/3, 2], ![-10/3, 10/3]] := by sorry

end matrix_equality_l1487_148718


namespace inequality_condition_l1487_148744

theorem inequality_condition (a b : ℝ) : 
  (∀ x : ℝ, (a + 1) * x^2 + a * x + a > b * (x^2 + x + 1)) ↔ b < a :=
by sorry

end inequality_condition_l1487_148744


namespace unfactorable_quartic_l1487_148716

theorem unfactorable_quartic : ¬∃ (a b c d : ℤ), ∀ (x : ℝ), 
  x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end unfactorable_quartic_l1487_148716


namespace power_equation_solution_l1487_148742

theorem power_equation_solution (n : ℕ) : 3^n = 3 * 9^3 * 81^2 → n = 15 := by
  sorry

end power_equation_solution_l1487_148742


namespace proper_subsets_count_l1487_148720

def S : Finset ℕ := {0, 3, 4}

theorem proper_subsets_count : (Finset.powerset S).card - 1 = 7 := by
  sorry

end proper_subsets_count_l1487_148720


namespace imaginary_complex_implies_m_conditions_l1487_148766

theorem imaginary_complex_implies_m_conditions (m : ℝ) : 
  (∃ (z : ℂ), z = Complex.mk (m^2 - 3*m - 4) (m^2 - 5*m - 6) ∧ z.re = 0 ∧ z.im ≠ 0) →
  (m ≠ -1 ∧ m ≠ 6) := by
  sorry

end imaginary_complex_implies_m_conditions_l1487_148766


namespace defective_pens_l1487_148713

theorem defective_pens (total_pens : ℕ) (prob_non_defective : ℚ) : 
  total_pens = 12 →
  prob_non_defective = 7/33 →
  (∃ (defective : ℕ), 
    defective ≤ total_pens ∧ 
    (total_pens - defective : ℚ) / total_pens * ((total_pens - defective - 1) : ℚ) / (total_pens - 1) = prob_non_defective ∧
    defective = 4) := by
  sorry

end defective_pens_l1487_148713


namespace seventh_term_is_four_l1487_148719

/-- A geometric sequence with first term 1 and a specific condition on terms 3, 4, and 5 -/
def special_geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, ∃ r : ℝ, ∀ k : ℕ, a (k + 1) = a k * r) ∧  -- geometric sequence condition
  a 1 = 1 ∧                                           -- first term is 1
  a 3 * a 5 = 4 * (a 4 - 1)                           -- given condition

/-- The 7th term of the special geometric sequence is 4 -/
theorem seventh_term_is_four (a : ℕ → ℝ) (h : special_geometric_sequence a) : a 7 = 4 := by
  sorry

end seventh_term_is_four_l1487_148719


namespace journey_length_l1487_148765

theorem journey_length (first_part second_part third_part total : ℝ) 
  (h1 : first_part = (1/4) * total)
  (h2 : second_part = 30)
  (h3 : third_part = (1/3) * total)
  (h4 : total = first_part + second_part + third_part) :
  total = 72 := by
  sorry

end journey_length_l1487_148765


namespace circle_intersection_chord_l1487_148743

/-- Given two circles C₁ and C₂, where C₁ passes through the center of C₂,
    the equation of their chord of intersection is 5x + y - 19 = 0 -/
theorem circle_intersection_chord 
  (C₁ : ℝ → ℝ → Prop) 
  (C₂ : ℝ → ℝ → Prop) 
  (h₁ : ∀ x y, C₁ x y ↔ (x + 1)^2 + y^2 = r^2) 
  (h₂ : ∀ x y, C₂ x y ↔ (x - 4)^2 + (y - 1)^2 = 4) 
  (h₃ : C₁ 4 1) :
  ∀ x y, (C₁ x y ∧ C₂ x y) ↔ 5*x + y - 19 = 0 :=
sorry

end circle_intersection_chord_l1487_148743


namespace greatest_savings_l1487_148796

def plane_cost : ℚ := 600
def boat_cost : ℚ := 254
def helicopter_cost : ℚ := 850

def savings (cost1 cost2 : ℚ) : ℚ := max cost1 cost2 - min cost1 cost2

theorem greatest_savings :
  max (savings plane_cost boat_cost) (savings helicopter_cost boat_cost) = 596 :=
by sorry

end greatest_savings_l1487_148796


namespace total_marbles_count_l1487_148798

def initial_marbles : ℝ := 87.0
def received_marbles : ℝ := 8.0

theorem total_marbles_count : 
  initial_marbles + received_marbles = 95.0 := by
  sorry

end total_marbles_count_l1487_148798


namespace greatest_integer_quadratic_inequality_l1487_148793

theorem greatest_integer_quadratic_inequality :
  ∀ n : ℤ, n^2 - 13*n + 36 ≤ 0 → n ≤ 9 ∧
  ∃ m : ℤ, m^2 - 13*m + 36 ≤ 0 ∧ m = 9 :=
by
  sorry

end greatest_integer_quadratic_inequality_l1487_148793


namespace buratino_betting_strategy_l1487_148772

theorem buratino_betting_strategy :
  ∃ (x₁ x₂ x₃ y : ℕ+),
    x₁ + x₂ + x₃ + y = 20 ∧
    5 * x₁ + y ≥ 21 ∧
    4 * x₂ + y ≥ 21 ∧
    2 * x₃ + y ≥ 21 :=
by sorry

end buratino_betting_strategy_l1487_148772


namespace lowest_degree_is_four_l1487_148710

/-- A polynomial with coefficients in ℤ -/
def IntPolynomial := Polynomial ℤ

/-- The set of coefficients of a polynomial -/
def coefficientSet (p : IntPolynomial) : Set ℤ :=
  {a : ℤ | ∃ (n : ℕ), p.coeff n = a}

/-- The property that a polynomial satisfies the given conditions -/
def satisfiesCondition (p : IntPolynomial) : Prop :=
  ∃ (b : ℤ),
    (∃ (a₁ : ℤ), a₁ ∈ coefficientSet p ∧ a₁ < b) ∧
    (∃ (a₂ : ℤ), a₂ ∈ coefficientSet p ∧ a₂ > b) ∧
    b ∉ coefficientSet p

/-- The theorem stating that the lowest degree of a polynomial satisfying the condition is 4 -/
theorem lowest_degree_is_four :
  ∃ (p : IntPolynomial),
    satisfiesCondition p ∧
    p.degree = 4 ∧
    ∀ (q : IntPolynomial), satisfiesCondition q → q.degree ≥ 4 := by
  sorry

end lowest_degree_is_four_l1487_148710


namespace final_state_theorem_l1487_148783

/-- Represents the state of the cage -/
structure CageState where
  crickets : ℕ
  katydids : ℕ

/-- Represents a magician's transformation -/
inductive Transformation
  | Red
  | Green

/-- Applies a single transformation to the cage state -/
def applyTransformation (state : CageState) (t : Transformation) : CageState :=
  match t with
  | Transformation.Red => 
      { crickets := state.crickets + 1, katydids := state.katydids - 2 }
  | Transformation.Green => 
      { crickets := state.crickets - 5, katydids := state.katydids + 2 }

/-- Applies a sequence of transformations to the cage state -/
def applyTransformations (state : CageState) (ts : List Transformation) : CageState :=
  match ts with
  | [] => state
  | t::rest => applyTransformations (applyTransformation state t) rest

theorem final_state_theorem (transformations : List Transformation) :
  transformations.length = 15 →
  (applyTransformations { crickets := 21, katydids := 30 } transformations).crickets = 0 →
  (applyTransformations { crickets := 21, katydids := 30 } transformations).katydids = 24 :=
by
  sorry


end final_state_theorem_l1487_148783


namespace sqrt_mixed_number_simplification_l1487_148733

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 := by sorry

end sqrt_mixed_number_simplification_l1487_148733


namespace power_product_evaluation_l1487_148703

theorem power_product_evaluation : 
  let a : ℕ := 2
  (a^3 * a^4 : ℕ) = 128 := by
  sorry

end power_product_evaluation_l1487_148703


namespace polynomial_divisibility_l1487_148726

def f (a b x : ℝ) : ℝ := x^5 - 3*x^4 + a*x^3 + b*x^2 - 5*x - 5

theorem polynomial_divisibility (a b : ℝ) :
  (∀ x, (x^2 - 1) ∣ f a b x) ↔ (a = 4 ∧ b = 8) :=
sorry

end polynomial_divisibility_l1487_148726


namespace min_distance_to_line_l1487_148704

/-- Given a right triangle with sides a, b, and hypotenuse c, and a point (m, n) on the line ax + by + 2c = 0, 
    the minimum value of m^2 + n^2 is 4. -/
theorem min_distance_to_line (a b c m n : ℝ) : 
  a^2 + b^2 = c^2 →  -- Right triangle condition
  a * m + b * n + 2 * c = 0 →  -- Point (m, n) lies on the line
  ∃ (m₀ n₀ : ℝ), a * m₀ + b * n₀ + 2 * c = 0 ∧ 
    ∀ (m' n' : ℝ), a * m' + b * n' + 2 * c = 0 → m₀^2 + n₀^2 ≤ m'^2 + n'^2 ∧
    m₀^2 + n₀^2 = 4 :=
by sorry

end min_distance_to_line_l1487_148704


namespace fraction_of_25_smaller_than_40_percent_of_60_by_4_l1487_148788

theorem fraction_of_25_smaller_than_40_percent_of_60_by_4 : 
  (25 * (40 / 100 * 60 - 4)) / 25 = 4 / 5 := by
  sorry

end fraction_of_25_smaller_than_40_percent_of_60_by_4_l1487_148788


namespace initial_balance_was_800_liza_initial_balance_l1487_148711

/-- Represents the transactions in Liza's checking account --/
structure AccountTransactions where
  initial_balance : ℕ
  rent_payment : ℕ
  paycheck_deposit : ℕ
  electricity_bill : ℕ
  internet_bill : ℕ
  phone_bill : ℕ
  final_balance : ℕ

/-- Theorem stating that given the transactions and final balance, the initial balance was 800 --/
theorem initial_balance_was_800 (t : AccountTransactions) 
  (h1 : t.rent_payment = 450)
  (h2 : t.paycheck_deposit = 1500)
  (h3 : t.electricity_bill = 117)
  (h4 : t.internet_bill = 100)
  (h5 : t.phone_bill = 70)
  (h6 : t.final_balance = 1563)
  (h7 : t.initial_balance - t.rent_payment + t.paycheck_deposit - t.electricity_bill - t.internet_bill - t.phone_bill = t.final_balance) :
  t.initial_balance = 800 := by
  sorry

/-- Main theorem that proves Liza had $800 in her checking account on Tuesday --/
theorem liza_initial_balance : ∃ (t : AccountTransactions), t.initial_balance = 800 ∧ 
  t.rent_payment = 450 ∧
  t.paycheck_deposit = 1500 ∧
  t.electricity_bill = 117 ∧
  t.internet_bill = 100 ∧
  t.phone_bill = 70 ∧
  t.final_balance = 1563 ∧
  t.initial_balance - t.rent_payment + t.paycheck_deposit - t.electricity_bill - t.internet_bill - t.phone_bill = t.final_balance := by
  sorry

end initial_balance_was_800_liza_initial_balance_l1487_148711


namespace chairs_built_in_ten_days_l1487_148761

/-- Calculates the number of chairs a worker can build in a given number of days -/
def chairs_built (hours_per_shift : ℕ) (hours_per_chair : ℕ) (days : ℕ) : ℕ :=
  (hours_per_shift * days) / hours_per_chair

/-- Proves that a worker working 8-hour shifts, taking 5 hours per chair, can build 16 chairs in 10 days -/
theorem chairs_built_in_ten_days :
  chairs_built 8 5 10 = 16 := by
  sorry

end chairs_built_in_ten_days_l1487_148761


namespace square_sum_equals_34_l1487_148758

theorem square_sum_equals_34 (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 4.5) : a^2 + b^2 = 34 := by
  sorry

end square_sum_equals_34_l1487_148758


namespace smallest_integer_larger_than_root_sum_eighth_power_l1487_148707

theorem smallest_integer_larger_than_root_sum_eighth_power :
  ∃ n : ℤ, n = 1631 ∧ (∀ m : ℤ, m > (Real.sqrt 5 + Real.sqrt 3)^8 → m ≥ n) ∧
  (n - 1 : ℝ) ≤ (Real.sqrt 5 + Real.sqrt 3)^8 := by
  sorry

end smallest_integer_larger_than_root_sum_eighth_power_l1487_148707


namespace exists_consecutive_numbers_with_54_times_product_l1487_148745

def nonZeroDigits (n : ℕ) : List ℕ :=
  (n.digits 10).filter (· ≠ 0)

def productOfNonZeroDigits (n : ℕ) : ℕ :=
  (nonZeroDigits n).prod

theorem exists_consecutive_numbers_with_54_times_product : 
  ∃ n : ℕ, productOfNonZeroDigits (n + 1) = 54 * productOfNonZeroDigits n := by
  sorry

end exists_consecutive_numbers_with_54_times_product_l1487_148745


namespace salt_solution_mixture_l1487_148734

/-- Proves that adding 70 ounces of 60% salt solution to 70 ounces of 20% salt solution results in a 40% salt solution -/
theorem salt_solution_mixture : 
  let initial_volume : ℝ := 70
  let initial_concentration : ℝ := 0.2
  let added_volume : ℝ := 70
  let added_concentration : ℝ := 0.6
  let final_concentration : ℝ := 0.4
  (initial_volume * initial_concentration + added_volume * added_concentration) / (initial_volume + added_volume) = final_concentration :=
by sorry

end salt_solution_mixture_l1487_148734


namespace geometric_sequence_problem_l1487_148773

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 160 * r = b ∧ b * r = 1) → 
  b = 4 * Real.sqrt 10 := by
sorry

end geometric_sequence_problem_l1487_148773


namespace rectangle_area_ratio_l1487_148755

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a/c = b/d = 4/5, then the ratio of their areas is 16/25. -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 4 / 5) (h2 : b / d = 4 / 5) :
  (a * b) / (c * d) = 16 / 25 := by
  sorry

end rectangle_area_ratio_l1487_148755


namespace ratio_cubes_equals_729_l1487_148701

theorem ratio_cubes_equals_729 : (81000 ^ 3) / (9000 ^ 3) = 729 := by
  sorry

end ratio_cubes_equals_729_l1487_148701


namespace circular_seating_arrangement_l1487_148785

theorem circular_seating_arrangement (n : ℕ) (π : Fin (2*n) → Fin (2*n)) 
  (hπ : Function.Bijective π) : 
  ∃ (i j : Fin (2*n)), i ≠ j ∧ (π i - π j) % (2*n) = (i - j) % (2*n) := by
  sorry

end circular_seating_arrangement_l1487_148785


namespace first_player_can_draw_l1487_148775

/-- Represents a chess position -/
def ChessPosition : Type := Unit

/-- Represents a chess move -/
def ChessMove : Type := Unit

/-- Represents a strategy in double chess -/
def DoubleChessStrategy : Type := ChessPosition → ChessMove × ChessMove

/-- The initial chess position -/
def initialPosition : ChessPosition := sorry

/-- Applies a move to a position, returning the new position -/
def applyMove (pos : ChessPosition) (move : ChessMove) : ChessPosition := sorry

/-- Applies two consecutive moves to a position, returning the new position -/
def applyDoubleMoves (pos : ChessPosition) (moves : ChessMove × ChessMove) : ChessPosition := sorry

/-- Determines if a position is a win for the current player -/
def isWinningPosition (pos : ChessPosition) : Prop := sorry

/-- A knight move that doesn't change the position -/
def neutralKnightMove : ChessMove := sorry

/-- Theorem: The first player in double chess can always force at least a draw -/
theorem first_player_can_draw :
  ∀ (secondPlayerStrategy : DoubleChessStrategy),
  ∃ (firstPlayerStrategy : DoubleChessStrategy),
  ¬(isWinningPosition (applyDoubleMoves (applyDoubleMoves initialPosition (neutralKnightMove, neutralKnightMove)) (secondPlayerStrategy (applyDoubleMoves initialPosition (neutralKnightMove, neutralKnightMove))))) :=
sorry

end first_player_can_draw_l1487_148775


namespace ball_arrangements_count_l1487_148747

/-- The number of ways to arrange guests in circles with alternating hat colors -/
def ball_arrangements (N : ℕ) : ℕ := (2 * N).factorial

/-- Theorem stating that the number of valid arrangements is (2N)! -/
theorem ball_arrangements_count (N : ℕ) :
  ball_arrangements N = (2 * N).factorial :=
by sorry

end ball_arrangements_count_l1487_148747


namespace parallel_vectors_x_equals_one_l1487_148727

/-- Given two parallel vectors a and b, prove that x = 1 -/
theorem parallel_vectors_x_equals_one (x : ℝ) :
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (2, 4)
  (∃ (k : ℝ), a = k • b) →
  x = 1 := by
  sorry

end parallel_vectors_x_equals_one_l1487_148727


namespace herb_leaf_difference_l1487_148792

theorem herb_leaf_difference : 
  ∀ (basil sage verbena : ℕ),
  basil = 2 * sage →
  basil + sage + verbena = 29 →
  basil = 12 →
  verbena - sage = 5 := by
sorry

end herb_leaf_difference_l1487_148792


namespace octagon_semicircles_area_l1487_148731

/-- The area of the region inside a regular octagon with side length 3 and eight inscribed semicircles --/
theorem octagon_semicircles_area : 
  let s : Real := 3  -- side length of the octagon
  let r : Real := s / 2  -- radius of each semicircle
  let octagon_area : Real := 2 * (1 + Real.sqrt 2) * s^2
  let semicircle_area : Real := π * r^2 / 2
  let total_semicircle_area : Real := 8 * semicircle_area
  octagon_area - total_semicircle_area = 18 * (1 + Real.sqrt 2) - 9 * π := by
sorry

end octagon_semicircles_area_l1487_148731


namespace intersection_of_sets_l1487_148787

theorem intersection_of_sets : 
  let A : Set ℕ := {x | ∃ n, x = 2 * n}
  let B : Set ℕ := {x | ∃ n, x = 3 * n}
  let C : Set ℕ := {x | ∃ n, x = n * n}
  A ∩ B ∩ C = {x | ∃ n, x = 36 * n * n} := by
  sorry

end intersection_of_sets_l1487_148787


namespace cubic_equation_root_l1487_148777

theorem cubic_equation_root (a b : ℚ) :
  (2 + Real.sqrt 3 : ℝ) ^ 3 + a * (2 + Real.sqrt 3 : ℝ) ^ 2 + b * (2 + Real.sqrt 3 : ℝ) - 12 = 0 →
  b = -47 := by
sorry

end cubic_equation_root_l1487_148777


namespace trapezium_area_l1487_148795

/-- The area of a trapezium with given dimensions -/
theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) :
  (a + b) * h / 2 = 285 := by
  sorry

end trapezium_area_l1487_148795


namespace geometric_arithmetic_sum_l1487_148702

/-- A geometric series with the given property -/
def geometric_series (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- An arithmetic series -/
def arithmetic_series (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_arithmetic_sum (a b : ℕ → ℝ) :
  geometric_series a →
  arithmetic_series b →
  a 3 * a 11 = 4 * a 7 →
  b 7 = a 7 →
  b 5 + b 9 = 8 := by
  sorry

end geometric_arithmetic_sum_l1487_148702
