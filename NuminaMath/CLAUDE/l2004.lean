import Mathlib

namespace NUMINAMATH_CALUDE_airplane_cost_is_428_l2004_200477

/-- The cost of an airplane, given the initial amount and change received. -/
def airplane_cost (initial_amount change : ℚ) : ℚ :=
  initial_amount - change

/-- Theorem stating that the cost of the airplane is $4.28 -/
theorem airplane_cost_is_428 :
  airplane_cost 5 0.72 = 4.28 := by
  sorry

end NUMINAMATH_CALUDE_airplane_cost_is_428_l2004_200477


namespace NUMINAMATH_CALUDE_prob_heart_then_king_is_one_fiftytwo_l2004_200492

/-- A standard deck of cards. -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- The suit of a card. -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- The rank of a card. -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A playing card. -/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- The probability of drawing a heart first and a king second from a standard 52-card deck. -/
def prob_heart_then_king (d : Deck) : ℚ :=
  1 / 52

/-- Theorem stating that the probability of drawing a heart first and a king second
    from a standard 52-card deck is 1/52. -/
theorem prob_heart_then_king_is_one_fiftytwo (d : Deck) :
  prob_heart_then_king d = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_prob_heart_then_king_is_one_fiftytwo_l2004_200492


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2004_200454

/-- Prove that the equation (ax^2 - 2xy + y^2) - (-x^2 + bxy + 2y^2) = 5x^2 - 9xy + cy^2 
    holds true if and only if a = 4, b = 7, and c = -1 -/
theorem quadratic_equation_solution (a b c : ℝ) (x y : ℝ) :
  (a * x^2 - 2 * x * y + y^2) - (-x^2 + b * x * y + 2 * y^2) = 5 * x^2 - 9 * x * y + c * y^2 ↔ 
  a = 4 ∧ b = 7 ∧ c = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2004_200454


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2004_200464

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, |x - 2| - |x - 5| - k > 0) → k < -3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2004_200464


namespace NUMINAMATH_CALUDE_half_difference_donations_l2004_200417

theorem half_difference_donations (julie_donation margo_donation : ℕ) 
  (h1 : julie_donation = 4700)
  (h2 : margo_donation = 4300) :
  (julie_donation - margo_donation) / 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_half_difference_donations_l2004_200417


namespace NUMINAMATH_CALUDE_difference_expression_correct_l2004_200457

/-- The expression that represents "the difference between the opposite of a and 5 times b" -/
def difference_expression (a b : ℝ) : ℝ := -a - 5*b

/-- The difference_expression correctly represents "the difference between the opposite of a and 5 times b" -/
theorem difference_expression_correct (a b : ℝ) :
  difference_expression a b = (-a) - (5*b) := by sorry

end NUMINAMATH_CALUDE_difference_expression_correct_l2004_200457


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_one_l2004_200419

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 / 3 * x^3 + 2

theorem tangent_slope_angle_at_one :
  let f' : ℝ → ℝ := λ x ↦ deriv f x
  let slope : ℝ := f' 1
  let slope_angle : ℝ := Real.pi - Real.arctan slope
  slope_angle = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_one_l2004_200419


namespace NUMINAMATH_CALUDE_type_b_sample_count_l2004_200478

/-- Represents the number of items of type B in a stratified sample -/
def stratifiedSampleCount (totalPopulation : ℕ) (typeBPopulation : ℕ) (sampleSize : ℕ) : ℕ :=
  (typeBPopulation * sampleSize) / totalPopulation

/-- Theorem stating that the number of type B items in the sample is 15 -/
theorem type_b_sample_count :
  stratifiedSampleCount 5000 1250 60 = 15 := by
  sorry

end NUMINAMATH_CALUDE_type_b_sample_count_l2004_200478


namespace NUMINAMATH_CALUDE_square_product_equals_sum_implies_zero_l2004_200456

theorem square_product_equals_sum_implies_zero (x y : ℤ) 
  (h : x^2 * y^2 = x^2 + y^2) : x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_product_equals_sum_implies_zero_l2004_200456


namespace NUMINAMATH_CALUDE_extremum_property_l2004_200486

/-- Given a function f(x) = 1 - x*sin(x) that attains an extremum at x₀,
    prove that (1 + x₀²)(1 + cos(2x₀)) = 2 -/
theorem extremum_property (x₀ : ℝ) :
  let f : ℝ → ℝ := fun x ↦ 1 - x * Real.sin x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x₀ ≥ f x ∨ f x₀ ≤ f x) →
  (1 + x₀^2) * (1 + Real.cos (2 * x₀)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_extremum_property_l2004_200486


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2004_200463

theorem sum_of_a_and_b (a b : ℝ) : 
  a^2*b^2 + a^2 + b^2 + 1 - 2*a*b = 2*a*b → a + b = 2 ∨ a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2004_200463


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2004_200432

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (-6, 2)
  let b : ℝ × ℝ := (m, -3)
  parallel a b → m = 9 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2004_200432


namespace NUMINAMATH_CALUDE_number_difference_l2004_200466

theorem number_difference (x y : ℕ) : 
  x + y = 50 → 
  y = 31 → 
  x < 2 * y → 
  2 * y - x = 43 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2004_200466


namespace NUMINAMATH_CALUDE_polynomial_value_l2004_200483

theorem polynomial_value (m n : ℤ) (h : m - 2*n = 7) : 
  2023 - 2*m + 4*n = 2009 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_l2004_200483


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2004_200403

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 5*x₁ + 3 = 0 → 
  x₂^2 - 5*x₂ + 3 = 0 → 
  x₁^2 + x₂^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2004_200403


namespace NUMINAMATH_CALUDE_max_cables_equals_max_edges_l2004_200436

/-- Represents a bipartite graph with two sets of nodes -/
structure BipartiteGraph where
  setA : Nat
  setB : Nat

/-- Calculates the maximum number of edges in a bipartite graph -/
def maxEdges (g : BipartiteGraph) : Nat :=
  g.setA * g.setB

/-- Represents the company's computer network -/
def companyNetwork : BipartiteGraph :=
  { setA := 16, setB := 12 }

/-- The maximum number of cables needed is equal to the maximum number of edges in the bipartite graph -/
theorem max_cables_equals_max_edges :
  maxEdges companyNetwork = 192 := by
  sorry

#eval maxEdges companyNetwork

end NUMINAMATH_CALUDE_max_cables_equals_max_edges_l2004_200436


namespace NUMINAMATH_CALUDE_robot_number_difference_l2004_200462

def largest_three_digit (a b c : Nat) : Nat :=
  100 * max a (max b c) + 10 * max (min (max a b) (max b c)) (min a (min b c)) + min a (min b c)

def smallest_three_digit (a b c : Nat) : Nat :=
  if min a (min b c) = 0
  then 100 * min (max a b) (max b c) + 10 * max (min a (min b c)) (min (max a b) (max b c)) + 0
  else 100 * min a (min b c) + 10 * min (max a b) (max b c) + max a (max b c)

theorem robot_number_difference :
  largest_three_digit 2 3 5 - smallest_three_digit 4 0 6 = 126 := by
  sorry

end NUMINAMATH_CALUDE_robot_number_difference_l2004_200462


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2004_200425

theorem intersection_of_sets :
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {1, 3}
  A ∩ B = {1, 3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2004_200425


namespace NUMINAMATH_CALUDE_tournament_games_count_l2004_200422

/-- Calculates the number of games in a single-elimination tournament. -/
def gamesInSingleElimination (n : ℕ) : ℕ := n - 1

/-- Represents the tournament structure and calculates the total number of games. -/
def totalGames (interestedTeams preliminaryTeams mainTournamentTeams : ℕ) : ℕ :=
  let preliminaryGames := gamesInSingleElimination preliminaryTeams
  let mainTournamentGames := gamesInSingleElimination mainTournamentTeams
  preliminaryGames + mainTournamentGames

/-- Theorem stating that the total number of games in the described tournament is 23. -/
theorem tournament_games_count :
  totalGames 25 9 16 = 23 := by
  sorry

#eval totalGames 25 9 16

end NUMINAMATH_CALUDE_tournament_games_count_l2004_200422


namespace NUMINAMATH_CALUDE_max_value_fraction_l2004_200482

theorem max_value_fraction (x y : ℝ) : 
  (2*x + 3*y + 4) / Real.sqrt (2*x^2 + 3*y^2 + 5) ≤ Real.sqrt 28 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2004_200482


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l2004_200409

theorem quadratic_real_roots_range (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 - 3 * x = k) ↔ k ≥ -9/8 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l2004_200409


namespace NUMINAMATH_CALUDE_water_for_bread_dough_l2004_200467

/-- The amount of water (in mL) needed for a given amount of flour (in mL),
    given a water-to-flour ratio. -/
def water_needed (water_ratio : ℚ) (flour_amount : ℚ) : ℚ :=
  (water_ratio * flour_amount)

/-- Theorem stating that for 1000 mL of flour, given the ratio of 80 mL water
    to 200 mL flour, the amount of water needed is 400 mL. -/
theorem water_for_bread_dough : water_needed (80 / 200) 1000 = 400 := by
  sorry

end NUMINAMATH_CALUDE_water_for_bread_dough_l2004_200467


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_t_l2004_200442

-- Define the function f
def f (x : ℝ) : ℝ := |x| - 2*|x + 3|

-- Theorem for the first part of the problem
theorem solution_set_f (x : ℝ) :
  f x ≥ 2 ↔ -4 ≤ x ∧ x ≤ -8/3 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_t :
  ∃ t₁ t₂ : ℝ, t₁ = -1/3 ∧ t₂ = 5/3 ∧
  (∀ t : ℝ, (∃ x : ℝ, f x - |3*t - 2| ≥ 0) ↔ t₁ ≤ t ∧ t ≤ t₂) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_t_l2004_200442


namespace NUMINAMATH_CALUDE_min_value_theorem_l2004_200489

theorem min_value_theorem (x : ℝ) (h : x > -3) :
  x + 2 / (x + 3) ≥ 2 * Real.sqrt 2 - 3 ∧
  (x + 2 / (x + 3) = 2 * Real.sqrt 2 - 3 ↔ x = -3 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2004_200489


namespace NUMINAMATH_CALUDE_downstream_speed_calculation_l2004_200402

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed given the rowing speeds in still water and upstream -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.upstream

/-- Theorem stating that for a man with given upstream and still water speeds, 
    the downstream speed is 65 kmph -/
theorem downstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 60) 
  (h2 : s.upstream = 55) : 
  downstreamSpeed s = 65 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_calculation_l2004_200402


namespace NUMINAMATH_CALUDE_inequality_of_cube_roots_l2004_200476

theorem inequality_of_cube_roots (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.rpow ((a / (b + c))^2) (1/3) + Real.rpow ((b / (c + a))^2) (1/3) + Real.rpow ((c / (a + b))^2) (1/3) ≥ 3 / Real.rpow 4 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_cube_roots_l2004_200476


namespace NUMINAMATH_CALUDE_max_value_constraint_l2004_200431

theorem max_value_constraint (x y : ℝ) : 
  x^2 + y^2 = 18*x + 8*y + 10 → (∀ a b : ℝ, a^2 + b^2 = 18*a + 8*b + 10 → 4*x + 3*y ≥ 4*a + 3*b) → 4*x + 3*y = 45 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2004_200431


namespace NUMINAMATH_CALUDE_g_one_equals_three_l2004_200416

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x

-- Define the given equations
axiom eq1 : f (-1) + g 1 = 2
axiom eq2 : f 1 + g (-1) = 4

-- State the theorem to be proved
theorem g_one_equals_three : g 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_one_equals_three_l2004_200416


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2004_200400

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 2| ≤ 2}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y = x^2 - 2*x + 2}

-- Theorem statement
theorem complement_A_intersect_B :
  (Set.univ : Set ℝ) \ (A ∩ B) = {x : ℝ | x < 1 ∨ x > 4} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2004_200400


namespace NUMINAMATH_CALUDE_bun_sets_problem_l2004_200420

theorem bun_sets_problem (N : ℕ) : 
  (∃ x y u v : ℕ, 
    3 * x + 5 * y = 25 ∧ 
    3 * u + 5 * v = 35 ∧ 
    x + y = N ∧ 
    u + v = N) → 
  N = 7 := by
sorry

end NUMINAMATH_CALUDE_bun_sets_problem_l2004_200420


namespace NUMINAMATH_CALUDE_parallel_plane_through_point_l2004_200407

def plane_equation (x y z : ℝ) := 3*x - 2*y + 4*z - 32

theorem parallel_plane_through_point :
  let given_plane (x y z : ℝ) := 3*x - 2*y + 4*z - 6
  (∀ (x y z : ℝ), plane_equation x y z = 0 ↔ ∃ (t : ℝ), given_plane x y z = t) ∧ 
  plane_equation 2 (-3) 5 = 0 ∧
  (∃ (A B C D : ℤ), ∀ (x y z : ℝ), plane_equation x y z = A*x + B*y + C*z + D) ∧
  (∃ (A : ℤ), A > 0 ∧ ∀ (x y z : ℝ), plane_equation x y z = A*x + plane_equation 0 1 0*y + plane_equation 0 0 1*z + plane_equation 0 0 0) ∧
  (Nat.gcd (Int.natAbs 3) (Int.natAbs (-2)) = 1 ∧ 
   Nat.gcd (Int.natAbs 3) (Int.natAbs 4) = 1 ∧ 
   Nat.gcd (Int.natAbs 3) (Int.natAbs (-32)) = 1) :=
by sorry

end NUMINAMATH_CALUDE_parallel_plane_through_point_l2004_200407


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2004_200458

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 2) :
  (1/a + 1/b) ≥ (3 + 2*Real.sqrt 2) / 2 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 2 ∧ 1/a₀ + 1/b₀ = (3 + 2*Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2004_200458


namespace NUMINAMATH_CALUDE_polynomial_four_positive_roots_l2004_200470

/-- A polynomial with four positive real roots -/
def P (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 - 8*a * x^3 + b * x^2 - 32*c * x + 16*c

/-- The theorem stating the conditions for the polynomial to have four positive real roots -/
theorem polynomial_four_positive_roots :
  ∀ (a : ℝ), a ≠ 0 →
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
    ∀ (x : ℝ), P a (16*a) a x = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) ∧
  (∀ (b c : ℝ), 
    (∃ (x₁ x₂ x₃ x₄ : ℝ), 
      x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
      ∀ (x : ℝ), P a b c x = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) →
    b = 16*a ∧ c = a) := by
  sorry


end NUMINAMATH_CALUDE_polynomial_four_positive_roots_l2004_200470


namespace NUMINAMATH_CALUDE_quadratic_equation_linear_coefficient_l2004_200414

theorem quadratic_equation_linear_coefficient :
  ∀ a b c : ℝ, 
    (∀ x, 2 * x^2 = 3 * x - 1 ↔ a * x^2 + b * x + c = 0) →
    a = 2 →
    b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_linear_coefficient_l2004_200414


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2004_200418

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 90 → b = 120 → c^2 = a^2 + b^2 → c = 150 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2004_200418


namespace NUMINAMATH_CALUDE_pyramid_angle_closest_to_40_l2004_200401

theorem pyramid_angle_closest_to_40 (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 2017) (h_lateral : lateral_edge = 2000) : 
  let angle := Real.arctan ((base_edge / Real.sqrt 2) / lateral_edge)
  let options := [30, 40, 50, 60]
  (40 : ℝ) ∈ options ∧ 
  ∀ x ∈ options, |angle - 40| ≤ |angle - x| :=
by sorry

end NUMINAMATH_CALUDE_pyramid_angle_closest_to_40_l2004_200401


namespace NUMINAMATH_CALUDE_sphere_volume_equal_surface_area_implies_radius_three_l2004_200445

theorem sphere_volume_equal_surface_area_implies_radius_three 
  (r : ℝ) (h : r > 0) : 
  (4 / 3 * Real.pi * r^3 = 4 * Real.pi * r^2) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equal_surface_area_implies_radius_three_l2004_200445


namespace NUMINAMATH_CALUDE_gcd_problem_l2004_200448

def gcd_notation (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_problem (b : ℕ) : 
  (gcd_notation (gcd_notation 20 16) (18 * b) = 2) → b = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2004_200448


namespace NUMINAMATH_CALUDE_committee_selection_count_l2004_200408

/-- Represents the total number of members in the class committee -/
def totalMembers : Nat := 5

/-- Represents the number of roles to be filled -/
def rolesToFill : Nat := 3

/-- Represents the number of members who cannot serve in a specific role -/
def restrictedMembers : Nat := 2

/-- Calculates the number of ways to select committee members under given constraints -/
def selectCommitteeMembers (total : Nat) (roles : Nat) (restricted : Nat) : Nat :=
  (total - restricted) * (total - 1) * (total - 2)

theorem committee_selection_count :
  selectCommitteeMembers totalMembers rolesToFill restrictedMembers = 36 := by
  sorry

#eval selectCommitteeMembers totalMembers rolesToFill restrictedMembers

end NUMINAMATH_CALUDE_committee_selection_count_l2004_200408


namespace NUMINAMATH_CALUDE_gcd_of_162_180_450_l2004_200437

theorem gcd_of_162_180_450 : Nat.gcd 162 (Nat.gcd 180 450) = 18 := by sorry

end NUMINAMATH_CALUDE_gcd_of_162_180_450_l2004_200437


namespace NUMINAMATH_CALUDE_remaining_juice_bottles_l2004_200444

/-- Given the initial number of juice bottles in the refrigerator and pantry,
    the number of bottles bought, and the number of bottles consumed,
    calculate the remaining number of bottles. -/
theorem remaining_juice_bottles
  (refrigerator_bottles : ℕ)
  (pantry_bottles : ℕ)
  (bought_bottles : ℕ)
  (consumed_bottles : ℕ)
  (h1 : refrigerator_bottles = 4)
  (h2 : pantry_bottles = 4)
  (h3 : bought_bottles = 5)
  (h4 : consumed_bottles = 3) :
  refrigerator_bottles + pantry_bottles + bought_bottles - consumed_bottles = 10 := by
  sorry

#check remaining_juice_bottles

end NUMINAMATH_CALUDE_remaining_juice_bottles_l2004_200444


namespace NUMINAMATH_CALUDE_paperback_copies_sold_l2004_200490

theorem paperback_copies_sold (hardback_copies : ℕ) (total_copies : ℕ) : 
  hardback_copies = 36000 →
  total_copies = 440000 →
  ∃ paperback_copies : ℕ, 
    paperback_copies = 9 * hardback_copies ∧
    hardback_copies + paperback_copies = total_copies ∧
    paperback_copies = 360000 := by
  sorry

end NUMINAMATH_CALUDE_paperback_copies_sold_l2004_200490


namespace NUMINAMATH_CALUDE_inverse_of_A_is_B_l2004_200451

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 7; 2, 3]

def B : Matrix (Fin 2) (Fin 2) ℚ := !![-3/2, 7/2; 1, -2]

theorem inverse_of_A_is_B :
  (Matrix.det A ≠ 0) → (A * B = 1 ∧ B * A = 1) :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_A_is_B_l2004_200451


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l2004_200455

theorem two_digit_number_problem : 
  ∃! n : ℕ, 
    10 ≤ n ∧ n < 100 ∧  -- two-digit number
    (n / 10 + n % 10 = 11) ∧  -- sum of digits is 11
    (10 * (n % 10) + (n / 10) = n + 63) ∧  -- swapped number is 63 greater
    n = 29  -- the number is 29
  := by sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l2004_200455


namespace NUMINAMATH_CALUDE_ellipse_points_equiv_target_set_l2004_200423

/-- Represents an ellipse passing through (2,1) with a > b > 0 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_passes_through : 4 / a^2 + 1 / b^2 = 1

/-- The set of points (x, y) on the ellipse satisfying |y| > 1 -/
def ellipse_points (e : Ellipse) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1 ∧ |p.2| > 1}

/-- The set of points (x, y) satisfying x^2 + y^2 < 5 and |y| > 1 -/
def target_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 < 5 ∧ |p.2| > 1}

/-- Theorem stating the equivalence of the two sets -/
theorem ellipse_points_equiv_target_set (e : Ellipse) :
  ellipse_points e = target_set := by sorry

end NUMINAMATH_CALUDE_ellipse_points_equiv_target_set_l2004_200423


namespace NUMINAMATH_CALUDE_shirt_cost_l2004_200496

theorem shirt_cost (j s : ℝ) 
  (eq1 : 3 * j + 2 * s = 69) 
  (eq2 : 2 * j + 3 * s = 81) : 
  s = 21 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l2004_200496


namespace NUMINAMATH_CALUDE_complex_equation_roots_l2004_200429

theorem complex_equation_roots : 
  let z₁ : ℂ := 1 + Real.sqrt 6 - (Real.sqrt 6 / 2) * Complex.I
  let z₂ : ℂ := 1 - Real.sqrt 6 + (Real.sqrt 6 / 2) * Complex.I
  (z₁^2 - 2*z₁ = 4 - 3*Complex.I) ∧ (z₂^2 - 2*z₂ = 4 - 3*Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l2004_200429


namespace NUMINAMATH_CALUDE_greatest_c_for_non_range_greatest_integer_c_l2004_200468

theorem greatest_c_for_non_range (c : ℤ) : 
  (∀ x : ℝ, x^2 + c*x + 20 ≠ 5) ↔ c^2 < 60 :=
sorry

theorem greatest_integer_c : 
  ∃ c : ℤ, c = 7 ∧ (∀ x : ℝ, x^2 + c*x + 20 ≠ 5) ∧ 
  (∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 20 = 5) :=
sorry

end NUMINAMATH_CALUDE_greatest_c_for_non_range_greatest_integer_c_l2004_200468


namespace NUMINAMATH_CALUDE_smallest_number_l2004_200498

theorem smallest_number (a b c d : ℤ) (ha : a = -2) (hb : b = -1) (hc : c = 1) (hd : d = 0) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2004_200498


namespace NUMINAMATH_CALUDE_investment_of_a_l2004_200441

/-- Represents a partnership investment. -/
structure Partnership where
  investmentA : ℝ
  investmentB : ℝ
  investmentC : ℝ
  profitB : ℝ
  profitA : ℝ
  months : ℕ

/-- Theorem stating the investment of partner A given the conditions. -/
theorem investment_of_a (p : Partnership) 
  (hB : p.investmentB = 21000)
  (hprofitB : p.profitB = 1540)
  (hprofitA : p.profitA = 1100)
  (hmonths : p.months = 8)
  (h_profit_prop : p.profitA / p.profitB = p.investmentA / p.investmentB) :
  p.investmentA = 15000 :=
sorry

end NUMINAMATH_CALUDE_investment_of_a_l2004_200441


namespace NUMINAMATH_CALUDE_maria_water_bottles_l2004_200495

/-- The number of bottles Maria initially had -/
def initial_bottles : ℕ := 14

/-- The number of bottles Maria drank -/
def bottles_drunk : ℕ := 8

/-- The number of bottles Maria bought -/
def bottles_bought : ℕ := 45

/-- The final number of bottles Maria has -/
def final_bottles : ℕ := 51

theorem maria_water_bottles : 
  initial_bottles - bottles_drunk + bottles_bought = final_bottles :=
by sorry

end NUMINAMATH_CALUDE_maria_water_bottles_l2004_200495


namespace NUMINAMATH_CALUDE_investment_dividend_income_l2004_200421

/-- Calculates the annual dividend income based on investment parameters -/
def annual_dividend_income (investment : ℚ) (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) : ℚ :=
  let num_shares := investment / quoted_price
  let dividend_per_share := (dividend_rate / 100) * face_value
  num_shares * dividend_per_share

/-- Theorem stating that the annual dividend income for the given parameters is 728 -/
theorem investment_dividend_income :
  annual_dividend_income 4940 10 9.50 14 = 728 := by
  sorry

end NUMINAMATH_CALUDE_investment_dividend_income_l2004_200421


namespace NUMINAMATH_CALUDE_train_time_theorem_l2004_200461

/-- The time in minutes for a train to travel between two platforms --/
def train_travel_time (X : ℝ) : Prop :=
  0 < X ∧ X < 60 ∧
  ∀ (start_hour start_minute end_hour end_minute : ℝ),
    -- Angle between hour and minute hands at start
    |30 * start_hour - 5.5 * start_minute| = X →
    -- Angle between hour and minute hands at end
    |30 * end_hour - 5.5 * end_minute| = X →
    -- Time difference between start and end
    (end_hour - start_hour) * 60 + (end_minute - start_minute) = X →
    X = 48

theorem train_time_theorem :
  ∀ X, train_travel_time X → X = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_train_time_theorem_l2004_200461


namespace NUMINAMATH_CALUDE_nilpotent_in_finite_ring_l2004_200443

/-- A ring with exactly n elements -/
class FiniteRing (A : Type) extends Ring A where
  card : ℕ
  finite : Fintype A
  card_eq : Fintype.card A = card

theorem nilpotent_in_finite_ring {n m : ℕ} {A : Type} [FiniteRing A] (hn : n ≥ 2) (hm : m ≥ 0) 
  (h_card : (FiniteRing.card A) = n) (a : A) 
  (h_inv : ∀ k ∈ Finset.range (n - 1), k ≥ m + 1 → IsUnit (1 - a ^ (k + 1))) :
  ∃ k : ℕ, a ^ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_nilpotent_in_finite_ring_l2004_200443


namespace NUMINAMATH_CALUDE_cut_depth_proof_l2004_200471

theorem cut_depth_proof (sheet_width sheet_height : ℕ) 
  (cut_width_1 cut_width_2 cut_width_3 : ℕ → ℕ) 
  (remaining_area : ℕ) : 
  sheet_width = 80 → 
  sheet_height = 15 → 
  (∀ d : ℕ, cut_width_1 d = 5 * d) →
  (∀ d : ℕ, cut_width_2 d = 15 * d) →
  (∀ d : ℕ, cut_width_3 d = 10 * d) →
  remaining_area = 990 →
  ∃ d : ℕ, d = 7 ∧ 
    sheet_width * sheet_height - (cut_width_1 d + cut_width_2 d + cut_width_3 d) = remaining_area :=
by sorry

end NUMINAMATH_CALUDE_cut_depth_proof_l2004_200471


namespace NUMINAMATH_CALUDE_tan_domain_shift_l2004_200406

theorem tan_domain_shift (x : ℝ) :
  (∃ k : ℤ, x = k * π / 2 + π / 12) ↔ (∃ k : ℤ, 2 * x + π / 3 = k * π + π / 2) :=
by sorry

end NUMINAMATH_CALUDE_tan_domain_shift_l2004_200406


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l2004_200450

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l2004_200450


namespace NUMINAMATH_CALUDE_smallest_a_for_minimum_l2004_200469

noncomputable def f (a x : ℝ) : ℝ := -Real.log x / x + Real.exp (a * x - 1)

theorem smallest_a_for_minimum (a : ℝ) : 
  (∀ x > 0, f a x ≥ a) ∧ (∃ x > 0, f a x = a) ↔ a = -Real.exp (-2) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_minimum_l2004_200469


namespace NUMINAMATH_CALUDE_equation_solution_l2004_200439

theorem equation_solution (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 4/3) :
  ∃! x : ℝ, (Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x) ∧
            (x = (4 - p) / Real.sqrt (8 * (2 - p))) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2004_200439


namespace NUMINAMATH_CALUDE_canal_digging_time_l2004_200446

/-- Represents the time taken to dig a canal given the number of men, hours per day, and days worked. -/
def diggingTime (men : ℕ) (hoursPerDay : ℕ) (days : ℚ) : ℚ := men * hoursPerDay * days

/-- Theorem stating that 30 men working 8 hours a day will take 1.5 days to dig a canal
    that originally took 20 men working 6 hours a day for 3 days, assuming constant work rate. -/
theorem canal_digging_time :
  diggingTime 20 6 3 = diggingTime 30 8 (3/2 : ℚ) := by
  sorry

#check canal_digging_time

end NUMINAMATH_CALUDE_canal_digging_time_l2004_200446


namespace NUMINAMATH_CALUDE_f_minimum_value_l2004_200415

/-- The function f(x, y) as defined in the problem -/
def f (x y : ℝ) : ℝ := 6 * (x^2 + y^2) * (x + y) - 4 * (x^2 + x*y + y^2) - 3 * (x + y) + 5

/-- The theorem stating the minimum value of f(x, y) -/
theorem f_minimum_value :
  (∀ x y : ℝ, x > 0 → y > 0 → f x y ≥ 2) ∧ f (1/2) (1/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_value_l2004_200415


namespace NUMINAMATH_CALUDE_man_rowing_speed_l2004_200480

/-- Proves that given a man's speed in still water and downstream speed, his upstream speed can be calculated. -/
theorem man_rowing_speed (v_still : ℝ) (v_downstream : ℝ) (h1 : v_still = 50) (h2 : v_downstream = 80) :
  v_still - (v_downstream - v_still) = 20 := by
  sorry

#check man_rowing_speed

end NUMINAMATH_CALUDE_man_rowing_speed_l2004_200480


namespace NUMINAMATH_CALUDE_min_balls_to_guarantee_20_l2004_200413

/-- Represents the number of balls of each color in the box -/
structure BoxContents where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to guarantee at least n of a single color -/
def minBallsToGuarantee (box : BoxContents) (n : Nat) : Nat :=
  sorry

/-- The specific box contents from the problem -/
def problemBox : BoxContents :=
  { red := 36, green := 24, yellow := 18, blue := 15, white := 12, black := 10 }

/-- The theorem stating the minimum number of balls to draw -/
theorem min_balls_to_guarantee_20 :
  minBallsToGuarantee problemBox 20 = 94 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_to_guarantee_20_l2004_200413


namespace NUMINAMATH_CALUDE_cans_in_sixth_bin_l2004_200435

theorem cans_in_sixth_bin (n : ℕ) (cans : ℕ → ℕ) : 
  (∀ k, cans k = k * (k + 1) / 2) → cans 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_cans_in_sixth_bin_l2004_200435


namespace NUMINAMATH_CALUDE_min_squares_6x7_l2004_200497

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- A tiling of a rectangle with squares -/
def Tiling (r : Rectangle) := List Square

/-- The area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ :=
  r.width * r.height

/-- The area of a square -/
def squareArea (s : Square) : ℕ :=
  s.side * s.side

/-- Check if a tiling is valid for a given rectangle -/
def isValidTiling (r : Rectangle) (t : Tiling r) : Prop :=
  (t.map squareArea).sum = rectangleArea r

/-- The main theorem -/
theorem min_squares_6x7 :
  ∃ (t : Tiling ⟨6, 7⟩), 
    isValidTiling ⟨6, 7⟩ t ∧ 
    t.length = 7 ∧ 
    (∀ (t' : Tiling ⟨6, 7⟩), isValidTiling ⟨6, 7⟩ t' → t'.length ≥ 7) :=
  sorry

end NUMINAMATH_CALUDE_min_squares_6x7_l2004_200497


namespace NUMINAMATH_CALUDE_neg_two_oplus_three_solve_equation_find_expression_value_l2004_200479

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := 2 * a - a * b

-- Theorem 1
theorem neg_two_oplus_three : oplus (-2) 3 = 2 := by sorry

-- Theorem 2
theorem solve_equation (x : ℝ) : oplus (-3) x = oplus (x + 1) 5 → x = 1/2 := by sorry

-- Theorem 3
theorem find_expression_value (x y : ℝ) : oplus x 1 = 2 * (oplus 1 y) → (1/2) * x + y + 1 = 3 := by sorry

end NUMINAMATH_CALUDE_neg_two_oplus_three_solve_equation_find_expression_value_l2004_200479


namespace NUMINAMATH_CALUDE_intersection_distance_difference_l2004_200453

/-- The line y - 2x - 1 = 0 intersects the parabola y^2 = 4x + 1 at points C and D.
    Q is the point (2, 0). -/
theorem intersection_distance_difference (C D Q : ℝ × ℝ) : 
  (C.2 - 2 * C.1 - 1 = 0) →
  (D.2 - 2 * D.1 - 1 = 0) →
  (C.2^2 = 4 * C.1 + 1) →
  (D.2^2 = 4 * D.1 + 1) →
  (Q = (2, 0)) →
  |Real.sqrt ((C.1 - Q.1)^2 + (C.2 - Q.2)^2) - Real.sqrt ((D.1 - Q.1)^2 + (D.2 - Q.2)^2)| = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_difference_l2004_200453


namespace NUMINAMATH_CALUDE_log_identity_l2004_200447

theorem log_identity : Real.log 4 / Real.log 10 + 2 * Real.log 5 / Real.log 10 + 
  (Real.log 5 / Real.log 2) * (Real.log 8 / Real.log 5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l2004_200447


namespace NUMINAMATH_CALUDE_drivers_distance_comparison_l2004_200472

/-- Conversion factor from miles to kilometers -/
def mile_to_km : ℝ := 1.60934

/-- Gervais's distance in miles per day -/
def gervais_miles_per_day : ℝ := 315

/-- Number of days Gervais drove -/
def gervais_days : ℕ := 3

/-- Gervais's speed in miles per hour -/
def gervais_speed : ℝ := 60

/-- Henri's total distance in miles -/
def henri_miles : ℝ := 1250

/-- Henri's speed in miles per hour -/
def henri_speed : ℝ := 50

/-- Madeleine's distance in miles per day -/
def madeleine_miles_per_day : ℝ := 100

/-- Number of days Madeleine drove -/
def madeleine_days : ℕ := 5

/-- Madeleine's speed in miles per hour -/
def madeleine_speed : ℝ := 40

/-- Calculate total distance driven by all three drivers in kilometers -/
def total_distance : ℝ :=
  (gervais_miles_per_day * gervais_days * mile_to_km) +
  (henri_miles * mile_to_km) +
  (madeleine_miles_per_day * madeleine_days * mile_to_km)

/-- Calculate Henri's distance in kilometers -/
def henri_distance : ℝ := henri_miles * mile_to_km

theorem drivers_distance_comparison :
  total_distance = 4337.16905 ∧
  henri_distance = 2011.675 ∧
  henri_distance > gervais_miles_per_day * gervais_days * mile_to_km ∧
  henri_distance > madeleine_miles_per_day * madeleine_days * mile_to_km :=
by sorry

end NUMINAMATH_CALUDE_drivers_distance_comparison_l2004_200472


namespace NUMINAMATH_CALUDE_cars_produced_in_north_america_l2004_200494

theorem cars_produced_in_north_america :
  ∀ (total_cars europe_cars north_america_cars : ℕ),
    total_cars = 6755 →
    europe_cars = 2871 →
    total_cars = north_america_cars + europe_cars →
    north_america_cars = 3884 := by
  sorry

end NUMINAMATH_CALUDE_cars_produced_in_north_america_l2004_200494


namespace NUMINAMATH_CALUDE_decomposition_theorem_l2004_200465

theorem decomposition_theorem (d n : ℕ) (hd : d > 0) (hn : n > 0) :
  ∃ (A B : Set ℕ), 
    (∀ k : ℕ, k > 0 → (k ∈ A ∨ k ∈ B)) ∧
    (A ∩ B = ∅) ∧
    (∀ x ∈ A, ∃ y ∈ B, d * x = n * d * y) ∧
    (∀ y ∈ B, ∃ x ∈ A, d * x = n * d * y) :=
sorry

end NUMINAMATH_CALUDE_decomposition_theorem_l2004_200465


namespace NUMINAMATH_CALUDE_product_remainder_by_ten_l2004_200481

theorem product_remainder_by_ten : 
  (2468 * 7531 * 92045) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_ten_l2004_200481


namespace NUMINAMATH_CALUDE_range_of_a_l2004_200424

theorem range_of_a (a : ℝ) : 
  (¬∃x ∈ Set.Icc 0 1, x^2 - 2*x - 2 + a > 0) ∧
  (¬∀x : ℝ, x^2 - 2*x - a ≠ 0) →
  a ∈ Set.Icc (-1) 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2004_200424


namespace NUMINAMATH_CALUDE_y_range_l2004_200491

theorem y_range (x y : ℝ) (h1 : |y - 2*x| = x^2) (h2 : -1 < x) (h3 : x < 0) :
  ∃ (a b : ℝ), a = -3 ∧ b = 0 ∧ a < y ∧ y < b ∧
  ∀ (z : ℝ), (∃ (w : ℝ), -1 < w ∧ w < 0 ∧ |z - 2*w| = w^2) → a ≤ z ∧ z ≤ b :=
sorry

end NUMINAMATH_CALUDE_y_range_l2004_200491


namespace NUMINAMATH_CALUDE_total_lemons_l2004_200427

def lemon_problem (levi jayden eli ian : ℕ) : Prop :=
  levi = 5 ∧
  jayden = levi + 6 ∧
  3 * jayden = eli ∧
  2 * eli = ian ∧
  levi + jayden + eli + ian = 115

theorem total_lemons : ∃ levi jayden eli ian : ℕ, lemon_problem levi jayden eli ian := by
  sorry

end NUMINAMATH_CALUDE_total_lemons_l2004_200427


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_roots_difference_one_l2004_200426

/-- The quadratic equation x^2 + (m+3)x + m+2 = 0 -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + (m+3)*x + m+2

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := (m+3)^2 - 4*(m+2)

theorem quadratic_always_real_roots (m : ℝ) :
  discriminant m ≥ 0 := by sorry

theorem roots_difference_one (m : ℝ) :
  (∃ a b, quadratic m a = 0 ∧ quadratic m b = 0 ∧ |a - b| = 1) →
  (m = -2 ∨ m = 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_roots_difference_one_l2004_200426


namespace NUMINAMATH_CALUDE_existence_of_special_point_l2004_200412

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in a 2D plane -/
def Point := ℝ × ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  point : Point
  direction : ℝ × ℝ

/-- Checks if a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 > c.radius^2

/-- Checks if a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop :=
  sorry -- Definition of line-circle intersection

/-- Main theorem -/
theorem existence_of_special_point (c1 c2 : Circle) :
  ∃ p : Point, (isOutside p c1 ∧ isOutside p c2) ∧
    ∀ l : Line, l.point = p → (intersects l c1 ∨ intersects l c2) :=
  sorry

end NUMINAMATH_CALUDE_existence_of_special_point_l2004_200412


namespace NUMINAMATH_CALUDE_statement_II_must_be_true_l2004_200473

-- Define the possible contents of the card
inductive CardContent
| Number : Nat → CardContent
| Symbol : Char → CardContent

-- Define the statements
def statementI (c : CardContent) : Prop :=
  match c with
  | CardContent.Symbol _ => True
  | CardContent.Number _ => False

def statementII (c : CardContent) : Prop :=
  match c with
  | CardContent.Symbol '%' => False
  | _ => True

def statementIII (c : CardContent) : Prop :=
  c = CardContent.Number 3

def statementIV (c : CardContent) : Prop :=
  c ≠ CardContent.Number 4

-- Theorem statement
theorem statement_II_must_be_true :
  ∃ (c : CardContent),
    (statementI c ∧ statementII c ∧ statementIII c) ∨
    (statementI c ∧ statementII c ∧ statementIV c) ∨
    (statementI c ∧ statementIII c ∧ statementIV c) ∨
    (statementII c ∧ statementIII c ∧ statementIV c) :=
  sorry

end NUMINAMATH_CALUDE_statement_II_must_be_true_l2004_200473


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_fraction_simplification_l2004_200411

-- Problem 1
theorem sqrt_expression_simplification :
  3 * Real.sqrt 2 - (Real.sqrt 3 + 2 * Real.sqrt 2) * Real.sqrt 6 = -4 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem fraction_simplification (a : ℝ) (h1 : a^2 ≠ 4) (h2 : a ≠ 2) :
  a / (a^2 - 4) + 1 / (4 - 2*a) = 1 / (2*a + 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_fraction_simplification_l2004_200411


namespace NUMINAMATH_CALUDE_factorization_proof_l2004_200499

theorem factorization_proof (a x y : ℝ) : 2*a*(x-y) - (x-y) = (x-y)*(2*a-1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2004_200499


namespace NUMINAMATH_CALUDE_mary_score_l2004_200484

def AHSME_score (c w : ℕ) : ℕ := 30 + 4 * c - w

def unique_solution (s : ℕ) : Prop :=
  ∃! (c w : ℕ), AHSME_score c w = s ∧ c + w ≤ 30

def multiple_solutions (s : ℕ) : Prop :=
  ∃ (c₁ w₁ c₂ w₂ : ℕ), c₁ ≠ c₂ ∧ AHSME_score c₁ w₁ = s ∧ AHSME_score c₂ w₂ = s ∧ c₁ + w₁ ≤ 30 ∧ c₂ + w₂ ≤ 30

theorem mary_score :
  ∃ (s : ℕ),
    s = 119 ∧
    s > 80 ∧
    unique_solution s ∧
    ∀ s', 80 < s' ∧ s' < s → multiple_solutions s' :=
by sorry

end NUMINAMATH_CALUDE_mary_score_l2004_200484


namespace NUMINAMATH_CALUDE_two_digit_number_condition_l2004_200488

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ :=
  n / 10

def units_digit (n : ℕ) : ℕ :=
  n % 10

def satisfies_condition (n : ℕ) : Prop :=
  2 * (tens_digit n + units_digit n) = tens_digit n * units_digit n

theorem two_digit_number_condition :
  ∀ n : ℕ, is_valid_two_digit_number n ∧ satisfies_condition n ↔ n = 36 ∨ n = 44 ∨ n = 63 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_condition_l2004_200488


namespace NUMINAMATH_CALUDE_evaluate_expression_l2004_200485

theorem evaluate_expression (x : ℝ) (h : x = -3) : 
  (5 + x*(5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2004_200485


namespace NUMINAMATH_CALUDE_division_equations_l2004_200405

theorem division_equations (h : 40 * 60 = 2400) : 
  (2400 / 40 = 60) ∧ (2400 / 60 = 40) := by
  sorry

end NUMINAMATH_CALUDE_division_equations_l2004_200405


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2004_200438

theorem inscribed_cube_surface_area (outer_cube_area : ℝ) (h : outer_cube_area = 54) :
  let outer_side := Real.sqrt (outer_cube_area / 6)
  let sphere_diameter := outer_side
  let inner_side := Real.sqrt (sphere_diameter^2 / 3)
  let inner_cube_area := 6 * inner_side^2
  inner_cube_area = 18 := by sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2004_200438


namespace NUMINAMATH_CALUDE_smallest_candy_count_l2004_200404

theorem smallest_candy_count : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 10) % 6 = 0 ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (m + 7) % 9 = 0 ∧ (m - 10) % 6 = 0) → False) ∧
  n = 146 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l2004_200404


namespace NUMINAMATH_CALUDE_fifteenth_row_seats_l2004_200474

/-- Represents the number of seats in a row of the stadium -/
def seats (n : ℕ) : ℕ := 5 + 2 * (n - 1)

/-- Theorem stating that the 15th row has 33 seats -/
theorem fifteenth_row_seats :
  seats 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_row_seats_l2004_200474


namespace NUMINAMATH_CALUDE_pinwheel_area_l2004_200440

-- Define the constants
def π : ℝ := 3.14
def large_diameter : ℝ := 20

-- Define the theorem
theorem pinwheel_area : 
  let large_radius : ℝ := large_diameter / 2
  let small_radius : ℝ := large_radius / 2
  let large_area : ℝ := π * large_radius ^ 2
  let small_area : ℝ := π * small_radius ^ 2
  let cut_out_area : ℝ := 2 * small_area
  let remaining_area : ℝ := large_area - cut_out_area
  remaining_area = 157 := by
sorry

end NUMINAMATH_CALUDE_pinwheel_area_l2004_200440


namespace NUMINAMATH_CALUDE_prob_one_makes_shot_is_point_seven_l2004_200493

/-- The probability that at least one player makes a shot -/
def prob_at_least_one_makes_shot (prob_a prob_b : ℝ) : ℝ :=
  1 - (1 - prob_a) * (1 - prob_b)

/-- Theorem: Given the shooting success rates of players A and B,
    the probability that at least one of them makes a shot is 0.7 -/
theorem prob_one_makes_shot_is_point_seven :
  prob_at_least_one_makes_shot 0.5 0.4 = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_makes_shot_is_point_seven_l2004_200493


namespace NUMINAMATH_CALUDE_distance_product_range_l2004_200410

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y^2 = 4*x
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 8

-- Define a line with slope 45°
def line_slope_45 (x₁ y₁ x₂ y₂ : ℝ) : Prop := y₂ - y₁ = x₂ - x₁

-- Define the property of a point being on a curve
def point_on_curve (C : ℝ → ℝ → Prop) (x y : ℝ) : Prop := C x y

-- Define the property of a line intersecting a curve at two distinct points
def line_intersects_curve_at_two_points (C : ℝ → ℝ → Prop) (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  line_slope_45 x₁ y₁ x₂ y₂ ∧ line_slope_45 x₁ y₁ x₃ y₃ ∧
  point_on_curve C x₂ y₂ ∧ point_on_curve C x₃ y₃ ∧
  (x₂ ≠ x₃ ∨ y₂ ≠ y₃)

-- Define the distance between two points
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := ((x₂ - x₁)^2 + (y₂ - y₁)^2)^(1/2)

-- Define the product of distances
def distance_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  distance x₁ y₁ x₂ y₂ * distance x₁ y₁ x₃ y₃

-- Main theorem
theorem distance_product_range :
  ∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    point_on_curve C₁ x₁ y₁ →
    line_intersects_curve_at_two_points C₂ x₁ y₁ x₂ y₂ x₃ y₃ →
    4 ≤ distance_product x₁ y₁ x₂ y₂ x₃ y₃ ∧
    distance_product x₁ y₁ x₂ y₂ x₃ y₃ < 8 ∨
    8 < distance_product x₁ y₁ x₂ y₂ x₃ y₃ ∧
    distance_product x₁ y₁ x₂ y₂ x₃ y₃ ≤ 200 :=
sorry

end NUMINAMATH_CALUDE_distance_product_range_l2004_200410


namespace NUMINAMATH_CALUDE_composite_sum_product_l2004_200475

theorem composite_sum_product (a b c d e : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧
  a^4 + b^4 = c^4 + d^4 ∧
  a^4 + b^4 = e^5 →
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ x * y = a * c + b * d :=
by sorry

end NUMINAMATH_CALUDE_composite_sum_product_l2004_200475


namespace NUMINAMATH_CALUDE_range_of_m_l2004_200487

theorem range_of_m (x y m : ℝ) : 
  x > 0 → 
  y > 0 → 
  2 / x + 1 / y = 1 → 
  (∀ (x y : ℝ), x > 0 → y > 0 → 2 / x + 1 / y = 1 → x^2 + 2*x*y > m^2 + 2*m) → 
  m > -4 ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2004_200487


namespace NUMINAMATH_CALUDE_max_value_of_f_l2004_200430

/-- The function f(x) = |x| - |x - 3| -/
def f (x : ℝ) : ℝ := |x| - |x - 3|

/-- The maximum value of f(x) is 3 -/
theorem max_value_of_f : 
  ∃ (M : ℝ), M = 3 ∧ ∀ x, f x ≤ M ∧ ∃ y, f y = M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2004_200430


namespace NUMINAMATH_CALUDE_return_trip_time_l2004_200460

/-- Represents the flight scenario between two towns -/
structure FlightScenario where
  d : ℝ  -- distance between towns
  p : ℝ  -- plane speed in still air
  w : ℝ  -- wind speed
  t : ℝ  -- time for return trip in still air

/-- The conditions of the flight scenario -/
def flight_conditions (f : FlightScenario) : Prop :=
  f.w = (1/3) * f.p ∧  -- wind speed is one-third of plane speed
  f.d / (f.p - f.w) = 120 ∧  -- time against wind
  f.d / (f.p + f.w) = f.t - 20  -- time with wind

/-- The theorem to prove -/
theorem return_trip_time (f : FlightScenario) 
  (h : flight_conditions f) : f.d / (f.p + f.w) = 60 := by
  sorry

#check return_trip_time

end NUMINAMATH_CALUDE_return_trip_time_l2004_200460


namespace NUMINAMATH_CALUDE_order_of_powers_l2004_200452

theorem order_of_powers : 
  let a : ℕ := 3^55
  let b : ℕ := 4^44
  let c : ℕ := 5^33
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_powers_l2004_200452


namespace NUMINAMATH_CALUDE_divisible_by_three_l2004_200434

theorem divisible_by_three (k : ℤ) : 3 ∣ ((2*k + 3)^2 - 4*k^2) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l2004_200434


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l2004_200459

theorem quadratic_root_condition (a : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ > 1 ∧ r₂ < 1 ∧ r₁^2 + 2*a*r₁ + 1 = 0 ∧ r₂^2 + 2*a*r₂ + 1 = 0) → 
  a < -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l2004_200459


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l2004_200449

theorem greatest_integer_b_for_all_real_domain : ∃ b : ℤ,
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 10 ≠ 0) ∧
  (∀ c : ℤ, c > b → ∃ x : ℝ, x^2 + (c : ℝ) * x + 10 = 0) ∧
  b = 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l2004_200449


namespace NUMINAMATH_CALUDE_problem_statement_l2004_200428

theorem problem_statement (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 1) 
  (heq : a^2 + 4*b^2 + c^2 - 2*c = 2) : 
  (a + 2*b + c ≤ 4) ∧ 
  (a = 2*b → 1/b + 1/(c-1) ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2004_200428


namespace NUMINAMATH_CALUDE_dividend_calculation_l2004_200433

theorem dividend_calculation (dividend divisor : ℕ) : 
  (dividend / divisor = 4) → 
  (dividend % divisor = 3) → 
  (dividend + divisor + 4 + 3 = 100) → 
  dividend = 75 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2004_200433
