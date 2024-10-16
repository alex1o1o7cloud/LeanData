import Mathlib

namespace NUMINAMATH_CALUDE_min_value_expression_l2539_253997

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2539_253997


namespace NUMINAMATH_CALUDE_range_of_x_for_inequality_l2539_253960

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem range_of_x_for_inequality (x : ℝ) :
  (∀ m : ℝ, m ∈ Set.Icc (-2) 2 → f (m*x - 2) + f x < 0) →
  x ∈ Set.Ioo (-2) (2/3) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_for_inequality_l2539_253960


namespace NUMINAMATH_CALUDE_election_results_l2539_253962

/-- Represents a candidate in the election -/
inductive Candidate
  | Montoran
  | AjudaPinto
  | VidameOfOussel

/-- Represents a voter group with their preferences -/
structure VoterGroup where
  size : Nat
  preferences : List Candidate

/-- Represents the election setup -/
structure Election where
  totalVoters : Nat
  candidates : List Candidate
  voterGroups : List VoterGroup

/-- One-round voting system -/
def oneRoundWinner (e : Election) : Candidate := sorry

/-- Two-round voting system -/
def twoRoundWinner (e : Election) : Candidate := sorry

/-- Three-round voting system -/
def threeRoundWinner (e : Election) : Candidate := sorry

/-- The election setup based on the problem description -/
def electionSetup : Election :=
  { totalVoters := 100000
  , candidates := [Candidate.Montoran, Candidate.AjudaPinto, Candidate.VidameOfOussel]
  , voterGroups :=
    [ { size := 33000
      , preferences := [Candidate.Montoran, Candidate.AjudaPinto, Candidate.VidameOfOussel]
      }
    , { size := 18000
      , preferences := [Candidate.AjudaPinto, Candidate.Montoran, Candidate.VidameOfOussel]
      }
    , { size := 12000
      , preferences := [Candidate.AjudaPinto, Candidate.VidameOfOussel, Candidate.Montoran]
      }
    , { size := 37000
      , preferences := [Candidate.VidameOfOussel, Candidate.AjudaPinto, Candidate.Montoran]
      }
    ]
  }

theorem election_results (e : Election) :
  e = electionSetup →
  oneRoundWinner e = Candidate.VidameOfOussel ∧
  twoRoundWinner e = Candidate.Montoran ∧
  threeRoundWinner e = Candidate.AjudaPinto :=
sorry

end NUMINAMATH_CALUDE_election_results_l2539_253962


namespace NUMINAMATH_CALUDE_right_triangle_circumradius_l2539_253978

theorem right_triangle_circumradius (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) :
  a^2 + b^2 = c^2 → (c / 2 : ℝ) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circumradius_l2539_253978


namespace NUMINAMATH_CALUDE_greatest_bound_of_r2_l2539_253977

/-- The function f(x) = x^2 - r_2x + r_3 -/
def f (r_2 r_3 : ℝ) (x : ℝ) : ℝ := x^2 - r_2*x + r_3

/-- The sequence g_n defined recursively -/
def g (r_2 r_3 : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f r_2 r_3 (g r_2 r_3 n)

/-- The property that g_{2i} < g_{2i+1} and g_{2i+1} > g_{2i+2} for 0 ≤ i ≤ 2011 -/
def alternating_property (r_2 r_3 : ℝ) : Prop :=
  ∀ i : ℕ, i ≤ 2011 → g r_2 r_3 (2*i) < g r_2 r_3 (2*i + 1) ∧ g r_2 r_3 (2*i + 1) > g r_2 r_3 (2*i + 2)

/-- The property that there exists j such that g_{i+1} > g_i for all i > j -/
def eventually_increasing (r_2 r_3 : ℝ) : Prop :=
  ∃ j : ℕ, ∀ i : ℕ, i > j → g r_2 r_3 (i + 1) > g r_2 r_3 i

/-- The property that the sequence g_n is unbounded -/
def unbounded_sequence (r_2 r_3 : ℝ) : Prop :=
  ∀ M : ℝ, ∃ N : ℕ, g r_2 r_3 N > M

/-- The main theorem -/
theorem greatest_bound_of_r2 :
  (∃ A : ℝ, ∀ r_2 r_3 : ℝ, 
    alternating_property r_2 r_3 → 
    eventually_increasing r_2 r_3 → 
    unbounded_sequence r_2 r_3 → 
    A ≤ |r_2| ∧ 
    (∀ B : ℝ, (∀ r_2' r_3' : ℝ, 
      alternating_property r_2' r_3' → 
      eventually_increasing r_2' r_3' → 
      unbounded_sequence r_2' r_3' → 
      B ≤ |r_2'|) → B ≤ A)) ∧
  (∀ A : ℝ, (∀ r_2 r_3 : ℝ, 
    alternating_property r_2 r_3 → 
    eventually_increasing r_2 r_3 → 
    unbounded_sequence r_2 r_3 → 
    A ≤ |r_2| ∧ 
    (∀ B : ℝ, (∀ r_2' r_3' : ℝ, 
      alternating_property r_2' r_3' → 
      eventually_increasing r_2' r_3' → 
      unbounded_sequence r_2' r_3' → 
      B ≤ |r_2'|) → B ≤ A)) → A = 2) := by
  sorry

end NUMINAMATH_CALUDE_greatest_bound_of_r2_l2539_253977


namespace NUMINAMATH_CALUDE_crayon_cost_theorem_l2539_253945

/-- The number of crayons in half a dozen -/
def half_dozen : ℕ := 6

/-- The number of half dozens bought -/
def num_half_dozens : ℕ := 4

/-- The cost of each crayon in dollars -/
def cost_per_crayon : ℕ := 2

/-- The total number of crayons bought -/
def total_crayons : ℕ := num_half_dozens * half_dozen

/-- The total cost of the crayons in dollars -/
def total_cost : ℕ := total_crayons * cost_per_crayon

theorem crayon_cost_theorem : total_cost = 48 := by
  sorry

end NUMINAMATH_CALUDE_crayon_cost_theorem_l2539_253945


namespace NUMINAMATH_CALUDE_vintik_votes_l2539_253976

theorem vintik_votes (total_percentage : ℝ) (shpuntik_votes : ℕ) 
  (h1 : total_percentage = 146)
  (h2 : shpuntik_votes > 1000) :
  ∃ (vintik_votes : ℕ), vintik_votes > 850 := by
  sorry

end NUMINAMATH_CALUDE_vintik_votes_l2539_253976


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l2539_253972

/-- A geometric sequence with specified terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  geometric_sequence a → a 2 = -2 → a 6 = -32 → a 4 = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l2539_253972


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2539_253910

theorem parabola_line_intersection (k : ℝ) : 
  (∃! x : ℝ, -2 = x^2 + k*x - 1) → (k = 2 ∨ k = -2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2539_253910


namespace NUMINAMATH_CALUDE_artist_profit_calculation_l2539_253932

/-- Calculates the total profit for an artist given contest winnings, painting sales, and expenses. -/
theorem artist_profit_calculation 
  (contest_prize : ℕ) 
  (num_paintings_sold : ℕ) 
  (price_per_painting : ℕ) 
  (art_supplies_cost : ℕ) 
  (exhibition_fee : ℕ) 
  (h1 : contest_prize = 150)
  (h2 : num_paintings_sold = 3)
  (h3 : price_per_painting = 50)
  (h4 : art_supplies_cost = 40)
  (h5 : exhibition_fee = 20) :
  contest_prize + num_paintings_sold * price_per_painting - (art_supplies_cost + exhibition_fee) = 240 :=
by sorry

end NUMINAMATH_CALUDE_artist_profit_calculation_l2539_253932


namespace NUMINAMATH_CALUDE_arithmetic_progression_theorem_l2539_253971

/-- An arithmetic progression with a non-zero common difference -/
def ArithmeticProgression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- The condition that b_n is also an arithmetic progression -/
def BnIsArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ d' : ℝ, d' ≠ 0 ∧ ∀ n, a (n + 1) * Real.cos (a (n + 1)) = a n * Real.cos (a n) + d'

/-- The given equation holds for all n -/
def EquationHolds (a : ℕ → ℝ) : Prop :=
  ∀ n, Real.sin (2 * a n) + Real.cos (a (n + 1)) = 0

theorem arithmetic_progression_theorem (a : ℕ → ℝ) (d : ℝ) :
  ArithmeticProgression a d →
  BnIsArithmeticProgression a →
  EquationHolds a →
  (∃ m k : ℤ, k ≠ 0 ∧ 
    ((a 1 = -π / 6 + 2 * π * ↑m ∧ d = 2 * π * ↑k) ∨
     (a 1 = -5 * π / 6 + 2 * π * ↑m ∧ d = 2 * π * ↑k))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_theorem_l2539_253971


namespace NUMINAMATH_CALUDE_unique_a_for_three_element_set_l2539_253933

theorem unique_a_for_three_element_set : ∃! (a : ℝ), 
  let A : Set ℝ := {a^2, 2-a, 4}
  (Fintype.card A = 3) ∧ (a = 6) := by sorry

end NUMINAMATH_CALUDE_unique_a_for_three_element_set_l2539_253933


namespace NUMINAMATH_CALUDE_locus_of_symmetric_point_l2539_253907

/-- Given a parabola y = x^2, a fixed point A(a, 0) where a ≠ 0, and a moving point P on the parabola,
    the point Q symmetric to A with respect to P has the locus y = (1/2)(x + a)^2 -/
theorem locus_of_symmetric_point (a : ℝ) (ha : a ≠ 0) :
  ∀ x₁ y₁ x y : ℝ,
  y₁ = x₁^2 →                        -- P(x₁, y₁) is on the parabola y = x^2
  x = 2*a - x₁ →                     -- x-coordinate of Q
  y = -y₁ →                          -- y-coordinate of Q
  y = (1/2) * (x + a)^2 := by sorry

end NUMINAMATH_CALUDE_locus_of_symmetric_point_l2539_253907


namespace NUMINAMATH_CALUDE_functional_relationship_l2539_253993

/-- Given a function y that is the sum of two components y₁ and y₂,
    where y₁ is directly proportional to x and y₂ is inversely proportional to (x-2),
    prove that y = x + 2/(x-2) when y = -1 at x = 1 and y = 5 at x = 3. -/
theorem functional_relationship (y y₁ y₂ : ℝ → ℝ) (k₁ k₂ : ℝ) :
  (∀ x, y x = y₁ x + y₂ x) →
  (∀ x, y₁ x = k₁ * x) →
  (∀ x, y₂ x = k₂ / (x - 2)) →
  y 1 = -1 →
  y 3 = 5 →
  ∀ x, y x = x + 2 / (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_functional_relationship_l2539_253993


namespace NUMINAMATH_CALUDE_marble_175_is_white_l2539_253994

/-- Represents the color of a marble -/
inductive MarbleColor
| Gray
| White
| Black

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 12 with
  | 0 | 1 | 2 | 3 | 4 => MarbleColor.Gray
  | 5 | 6 | 7 | 8 => MarbleColor.White
  | _ => MarbleColor.Black

/-- Theorem stating that the 175th marble is white -/
theorem marble_175_is_white : marbleColor 175 = MarbleColor.White := by
  sorry

end NUMINAMATH_CALUDE_marble_175_is_white_l2539_253994


namespace NUMINAMATH_CALUDE_additional_money_needed_l2539_253998

def michaels_money : ℕ := 50
def total_cost : ℕ := 20 + 36 + 5

theorem additional_money_needed : total_cost - michaels_money = 11 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l2539_253998


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l2539_253947

/-- Calculates the sampling interval for systematic sampling -/
def samplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The sampling interval for a population of 1200 and sample size of 40 is 30 -/
theorem systematic_sampling_interval :
  samplingInterval 1200 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l2539_253947


namespace NUMINAMATH_CALUDE_crayons_given_to_friends_l2539_253937

theorem crayons_given_to_friends 
  (initial_crayons : ℕ)
  (lost_crayons : ℕ)
  (extra_crayons_given : ℕ)
  (h1 : initial_crayons = 589)
  (h2 : lost_crayons = 161)
  (h3 : extra_crayons_given = 410) :
  lost_crayons + extra_crayons_given = 571 :=
by
  sorry

end NUMINAMATH_CALUDE_crayons_given_to_friends_l2539_253937


namespace NUMINAMATH_CALUDE_point_on_transformed_plane_l2539_253921

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem to be proved -/
theorem point_on_transformed_plane :
  let A : Point3D := { x := 4, y := 3, z := 1 }
  let a : Plane := { a := 3, b := -4, c := 5, d := -6 }
  let k : ℝ := 5/6
  pointOnPlane A (transformPlane a k) := by
  sorry

end NUMINAMATH_CALUDE_point_on_transformed_plane_l2539_253921


namespace NUMINAMATH_CALUDE_length_AB_area_OCD_l2539_253959

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define line l passing through the focus and perpendicular to x-axis
def line_l (x y : ℝ) : Prop := x = 2

-- Define line l1 passing through the focus with slope angle 45°
def line_l1 (x y : ℝ) : Prop := y = x - 2

-- Theorem 1: Length of AB
theorem length_AB : 
  ∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line_l A.1 A.2 ∧ 
    line_l B.1 B.2 ∧ 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by sorry

-- Theorem 2: Area of triangle OCD
theorem area_OCD : 
  ∃ C D : ℝ × ℝ, 
    parabola C.1 C.2 ∧ 
    parabola D.1 D.2 ∧ 
    line_l1 C.1 C.2 ∧ 
    line_l1 D.1 D.2 ∧ 
    (1/2) * Real.sqrt (C.1^2 + C.2^2) * Real.sqrt (D.1^2 + D.2^2) * 
    Real.sin (Real.arccos ((C.1*D.1 + C.2*D.2) / (Real.sqrt (C.1^2 + C.2^2) * Real.sqrt (D.1^2 + D.2^2)))) = 8 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_length_AB_area_OCD_l2539_253959


namespace NUMINAMATH_CALUDE_farm_animals_ratio_l2539_253991

theorem farm_animals_ratio :
  let goats : ℕ := 66
  let chickens : ℕ := 2 * goats
  let ducks : ℕ := (goats + chickens) / 2
  let pigs : ℕ := goats - 33
  (pigs : ℚ) / ducks = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_ratio_l2539_253991


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l2539_253966

theorem simple_interest_principal_calculation
  (rate : ℝ) (interest : ℝ) (time : ℝ) (principal : ℝ)
  (h_rate : rate = 4.166666666666667)
  (h_interest : interest = 130)
  (h_time : time = 4)
  (h_formula : interest = principal * rate * time / 100) :
  principal = 780 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l2539_253966


namespace NUMINAMATH_CALUDE_three_statements_imply_negation_l2539_253970

theorem three_statements_imply_negation (p q : Prop) :
  let statement1 := p ∨ q
  let statement2 := p ∨ ¬q
  let statement3 := p ∧ ¬q
  let statement4 := ¬p ∨ ¬q
  let negation_of_both_false := ¬(¬p ∧ ¬q)
  (statement1 → negation_of_both_false) ∧
  (statement2 → negation_of_both_false) ∧
  (statement3 → negation_of_both_false) ∧
  ¬(statement4 → negation_of_both_false) := by
  sorry

end NUMINAMATH_CALUDE_three_statements_imply_negation_l2539_253970


namespace NUMINAMATH_CALUDE_second_largest_prime_factor_of_sum_of_divisors_450_l2539_253973

def sum_of_divisors (n : ℕ) : ℕ := sorry

def second_largest_prime_factor (n : ℕ) : ℕ := sorry

theorem second_largest_prime_factor_of_sum_of_divisors_450 :
  second_largest_prime_factor (sum_of_divisors 450) = 13 := by sorry

end NUMINAMATH_CALUDE_second_largest_prime_factor_of_sum_of_divisors_450_l2539_253973


namespace NUMINAMATH_CALUDE_arithmetic_sequence_201_l2539_253989

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n : ℕ, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_201 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_5 : a 5 = 33) 
  (h_45 : a 45 = 153) : 
  a 61 = 201 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_201_l2539_253989


namespace NUMINAMATH_CALUDE_inverse_co_complementary_angles_equal_l2539_253964

/-- For any two angles α and β, if their co-complementary angles are equal, then α and β are equal. -/
theorem inverse_co_complementary_angles_equal (α β : Real) :
  (90 - α = 90 - β) → α = β := by
  sorry

end NUMINAMATH_CALUDE_inverse_co_complementary_angles_equal_l2539_253964


namespace NUMINAMATH_CALUDE_min_value_expression_l2539_253911

theorem min_value_expression (x y z : ℝ) 
  (hx : -1 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 1) 
  (hz : -1 < z ∧ z < 1) : 
  1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) ≥ 2 ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → 1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2539_253911


namespace NUMINAMATH_CALUDE_clothing_price_difference_l2539_253941

theorem clothing_price_difference :
  ∀ (x y : ℝ),
    9 * x + 10 * y = 1810 →
    11 * x + 8 * y = 1790 →
    x - y = -10 :=
by sorry

end NUMINAMATH_CALUDE_clothing_price_difference_l2539_253941


namespace NUMINAMATH_CALUDE_average_sale_calculation_l2539_253999

def sales : List ℕ := [5266, 5768, 5922, 5678, 6029]
def required_sale : ℕ := 4937

theorem average_sale_calculation (sales : List ℕ) (required_sale : ℕ) :
  sales = [5266, 5768, 5922, 5678, 6029] →
  required_sale = 4937 →
  (sales.sum + required_sale) / 6 = 5600 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_calculation_l2539_253999


namespace NUMINAMATH_CALUDE_fourth_student_guess_is_525_l2539_253957

/-- Represents the number of jellybeans guessed by each student -/
def jellybean_guess : Fin 4 → ℕ
  | 0 => 100  -- First student's guess
  | 1 => 8 * jellybean_guess 0  -- Second student's guess
  | 2 => jellybean_guess 1 - 200  -- Third student's guess
  | 3 => (jellybean_guess 0 + jellybean_guess 1 + jellybean_guess 2) / 3 + 25  -- Fourth student's guess

/-- Theorem stating that the fourth student's guess is 525 -/
theorem fourth_student_guess_is_525 : jellybean_guess 3 = 525 := by
  sorry

end NUMINAMATH_CALUDE_fourth_student_guess_is_525_l2539_253957


namespace NUMINAMATH_CALUDE_books_checked_out_thursday_l2539_253909

theorem books_checked_out_thursday (initial_books : ℕ) (wednesday_checkout : ℕ) 
  (thursday_return : ℕ) (friday_return : ℕ) (final_books : ℕ) :
  initial_books = 98 →
  wednesday_checkout = 43 →
  thursday_return = 23 →
  friday_return = 7 →
  final_books = 80 →
  ∃ (thursday_checkout : ℕ),
    final_books = initial_books - wednesday_checkout + thursday_return - thursday_checkout + friday_return ∧
    thursday_checkout = 5 :=
by sorry

end NUMINAMATH_CALUDE_books_checked_out_thursday_l2539_253909


namespace NUMINAMATH_CALUDE_sinusoidal_function_parameters_l2539_253981

open Real

theorem sinusoidal_function_parameters 
  (f : ℝ → ℝ)
  (ω φ : ℝ)
  (h1 : ∀ x, f x = 2 * sin (ω * x + φ))
  (h2 : ω > 0)
  (h3 : abs φ < π)
  (h4 : f (5 * π / 8) = 2)
  (h5 : f (11 * π / 8) = 0)
  (h6 : ∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ 3 * π) :
  ω = 2 / 3 ∧ φ = π / 12 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_function_parameters_l2539_253981


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l2539_253916

/-- A bag containing red and white balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- An event when drawing balls from the bag -/
structure Event (bag : Bag) where
  pred : (ℕ × ℕ) → Prop

/-- The bag in our problem -/
def problem_bag : Bag := { red := 3, white := 3 }

/-- The event "At least 2 white balls" -/
def at_least_2_white (bag : Bag) : Event bag :=
  { pred := λ (r, w) => w ≥ 2 }

/-- The event "All red balls" -/
def all_red (bag : Bag) : Event bag :=
  { pred := λ (r, w) => r = 3 ∧ w = 0 }

/-- Two events are mutually exclusive -/
def mutually_exclusive (bag : Bag) (e1 e2 : Event bag) : Prop :=
  ∀ r w, (r + w = 3) → ¬(e1.pred (r, w) ∧ e2.pred (r, w))

/-- Two events are contradictory -/
def contradictory (bag : Bag) (e1 e2 : Event bag) : Prop :=
  ∀ r w, (r + w = 3) → (e1.pred (r, w) ↔ ¬e2.pred (r, w))

/-- The main theorem to prove -/
theorem events_mutually_exclusive_not_contradictory :
  mutually_exclusive problem_bag (at_least_2_white problem_bag) (all_red problem_bag) ∧
  ¬contradictory problem_bag (at_least_2_white problem_bag) (all_red problem_bag) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l2539_253916


namespace NUMINAMATH_CALUDE_pages_read_tomorrow_l2539_253904

/-- The number of pages Melody needs to read for her English class -/
def english_pages : ℕ := 50

/-- The number of pages Melody needs to read for her Math class -/
def math_pages : ℕ := 30

/-- The number of pages Melody needs to read for her History class -/
def history_pages : ℕ := 20

/-- The number of pages Melody needs to read for her Chinese class -/
def chinese_pages : ℕ := 40

/-- The fraction of English pages Melody will read tomorrow -/
def english_fraction : ℚ := 1 / 5

/-- The percentage of Math pages Melody will read tomorrow -/
def math_percentage : ℚ := 30 / 100

/-- The fraction of History pages Melody will read tomorrow -/
def history_fraction : ℚ := 1 / 4

/-- The percentage of Chinese pages Melody will read tomorrow -/
def chinese_percentage : ℚ := 125 / 1000

/-- Theorem stating the total number of pages Melody will read tomorrow -/
theorem pages_read_tomorrow :
  (english_fraction * english_pages).floor +
  (math_percentage * math_pages).floor +
  (history_fraction * history_pages).floor +
  (chinese_percentage * chinese_pages).floor = 29 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_tomorrow_l2539_253904


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l2539_253951

/-- The equation 9x^2 - 36y^2 = 36 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), 9 * x^2 - 36 * y^2 = 36 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l2539_253951


namespace NUMINAMATH_CALUDE_square_sum_nonzero_iff_not_both_zero_l2539_253963

theorem square_sum_nonzero_iff_not_both_zero (a b : ℝ) :
  a^2 + b^2 ≠ 0 ↔ ¬(a = 0 ∧ b = 0) :=
sorry

end NUMINAMATH_CALUDE_square_sum_nonzero_iff_not_both_zero_l2539_253963


namespace NUMINAMATH_CALUDE_trig_fraction_equality_l2539_253983

theorem trig_fraction_equality (x : ℝ) (h : (1 - Real.sin x) / Real.cos x = 3/5) :
  Real.cos x / (1 + Real.sin x) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_equality_l2539_253983


namespace NUMINAMATH_CALUDE_added_amount_proof_l2539_253908

theorem added_amount_proof (n x : ℝ) : n = 20 → (1/2) * n + x = 15 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_added_amount_proof_l2539_253908


namespace NUMINAMATH_CALUDE_factorial_inequality_l2539_253918

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem factorial_inequality :
  factorial (factorial 100) < (factorial 99)^(factorial 100) * (factorial 100)^(factorial 99) :=
by sorry

end NUMINAMATH_CALUDE_factorial_inequality_l2539_253918


namespace NUMINAMATH_CALUDE_swimming_ratio_proof_l2539_253915

/-- Given information about the swimming abilities of Yvonne, Joel, and their younger sister,
    prove that the ratio of laps swum by the younger sister to Yvonne is 1:2. -/
theorem swimming_ratio_proof (yvonne_laps joel_laps : ℕ) (joel_ratio : ℕ) :
  yvonne_laps = 10 →
  joel_laps = 15 →
  joel_ratio = 3 →
  (joel_laps / joel_ratio : ℚ) / yvonne_laps = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_swimming_ratio_proof_l2539_253915


namespace NUMINAMATH_CALUDE_circle_area_right_triangle_circle_area_right_triangle_value_l2539_253940

/-- The area of a circle passing through the vertices of a right triangle with legs of lengths 4 and 3 -/
theorem circle_area_right_triangle (π : ℝ) : ℝ :=
  let a : ℝ := 3  -- Length of one leg
  let b : ℝ := 4  -- Length of the other leg
  let c : ℝ := Real.sqrt (a^2 + b^2)  -- Length of the hypotenuse
  let r : ℝ := c / 2  -- Radius of the circle
  π * r^2

/-- The area of the circle is equal to 25π/4 -/
theorem circle_area_right_triangle_value (π : ℝ) :
  circle_area_right_triangle π = 25 / 4 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_right_triangle_circle_area_right_triangle_value_l2539_253940


namespace NUMINAMATH_CALUDE_foldable_rectangle_short_side_l2539_253923

/-- A rectangle with the property that when folded along its diagonal,
    it forms a trapezoid with three equal sides. -/
structure FoldableRectangle where
  long_side : ℝ
  short_side : ℝ
  long_side_positive : 0 < long_side
  short_side_positive : 0 < short_side
  long_side_longer : short_side ≤ long_side
  forms_equal_sided_trapezoid : True  -- This is a placeholder for the folding property

/-- The theorem stating that a rectangle with longer side 12 cm, when folded to form
    a trapezoid with three equal sides, has a shorter side of 4√3 cm. -/
theorem foldable_rectangle_short_side
  (rect : FoldableRectangle)
  (h_long : rect.long_side = 12) :
  rect.short_side = 4 * Real.sqrt 3 := by
  sorry

#check foldable_rectangle_short_side

end NUMINAMATH_CALUDE_foldable_rectangle_short_side_l2539_253923


namespace NUMINAMATH_CALUDE_simple_interest_theorem_l2539_253979

def simple_interest_problem (principal rate time : ℝ) : Prop :=
  let simple_interest := principal * rate * time / 100
  principal - simple_interest = 2080

theorem simple_interest_theorem :
  simple_interest_problem 2600 4 5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_theorem_l2539_253979


namespace NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l2539_253926

theorem largest_prime_divisor_factorial_sum : 
  ∃ p : Nat, 
    Nat.Prime p ∧ 
    p ∣ (Nat.factorial 13 + Nat.factorial 14) ∧
    ∀ q : Nat, Nat.Prime q → q ∣ (Nat.factorial 13 + Nat.factorial 14) → q ≤ p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l2539_253926


namespace NUMINAMATH_CALUDE_sector_arc_length_l2539_253938

/-- Given a circular sector with central angle 2π/3 and area 25π/3, 
    its arc length is 10π/3 -/
theorem sector_arc_length (α : Real) (S : Real) (l : Real) :
  α = 2 * π / 3 →
  S = 25 * π / 3 →
  l = 10 * π / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_sector_arc_length_l2539_253938


namespace NUMINAMATH_CALUDE_abs_five_necessary_not_sufficient_l2539_253903

theorem abs_five_necessary_not_sufficient :
  (∀ x : ℝ, x = 5 → |x| = 5) ∧
  ¬(∀ x : ℝ, |x| = 5 → x = 5) :=
by sorry

end NUMINAMATH_CALUDE_abs_five_necessary_not_sufficient_l2539_253903


namespace NUMINAMATH_CALUDE_digitSquareSequenceReaches1Or4_l2539_253955

/-- Sum of squares of digits of a natural number -/
def sumOfSquaresOfDigits (n : ℕ) : ℕ := sorry

/-- The sequence of sum of squares of digits -/
def digitSquareSequence (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => sumOfSquaresOfDigits (digitSquareSequence start n)

/-- Predicate to check if a number is three digits -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem digitSquareSequenceReaches1Or4 (start : ℕ) (h : isThreeDigit start) :
  ∃ (k : ℕ), digitSquareSequence start k = 1 ∨ digitSquareSequence start k = 4 := by sorry

end NUMINAMATH_CALUDE_digitSquareSequenceReaches1Or4_l2539_253955


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2539_253990

/-- A quadrilateral with right angles at B and D, diagonal AC of length 5,
    and two sides with distinct integer lengths has an area of 12. -/
theorem quadrilateral_area : ∀ (A B C D : ℝ × ℝ),
  -- Right angles at B and D
  (B.2 - A.2) * (C.1 - B.1) + (B.1 - A.1) * (C.2 - B.2) = 0 →
  (D.2 - C.2) * (A.1 - D.1) + (D.1 - C.1) * (A.2 - D.2) = 0 →
  -- Diagonal AC = 5
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 25 →
  -- Two sides with distinct integer lengths
  ∃ (a b : ℕ), a ≠ b ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = a^2 ∧
     (B.1 - C.1)^2 + (B.2 - C.2)^2 = b^2) ∨
    ((A.1 - D.1)^2 + (A.2 - D.2)^2 = a^2 ∧
     (D.1 - C.1)^2 + (D.2 - C.2)^2 = b^2) →
  -- Area of ABCD is 12
  abs ((A.1 - C.1) * (B.2 - D.2) - (A.2 - C.2) * (B.1 - D.1)) / 2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2539_253990


namespace NUMINAMATH_CALUDE_deposit_equals_3400_l2539_253952

/-- Sheela's monthly income in rupees -/
def monthly_income : ℚ := 22666.67

/-- The percentage of monthly income deposited -/
def deposit_percentage : ℚ := 15

/-- The amount deposited in the bank savings account -/
def deposit_amount : ℚ := (deposit_percentage / 100) * monthly_income

/-- Theorem stating that the deposit amount is equal to 3400 rupees -/
theorem deposit_equals_3400 : deposit_amount = 3400 := by
  sorry

end NUMINAMATH_CALUDE_deposit_equals_3400_l2539_253952


namespace NUMINAMATH_CALUDE_arithmetic_proof_l2539_253987

theorem arithmetic_proof : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l2539_253987


namespace NUMINAMATH_CALUDE_oldest_child_age_l2539_253939

theorem oldest_child_age (ages : Fin 4 → ℕ) 
  (h_average : (ages 0 + ages 1 + ages 2 + ages 3) / 4 = 9)
  (h_younger : ages 0 = 6 ∧ ages 1 = 8 ∧ ages 2 = 10) :
  ages 3 = 12 := by
sorry

end NUMINAMATH_CALUDE_oldest_child_age_l2539_253939


namespace NUMINAMATH_CALUDE_difference_not_1998_l2539_253965

theorem difference_not_1998 (n m : ℕ) : (n^2 + 4*n) - (m^2 + 4*m) ≠ 1998 := by
  sorry

end NUMINAMATH_CALUDE_difference_not_1998_l2539_253965


namespace NUMINAMATH_CALUDE_equal_roots_implies_c_equals_one_fourth_l2539_253985

-- Define the quadratic equation
def quadratic_equation (x c : ℝ) : Prop := x^2 + x + c = 0

-- Define the condition for two equal real roots
def has_two_equal_real_roots (c : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation x c ∧ 
    ∀ y : ℝ, quadratic_equation y c → y = x

-- Theorem statement
theorem equal_roots_implies_c_equals_one_fourth :
  ∀ c : ℝ, has_two_equal_real_roots c → c = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_implies_c_equals_one_fourth_l2539_253985


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_minimal_l2539_253988

def triangle_side_1 : ℕ := 45
def triangle_side_2 : ℕ := 55
def triangle_side_3 : ℕ := 2 * triangle_side_1

def triangle_perimeter : ℕ := triangle_side_1 + triangle_side_2 + triangle_side_3

theorem triangle_perimeter_is_minimal : 
  triangle_perimeter = 190 ∧ 
  (∀ a b c : ℕ, a = triangle_side_1 → b = triangle_side_2 → c ≥ 2 * triangle_side_1 → 
   a + b > c ∧ a + c > b ∧ b + c > a → a + b + c ≥ triangle_perimeter) := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_minimal_l2539_253988


namespace NUMINAMATH_CALUDE_sum_of_squares_l2539_253906

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 0) (h2 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) :
  a^2 + b^2 + c^2 = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2539_253906


namespace NUMINAMATH_CALUDE_factorization_equality_l2539_253986

theorem factorization_equality (x y : ℝ) :
  3 * x^2 - x * y - y^2 = ((Real.sqrt 13 + 1) / 2 * x + y) * ((Real.sqrt 13 - 1) / 2 * x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2539_253986


namespace NUMINAMATH_CALUDE_rectangle_formations_with_restrictions_l2539_253948

/-- The number of ways to choose 4 lines to form a rectangle -/
def rectangleFormations (h v : ℕ) (hRestricted vRestricted : Fin 2 → ℕ) : ℕ :=
  let hChoices := (Nat.choose h 2) - 1
  let vChoices := (Nat.choose v 2) - 1
  hChoices * vChoices

/-- Theorem stating the number of ways to form a rectangle with given conditions -/
theorem rectangle_formations_with_restrictions :
  rectangleFormations 6 7 ![2, 5] ![3, 6] = 280 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formations_with_restrictions_l2539_253948


namespace NUMINAMATH_CALUDE_lao_you_fen_max_profit_l2539_253929

/-- Represents the cost and quantity information for Lao You Fen brands -/
structure LaoYouFen where
  cost_a : ℝ
  cost_b : ℝ
  quantity_a : ℝ
  quantity_b : ℝ

/-- Calculates the profit given the quantities of each brand -/
def profit (l : LaoYouFen) (qa qb : ℝ) : ℝ :=
  (13 - l.cost_a) * qa + (13 - l.cost_b) * qb

/-- Theorem stating the maximum profit for Lao You Fen sales -/
theorem lao_you_fen_max_profit (l : LaoYouFen) :
  l.cost_b = l.cost_a + 2 →
  2700 / l.cost_a = 3300 / l.cost_b →
  l.quantity_a + l.quantity_b = 800 →
  l.quantity_a ≤ 3 * l.quantity_b →
  (∀ qa qb : ℝ, qa + qb = 800 → qa ≤ 3 * qb → profit l qa qb ≤ 2800) ∧
  profit l 600 200 = 2800 :=
sorry

end NUMINAMATH_CALUDE_lao_you_fen_max_profit_l2539_253929


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l2539_253996

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 255) (h2 : Nat.gcd a c = 855) :
  ∃ (b' c' : ℕ+), Nat.gcd a b' = 255 ∧ Nat.gcd a c' = 855 ∧ 
    Nat.gcd b' c' = 15 ∧ 
    ∀ (b'' c'' : ℕ+), Nat.gcd a b'' = 255 → Nat.gcd a c'' = 855 → 
      Nat.gcd b'' c'' ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l2539_253996


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l2539_253924

/-- Proves that in a class with 35 students, where there are seven more girls than boys, 
    the ratio of girls to boys is 3:2 -/
theorem girls_to_boys_ratio (total : ℕ) (girls boys : ℕ) : 
  total = 35 →
  girls = boys + 7 →
  girls + boys = total →
  (girls : ℚ) / (boys : ℚ) = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l2539_253924


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l2539_253914

theorem quadratic_solution_property (a : ℝ) : 
  a^2 - 2*a - 1 = 0 → 2*a^2 - 4*a + 2022 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l2539_253914


namespace NUMINAMATH_CALUDE_weekly_wage_calculation_l2539_253942

def basic_daily_wage : ℕ := 200
def basic_task_quantity : ℕ := 40
def reward_per_excess : ℕ := 7
def deduction_per_incomplete : ℕ := 8
def work_days : ℕ := 5
def production_deviations : List ℤ := [5, -2, -1, 0, 4]

def total_weekly_wage : ℕ := 1039

theorem weekly_wage_calculation :
  (basic_daily_wage * work_days) +
  (production_deviations.filter (λ x => x > 0)).sum * reward_per_excess -
  (production_deviations.filter (λ x => x < 0)).sum.natAbs * deduction_per_incomplete =
  total_weekly_wage :=
sorry

end NUMINAMATH_CALUDE_weekly_wage_calculation_l2539_253942


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l2539_253936

-- Define the functions f and g
def f (A B C x : ℝ) : ℝ := A * x + B + C
def g (A B C x : ℝ) : ℝ := B * x + A - C

-- State the theorem
theorem sum_of_coefficients_is_zero 
  (A B C : ℝ) 
  (h1 : A ≠ B) 
  (h2 : C ≠ 0) 
  (h3 : ∀ x, f A B C (g A B C x) - g A B C (f A B C x) = 2 * C) : 
  A + B = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l2539_253936


namespace NUMINAMATH_CALUDE_max_a_for_integer_solution_l2539_253927

theorem max_a_for_integer_solution : 
  (∃ (a : ℕ+), ∀ (b : ℕ+), 
    (∃ (x : ℤ), x^2 + (b : ℤ) * x = -30) → 
    (b : ℤ) ≤ (a : ℤ)) ∧ 
  (∃ (x : ℤ), x^2 + 31 * x = -30) := by
  sorry

end NUMINAMATH_CALUDE_max_a_for_integer_solution_l2539_253927


namespace NUMINAMATH_CALUDE_ladder_length_l2539_253968

theorem ladder_length (angle : Real) (adjacent : Real) (hypotenuse : Real) : 
  angle = 60 * π / 180 →
  adjacent = 4.6 →
  Real.cos angle = adjacent / hypotenuse →
  hypotenuse = 9.2 := by
sorry

end NUMINAMATH_CALUDE_ladder_length_l2539_253968


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2539_253954

theorem min_value_sum_reciprocals (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) :
  1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d) ≥ 18 ∧
  (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d) = 18 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2539_253954


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l2539_253931

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := (a * 10 + b) / 99

/-- The fraction 0.overline{72} divided by 0.overline{27} is equal to 8/3 -/
theorem repeating_decimal_ratio : 
  (RepeatingDecimal 7 2) / (RepeatingDecimal 2 7) = 8 / 3 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l2539_253931


namespace NUMINAMATH_CALUDE_car_speed_time_relationship_l2539_253901

theorem car_speed_time_relationship 
  (distance : ℝ) 
  (speed_A time_A : ℝ) 
  (speed_B time_B : ℝ) 
  (h1 : distance > 0) 
  (h2 : speed_A > 0) 
  (h3 : speed_B = 3 * speed_A) 
  (h4 : distance = speed_A * time_A) 
  (h5 : distance = speed_B * time_B) : 
  time_B = time_A / 3 := by
sorry


end NUMINAMATH_CALUDE_car_speed_time_relationship_l2539_253901


namespace NUMINAMATH_CALUDE_correct_subtraction_l2539_253967

theorem correct_subtraction (x : ℤ) : x - 32 = 25 → x - 23 = 34 := by sorry

end NUMINAMATH_CALUDE_correct_subtraction_l2539_253967


namespace NUMINAMATH_CALUDE_prime_sum_probability_l2539_253912

def first_twelve_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_valid_pair (p q : Nat) : Bool :=
  p ∈ first_twelve_primes ∧ q ∈ first_twelve_primes ∧ p ≠ q ∧
  Nat.Prime (p + q) ∧ p + q > 20

def count_valid_pairs : Nat :=
  (List.filter (fun (pair : Nat × Nat) => is_valid_pair pair.1 pair.2)
    (List.product first_twelve_primes first_twelve_primes)).length

def total_pairs : Nat := (first_twelve_primes.length * (first_twelve_primes.length - 1)) / 2

theorem prime_sum_probability :
  count_valid_pairs / total_pairs = 1 / 66 := by sorry

end NUMINAMATH_CALUDE_prime_sum_probability_l2539_253912


namespace NUMINAMATH_CALUDE_quadratic_always_negative_l2539_253982

theorem quadratic_always_negative (m : ℝ) :
  (∀ x : ℝ, -x^2 + (2*m + 6)*x - m - 3 < 0) ↔ -3 < m ∧ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_always_negative_l2539_253982


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2539_253905

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 6*m^2 + 5*m = 27*n^3 + 27*n^2 + 9*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2539_253905


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2539_253975

/-- Given a parallelogram with area 108 cm² and height 9 cm, its base length is 12 cm. -/
theorem parallelogram_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 108 ∧ height = 9 ∧ area = base * height → base = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2539_253975


namespace NUMINAMATH_CALUDE_base6_to_base10_fraction_l2539_253920

/-- Converts a base-6 number to base-10 --/
def base6To10 (n : ℕ) : ℕ := sorry

/-- Determines if a natural number is a valid 3-digit base-10 number --/
def isValidBase10 (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- Extracts the hundreds digit from a 3-digit base-10 number --/
def hundredsDigit (n : ℕ) : ℕ := n / 100

/-- Extracts the ones digit from a 3-digit base-10 number --/
def onesDigit (n : ℕ) : ℕ := n % 10

theorem base6_to_base10_fraction (c d e : ℕ) :
  base6To10 532 = 100 * c + 10 * d + e →
  isValidBase10 (100 * c + 10 * d + e) →
  (c * e : ℚ) / 10 = 0 := by sorry

end NUMINAMATH_CALUDE_base6_to_base10_fraction_l2539_253920


namespace NUMINAMATH_CALUDE_solve_equation_l2539_253900

-- Define y as a constant real number
variable (y : ℝ)

-- Define the theorem
theorem solve_equation (x : ℝ) (h : Real.sqrt (x + y - 3) = 10) : x = 103 - y := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2539_253900


namespace NUMINAMATH_CALUDE_circle_equation_l2539_253984

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 = c.radius^2

def center_on_line (c : Circle) : Prop :=
  c.center.2 = 2 * c.center.1

def tangent_to_line (c : Circle) : Prop :=
  c.radius = |2 * c.center.1 - c.center.2 + 5| / Real.sqrt 5

-- Theorem statement
theorem circle_equation (c : Circle) :
  passes_through c (3, 2) ∧
  center_on_line c ∧
  tangent_to_line c →
  ((λ (x y : ℝ) => (x - 2)^2 + (y - 4)^2 = 5) c.center.1 c.center.2) ∨
  ((λ (x y : ℝ) => (x - 4/5)^2 + (y - 8/5)^2 = 5) c.center.1 c.center.2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2539_253984


namespace NUMINAMATH_CALUDE_line_properties_l2539_253950

/-- A line in the xy-plane represented by the equation x = ky + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Predicate for a line being perpendicular to the y-axis -/
def perpendicular_to_y_axis (l : Line) : Prop :=
  ∃ (x : ℝ), ∀ (y : ℝ), x = l.k * y + l.b

/-- Predicate for a line being perpendicular to the x-axis -/
def perpendicular_to_x_axis (l : Line) : Prop :=
  ∀ (y : ℝ), l.k * y + l.b = l.b

theorem line_properties :
  (¬ ∃ (l : Line), perpendicular_to_y_axis l) ∧
  (∃ (l : Line), perpendicular_to_x_axis l) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l2539_253950


namespace NUMINAMATH_CALUDE_consecutive_numbers_problem_l2539_253980

theorem consecutive_numbers_problem (x y z : ℤ) : 
  (y = z + 1) →  -- x, y, and z are consecutive
  (x = y + 1) →  -- x, y, and z are consecutive
  (x > y) →      -- x > y > z
  (y > z) →      -- x > y > z
  (2*x + 3*y + 3*z = 5*y + 11) →  -- given equation
  (z = 3) →      -- given value of z
  y = 4 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_problem_l2539_253980


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2539_253992

theorem x_squared_minus_y_squared (x y : ℝ) 
  (sum : x + y = 20) 
  (diff : x - y = 4) : 
  x^2 - y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2539_253992


namespace NUMINAMATH_CALUDE_odd_function_property_l2539_253922

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h_odd : IsOdd f) (h_sum : ∀ x, f (x + 1) + f x = 0) :
  f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l2539_253922


namespace NUMINAMATH_CALUDE_draw_four_from_fifteen_l2539_253946

/-- The number of balls in the bin -/
def n : ℕ := 15

/-- The number of balls to be drawn -/
def k : ℕ := 4

/-- The number of ways to draw k balls from n balls in order, without replacement -/
def drawWithoutReplacement (n k : ℕ) : ℕ :=
  (n - k + 1).factorial / (n - k).factorial

theorem draw_four_from_fifteen :
  drawWithoutReplacement n k = 32760 := by
  sorry

end NUMINAMATH_CALUDE_draw_four_from_fifteen_l2539_253946


namespace NUMINAMATH_CALUDE_smo_board_sum_l2539_253953

/-- Represents the state of the board at any given step -/
structure BoardState where
  numbers : List Nat

/-- Represents a single step in the process -/
def step (state : BoardState) : BoardState :=
  sorry

/-- The sum of all numbers on the board -/
def board_sum (state : BoardState) : Nat :=
  state.numbers.sum

theorem smo_board_sum (m : Nat) : 
  ∀ (final_state : BoardState),
    (∃ (initial_state : BoardState),
      initial_state.numbers = List.replicate (2^m) 1 ∧
      final_state = (step^[m * 2^(m-1)]) initial_state) →
    board_sum final_state ≥ 4^m :=
  sorry

end NUMINAMATH_CALUDE_smo_board_sum_l2539_253953


namespace NUMINAMATH_CALUDE_invisible_dots_sum_l2539_253913

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The number of dice -/
def num_dice : ℕ := 3

/-- The sum of visible numbers -/
def visible_sum : ℕ := 1 + 2 + 3 + 3 + 4

/-- The number of visible faces -/
def num_visible_faces : ℕ := 5

theorem invisible_dots_sum : 
  num_dice * die_sum - visible_sum = 50 := by sorry

end NUMINAMATH_CALUDE_invisible_dots_sum_l2539_253913


namespace NUMINAMATH_CALUDE_four_part_cut_possible_five_triangular_part_cut_possible_l2539_253928

-- Define the original figure
def original_figure : Set (ℝ × ℝ) :=
  sorry

-- Define the area of the original figure
def original_area : ℝ := 64

-- Define a square with area 64
def target_square : Set (ℝ × ℝ) :=
  sorry

-- Define a function that represents cutting the figure into parts
def cut (figure : Set (ℝ × ℝ)) (n : ℕ) : List (Set (ℝ × ℝ)) :=
  sorry

-- Define a function that represents assembling parts into a new figure
def assemble (parts : List (Set (ℝ × ℝ))) : Set (ℝ × ℝ) :=
  sorry

-- Define a predicate to check if a set is triangular
def is_triangular (s : Set (ℝ × ℝ)) : Prop :=
  sorry

-- Theorem for part a
theorem four_part_cut_possible :
  ∃ (parts : List (Set (ℝ × ℝ))),
    parts.length ≤ 4 ∧
    (∀ p ∈ parts, p ⊆ original_figure) ∧
    assemble parts = target_square :=
  sorry

-- Theorem for part b
theorem five_triangular_part_cut_possible :
  ∃ (parts : List (Set (ℝ × ℝ))),
    parts.length ≤ 5 ∧
    (∀ p ∈ parts, p ⊆ original_figure ∧ is_triangular p) ∧
    assemble parts = target_square :=
  sorry

end NUMINAMATH_CALUDE_four_part_cut_possible_five_triangular_part_cut_possible_l2539_253928


namespace NUMINAMATH_CALUDE_quadratic_solution_product_l2539_253958

theorem quadratic_solution_product (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) → 
  (3 * e^2 + 4 * e - 7 = 0) → 
  (d + 1) * (e + 1) = -8/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_product_l2539_253958


namespace NUMINAMATH_CALUDE_time_to_fill_tank_with_hole_l2539_253934

/-- Time to fill tank with hole present -/
theorem time_to_fill_tank_with_hole 
  (pipe_fill_time : ℝ) 
  (hole_empty_time : ℝ) 
  (h1 : pipe_fill_time = 15) 
  (h2 : hole_empty_time = 60.000000000000014) : 
  (1 : ℝ) / ((1 / pipe_fill_time) - (1 / hole_empty_time)) = 20.000000000000001 := by
  sorry

end NUMINAMATH_CALUDE_time_to_fill_tank_with_hole_l2539_253934


namespace NUMINAMATH_CALUDE_logarithm_properties_l2539_253995

-- Define the theorem
theorem logarithm_properties (a b m n : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hm1 : m ≠ 1) (hn : 0 < n) : 
  (Real.log a / Real.log b) * (Real.log b / Real.log a) = 1 ∧ 
  (Real.log n / Real.log a) / (Real.log n / Real.log (m * a)) = 1 + (Real.log m / Real.log a) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_properties_l2539_253995


namespace NUMINAMATH_CALUDE_smallest_cube_ending_144_l2539_253925

theorem smallest_cube_ending_144 : ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 144 ∧ ∀ (m : ℕ), m > 0 → m^3 % 1000 = 144 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_144_l2539_253925


namespace NUMINAMATH_CALUDE_profit_loss_ratio_l2539_253917

theorem profit_loss_ratio (c x y : ℝ) (hx : x = 0.85 * c) (hy : y = 1.15 * c) :
  y / x = 23 / 17 := by
  sorry

end NUMINAMATH_CALUDE_profit_loss_ratio_l2539_253917


namespace NUMINAMATH_CALUDE_average_study_time_difference_l2539_253961

/-- The daily differences in study time (in minutes) between Mira and Clara over a week -/
def study_time_differences : List Int := [15, 0, -15, 25, 5, -5, 10]

/-- The number of days in the week -/
def days_in_week : Nat := 7

/-- Theorem stating that the average difference in daily study time is 5 minutes -/
theorem average_study_time_difference :
  (study_time_differences.sum : ℚ) / days_in_week = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l2539_253961


namespace NUMINAMATH_CALUDE_river_depth_l2539_253930

/-- Proves that given a river with specified width, flow rate, and discharge, its depth is 2 meters -/
theorem river_depth (width : ℝ) (flow_rate : ℝ) (discharge : ℝ) : 
  width = 45 ∧ 
  flow_rate = 6 ∧ 
  discharge = 9000 → 
  discharge = width * 2 * (flow_rate * 1000 / 60) := by
  sorry

#check river_depth

end NUMINAMATH_CALUDE_river_depth_l2539_253930


namespace NUMINAMATH_CALUDE_like_terms_mn_value_l2539_253943

/-- 
Given two algebraic terms are like terms, prove that m^n = 8.
-/
theorem like_terms_mn_value (n m : ℕ) : 
  (∃ (k : ℚ), k * X^n * Y^2 = X^3 * Y^m) → m^n = 8 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_mn_value_l2539_253943


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2539_253919

theorem fraction_sum_equality : (20 : ℚ) / 50 - 3 / 8 + 1 / 4 = 11 / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2539_253919


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2539_253974

theorem negation_of_proposition (p : Prop) :
  (¬ (∃ m : ℝ, (m^2 + m - 6)⁻¹ > 0)) ↔ 
  (∀ m : ℝ, (m^2 + m - 6)⁻¹ < 0 ∨ m^2 + m - 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2539_253974


namespace NUMINAMATH_CALUDE_basket_weight_l2539_253956

/-- Given a basket of persimmons, prove the weight of the empty basket. -/
theorem basket_weight (total_weight half_weight : ℝ) 
  (h1 : total_weight = 62)
  (h2 : half_weight = 34) : 
  ∃ (basket_weight persimmons_weight : ℝ),
    basket_weight + persimmons_weight = total_weight ∧ 
    basket_weight + persimmons_weight / 2 = half_weight ∧
    basket_weight = 6 := by
  sorry

end NUMINAMATH_CALUDE_basket_weight_l2539_253956


namespace NUMINAMATH_CALUDE_log_27_3_l2539_253944

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_27_3_l2539_253944


namespace NUMINAMATH_CALUDE_absolute_value_problem_l2539_253969

theorem absolute_value_problem (y q : ℝ) (h1 : |y - 3| = q) (h2 : y < 3) : 
  y - 2*q = 3 - 3*q := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l2539_253969


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2539_253902

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (6 - i) / (1 + i) = Complex.mk (5/2) (-7/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2539_253902


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l2539_253935

/-- Represents the volume multiplication factor of a cylinder when its height is tripled and radius is increased by 300% -/
def cylinder_volume_factor : ℝ := 48

/-- Theorem stating that when a cylinder's height is tripled and its radius is increased by 300%, its volume is multiplied by a factor of 48 -/
theorem cylinder_volume_increase (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  let new_r := 4 * r
  let new_h := 3 * h
  (π * new_r^2 * new_h) / (π * r^2 * h) = cylinder_volume_factor :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l2539_253935


namespace NUMINAMATH_CALUDE_base5_23104_equals_1654_l2539_253949

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (d₄ d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₄ * 5^4 + d₃ * 5^3 + d₂ * 5^2 + d₁ * 5^1 + d₀ * 5^0

/-- The base 5 number 23104 is equal to 1654 in base 10 --/
theorem base5_23104_equals_1654 :
  base5ToBase10 2 3 1 0 4 = 1654 := by
  sorry

end NUMINAMATH_CALUDE_base5_23104_equals_1654_l2539_253949
