import Mathlib

namespace NUMINAMATH_CALUDE_micah_fish_count_l3727_372785

/-- Proves that Micah has 7 fish given the problem conditions -/
theorem micah_fish_count :
  ∀ (m k t : ℕ),
  k = 3 * m →                -- Kenneth has three times as many fish as Micah
  t = k - 15 →                -- Matthias has 15 less fish than Kenneth
  m + k + t = 34 →            -- The total number of fish for all three boys is 34
  m = 7 :=                    -- Micah has 7 fish
by
  sorry

end NUMINAMATH_CALUDE_micah_fish_count_l3727_372785


namespace NUMINAMATH_CALUDE_race_time_theorem_l3727_372799

/-- The time taken by this year's winner to complete the race around the town square. -/
def this_year_time (laps : ℕ) (square_length : ℚ) (last_year_time : ℚ) (time_improvement : ℚ) : ℚ :=
  let total_distance := laps * square_length
  let last_year_pace := last_year_time / total_distance
  let this_year_pace := last_year_pace - time_improvement
  this_year_pace * total_distance

/-- Theorem stating that this year's winner completed the race in 42 minutes. -/
theorem race_time_theorem :
  this_year_time 7 (3/4) 47.25 1 = 42 := by
  sorry

end NUMINAMATH_CALUDE_race_time_theorem_l3727_372799


namespace NUMINAMATH_CALUDE_complement_of_union_M_N_l3727_372755

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Finset Nat := {2, 3, 4}

-- Define set N
def N : Finset Nat := {4, 5}

-- Theorem statement
theorem complement_of_union_M_N :
  (U \ (M ∪ N)) = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_M_N_l3727_372755


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3727_372774

def i : ℂ := Complex.I

def z : ℂ := i * (1 + i)

theorem z_in_second_quadrant :
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3727_372774


namespace NUMINAMATH_CALUDE_first_number_proof_l3727_372745

theorem first_number_proof (x : ℝ) : x + 33 + 333 + 3.33 = 369.63 → x = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l3727_372745


namespace NUMINAMATH_CALUDE_max_value_a_l3727_372794

open Real

theorem max_value_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, a ≤ (1-x)/x + log x) → 
  a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_max_value_a_l3727_372794


namespace NUMINAMATH_CALUDE_permutation_reachable_l3727_372762

/-- A transformation step on a tuple of natural numbers -/
def transform (a : Fin 2015 → ℕ) (k l : Fin 2015) (h : Even (a k)) : Fin 2015 → ℕ :=
  fun i => if i = k then a k / 2
           else if i = l then a l + a k / 2
           else a i

/-- The set of all permutations of (1, 2, ..., 2015) -/
def permutations : Set (Fin 2015 → ℕ) :=
  {p | ∃ σ : Equiv.Perm (Fin 2015), ∀ i, p i = σ i + 1}

/-- The initial tuple (1, 2, ..., 2015) -/
def initial : Fin 2015 → ℕ := fun i => i + 1

/-- The set of all tuples reachable from the initial tuple -/
inductive reachable : (Fin 2015 → ℕ) → Prop
  | init : reachable initial
  | step {a b} (k l : Fin 2015) (h : Even (a k)) :
      reachable a → b = transform a k l h → reachable b

theorem permutation_reachable :
  ∀ p ∈ permutations, reachable p :=
sorry

end NUMINAMATH_CALUDE_permutation_reachable_l3727_372762


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3727_372720

theorem sphere_surface_area (r : ℝ) (h : r = 3) : 4 * Real.pi * r^2 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3727_372720


namespace NUMINAMATH_CALUDE_stock_investment_l3727_372739

theorem stock_investment (annual_income : ℝ) (stock_percentage : ℝ) (stock_price : ℝ) :
  annual_income = 2000 ∧ 
  stock_percentage = 40 ∧ 
  stock_price = 136 →
  ∃ amount_invested : ℝ, amount_invested = 6800 ∧
    annual_income = (amount_invested / stock_price) * (stock_percentage / 100) * 100 :=
by sorry

end NUMINAMATH_CALUDE_stock_investment_l3727_372739


namespace NUMINAMATH_CALUDE_sum_of_digits_1_to_9999_l3727_372766

/-- Sum of digits for numbers from 1 to n -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits for numbers from 1 to 4999 -/
def sumTo4999 : ℕ := sumOfDigits 4999

/-- Sum of digits for numbers from 5000 to 9999, considering mirroring and additional 5 -/
def sum5000To9999 : ℕ := sumTo4999 + 5000 * 5

/-- The total sum of digits for all numbers from 1 to 9999 -/
def totalSum : ℕ := sumTo4999 + sum5000To9999

theorem sum_of_digits_1_to_9999 : totalSum = 474090 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_1_to_9999_l3727_372766


namespace NUMINAMATH_CALUDE_complement_and_union_when_m_3_subset_condition_disjoint_condition_l3727_372722

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m ≤ x ∧ x ≤ m + 2}

-- Theorem 1
theorem complement_and_union_when_m_3 :
  (Set.univ \ B 3) = {x : ℝ | x < 3 ∨ x > 5} ∧
  A ∪ B 3 = {x : ℝ | 0 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem subset_condition :
  ∀ m : ℝ, B m ⊆ A ↔ 0 ≤ m ∧ m ≤ 2 := by sorry

-- Theorem 3
theorem disjoint_condition :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m < -2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_complement_and_union_when_m_3_subset_condition_disjoint_condition_l3727_372722


namespace NUMINAMATH_CALUDE_simplify_expression_l3727_372711

theorem simplify_expression (x : ℝ) : 2*x^3 - (7*x^2 - 9*x) - 2*(x^3 - 3*x^2 + 4*x) = -x^2 + x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3727_372711


namespace NUMINAMATH_CALUDE_tom_distance_covered_l3727_372778

theorem tom_distance_covered (swim_time : ℝ) (swim_speed : ℝ) : 
  swim_time = 2 →
  swim_speed = 2 →
  let run_time := swim_time / 2
  let run_speed := 4 * swim_speed
  let swim_distance := swim_time * swim_speed
  let run_distance := run_time * run_speed
  swim_distance + run_distance = 12 := by
  sorry

end NUMINAMATH_CALUDE_tom_distance_covered_l3727_372778


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3727_372705

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem seventh_term_of_geometric_sequence :
  let a₁ : ℚ := 3
  let a₂ : ℚ := -1/2
  let r : ℚ := a₂ / a₁
  geometric_sequence a₁ r 7 = 1/15552 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3727_372705


namespace NUMINAMATH_CALUDE_product_equality_l3727_372706

theorem product_equality : 100 * 19.98 * 1.998 * 999 = 3988008 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3727_372706


namespace NUMINAMATH_CALUDE_basic_astrophysics_degrees_l3727_372795

def microphotonics : ℝ := 14
def home_electronics : ℝ := 24
def food_additives : ℝ := 10
def genetically_modified_microorganisms : ℝ := 29
def industrial_lubricants : ℝ := 8
def total_circle_degrees : ℝ := 360

def other_sectors_total : ℝ := microphotonics + home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants

def basic_astrophysics_percentage : ℝ := 100 - other_sectors_total

theorem basic_astrophysics_degrees : 
  (basic_astrophysics_percentage / 100) * total_circle_degrees = 54 := by
  sorry

end NUMINAMATH_CALUDE_basic_astrophysics_degrees_l3727_372795


namespace NUMINAMATH_CALUDE_amy_baskets_l3727_372735

/-- The number of baskets Amy can fill with candies -/
def num_baskets (chocolate_bars : ℕ) (m_and_ms_ratio : ℕ) (marshmallow_ratio : ℕ) (candies_per_basket : ℕ) : ℕ :=
  let m_and_ms := chocolate_bars * m_and_ms_ratio
  let marshmallows := m_and_ms * marshmallow_ratio
  let total_candies := chocolate_bars + m_and_ms + marshmallows
  total_candies / candies_per_basket

/-- Theorem stating that Amy will fill 25 baskets given the conditions -/
theorem amy_baskets :
  num_baskets 5 7 6 10 = 25 := by
  sorry

end NUMINAMATH_CALUDE_amy_baskets_l3727_372735


namespace NUMINAMATH_CALUDE_quadratic_equation_nonnegative_integer_solutions_l3727_372756

theorem quadratic_equation_nonnegative_integer_solutions :
  ∃! (x : ℕ), x^2 + x - 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_nonnegative_integer_solutions_l3727_372756


namespace NUMINAMATH_CALUDE_polynomial_divisibility_sum_A_B_l3727_372721

-- Define the polynomial
def p (A B : ℂ) (x : ℂ) : ℂ := x^103 + A*x + B

-- Define the divisor polynomial
def d (x : ℂ) : ℂ := x^2 + x + 1

-- State the theorem
theorem polynomial_divisibility (A B : ℂ) :
  (∀ x, d x = 0 → p A B x = 0) →
  A = -1 ∧ B = 0 := by
  sorry

-- Corollary for A + B
theorem sum_A_B (A B : ℂ) :
  (∀ x, d x = 0 → p A B x = 0) →
  A + B = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_sum_A_B_l3727_372721


namespace NUMINAMATH_CALUDE_hubei_population_scientific_notation_l3727_372777

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The population of Hubei Province -/
def hubei_population : ℕ := 57000000

/-- Scientific notation for Hubei population -/
def hubei_scientific : ScientificNotation :=
  { coefficient := 5.7
  , exponent := 7
  , h1 := by sorry }

/-- Theorem stating that the scientific notation correctly represents the population -/
theorem hubei_population_scientific_notation :
  (hubei_scientific.coefficient * (10 : ℝ) ^ hubei_scientific.exponent) = hubei_population := by
  sorry

end NUMINAMATH_CALUDE_hubei_population_scientific_notation_l3727_372777


namespace NUMINAMATH_CALUDE_original_number_exists_and_unique_l3727_372729

theorem original_number_exists_and_unique :
  ∃! x : ℕ, 
    Odd (3 * x) ∧ 
    (∃ k : ℕ, 3 * x = 9 * k) ∧ 
    4 * x = 108 := by
  sorry

end NUMINAMATH_CALUDE_original_number_exists_and_unique_l3727_372729


namespace NUMINAMATH_CALUDE_problem_curve_is_ray_l3727_372789

/-- A curve defined by parametric equations -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Definition of a ray -/
def IsRay (c : ParametricCurve) : Prop :=
  ∃ (a b m : ℝ), ∀ t : ℝ, 
    c.x t = m * (c.y t) + b ∧ 
    c.x t ≥ a ∧ 
    c.y t ≥ -1

/-- The specific curve from the problem -/
def problemCurve : ParametricCurve :=
  { x := λ t : ℝ => 3 * t^2 + 2
    y := λ t : ℝ => t^2 - 1 }

/-- Theorem stating that the problem curve is a ray -/
theorem problem_curve_is_ray : IsRay problemCurve := by
  sorry


end NUMINAMATH_CALUDE_problem_curve_is_ray_l3727_372789


namespace NUMINAMATH_CALUDE_netflix_series_seasons_l3727_372750

theorem netflix_series_seasons (episodes_per_season : ℕ) (episodes_remaining : ℕ) : 
  episodes_per_season = 20 →
  episodes_remaining = 160 →
  (∃ (total_episodes : ℕ), 
    total_episodes * (1 / 3 : ℚ) = total_episodes - episodes_remaining ∧
    total_episodes / episodes_per_season = 12) :=
by sorry

end NUMINAMATH_CALUDE_netflix_series_seasons_l3727_372750


namespace NUMINAMATH_CALUDE_debate_club_next_meeting_l3727_372767

theorem debate_club_next_meeting (anthony bethany casey dana : ℕ) 
  (h1 : anthony = 5)
  (h2 : bethany = 6)
  (h3 : casey = 8)
  (h4 : dana = 10) :
  Nat.lcm (Nat.lcm (Nat.lcm anthony bethany) casey) dana = 120 := by
  sorry

end NUMINAMATH_CALUDE_debate_club_next_meeting_l3727_372767


namespace NUMINAMATH_CALUDE_cubic_inequality_l3727_372770

theorem cubic_inequality (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  a^3 + b^3 ≤ a*b^2 + a^2*b := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3727_372770


namespace NUMINAMATH_CALUDE_not_sum_of_three_squares_2015_l3727_372773

theorem not_sum_of_three_squares_2015 : ¬∃ (a b c : ℤ), a^2 + b^2 + c^2 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_not_sum_of_three_squares_2015_l3727_372773


namespace NUMINAMATH_CALUDE_angle_XZY_is_50_l3727_372759

/-- Given a diagram where AB and CD are straight lines -/
structure Diagram where
  /-- The angle AXB is 180 degrees -/
  angle_AXB : ℝ
  /-- The angle CYX is 120 degrees -/
  angle_CYX : ℝ
  /-- The angle YXB is 60 degrees -/
  angle_YXB : ℝ
  /-- The angle AXY is 50 degrees -/
  angle_AXY : ℝ
  /-- AB is a straight line -/
  h_AB_straight : angle_AXB = 180
  /-- CD is a straight line (not directly used but implied) -/
  h_CYX : angle_CYX = 120
  h_YXB : angle_YXB = 60
  h_AXY : angle_AXY = 50

/-- The theorem stating that the angle XZY is 50 degrees -/
theorem angle_XZY_is_50 (d : Diagram) : ∃ x, x = 50 ∧ x = d.angle_AXB - d.angle_CYX + d.angle_YXB - d.angle_AXY :=
  sorry

end NUMINAMATH_CALUDE_angle_XZY_is_50_l3727_372759


namespace NUMINAMATH_CALUDE_fraction_doubling_l3727_372716

theorem fraction_doubling (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1.4 * x) / (0.7 * y) = 2 * (x / y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_doubling_l3727_372716


namespace NUMINAMATH_CALUDE_solve_complex_equation_l3727_372741

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := 5 + 2 * i * z = 4 - 6 * i * z

-- State the theorem
theorem solve_complex_equation :
  ∃ z : ℂ, equation z ∧ z = -1/8 * i :=
sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l3727_372741


namespace NUMINAMATH_CALUDE_f_symmetry_l3727_372792

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem f_symmetry (a : ℝ) : f a - f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l3727_372792


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3727_372751

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, x^2 > 2012 → x^2 > 2011) ∧
  (∃ x : ℝ, x^2 > 2011 ∧ x^2 ≤ 2012) →
  (∀ x : ℝ, x^2 > 2012 → x^2 > 2011) ∧
  ¬(∀ x : ℝ, x^2 > 2011 → x^2 > 2012) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3727_372751


namespace NUMINAMATH_CALUDE_sum_in_base7_l3727_372731

/-- Converts a base 7 number to base 10 --/
def base7_to_base10 (x : ℕ) : ℕ :=
  (x / 10) * 7 + (x % 10)

/-- Converts a base 10 number to base 7 --/
def base10_to_base7 (x : ℕ) : ℕ :=
  if x < 7 then x
  else (base10_to_base7 (x / 7)) * 10 + (x % 7)

theorem sum_in_base7 :
  base10_to_base7 (base7_to_base10 15 + base7_to_base10 26) = 44 :=
by sorry

end NUMINAMATH_CALUDE_sum_in_base7_l3727_372731


namespace NUMINAMATH_CALUDE_mary_cut_ten_roses_l3727_372791

/-- The number of roses Mary cut from her flower garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Proof that Mary cut 10 roses from her flower garden -/
theorem mary_cut_ten_roses (initial_roses final_roses : ℕ) 
  (h1 : initial_roses = 6) 
  (h2 : final_roses = 16) : 
  roses_cut initial_roses final_roses = 10 := by
  sorry

#check mary_cut_ten_roses

end NUMINAMATH_CALUDE_mary_cut_ten_roses_l3727_372791


namespace NUMINAMATH_CALUDE_largest_square_with_three_lattice_points_l3727_372728

/-- A lattice point in a 2D plane. -/
def LatticePoint (p : ℝ × ℝ) : Prop := Int.floor p.1 = p.1 ∧ Int.floor p.2 = p.2

/-- A square in a 2D plane. -/
structure Square where
  center : ℝ × ℝ
  sideLength : ℝ
  rotation : ℝ  -- Angle of rotation in radians

/-- Predicate to check if a point is in the interior of a square. -/
def IsInteriorPoint (s : Square) (p : ℝ × ℝ) : Prop := sorry

/-- The number of lattice points in the interior of a square. -/
def InteriorLatticePointCount (s : Square) : ℕ := sorry

/-- Theorem stating that the area of the largest square containing exactly three lattice points in its interior is 5. -/
theorem largest_square_with_three_lattice_points :
  ∃ (s : Square), InteriorLatticePointCount s = 3 ∧
    ∀ (s' : Square), InteriorLatticePointCount s' = 3 → s'.sideLength^2 ≤ s.sideLength^2 ∧
    s.sideLength^2 = 5 := by sorry

end NUMINAMATH_CALUDE_largest_square_with_three_lattice_points_l3727_372728


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l3727_372775

/-- 
Given a man who is 20 years older than his son, and the son's present age is 18,
prove that the ratio of the man's age to his son's age in two years will be 2:1.
-/
theorem man_son_age_ratio : 
  ∀ (son_age man_age : ℕ),
  son_age = 18 →
  man_age = son_age + 20 →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l3727_372775


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l3727_372757

/-- Proves that given the specified blanket purchases and average price, 
    the unknown rate for two blankets must be 275. -/
theorem unknown_blanket_rate 
  (price1 : ℕ) (count1 : ℕ) 
  (price2 : ℕ) (count2 : ℕ) 
  (count3 : ℕ) 
  (avg_price : ℕ) 
  (h1 : price1 = 100) 
  (h2 : count1 = 3) 
  (h3 : price2 = 150) 
  (h4 : count2 = 5) 
  (h5 : count3 = 2) 
  (h6 : avg_price = 160) 
  (h7 : (price1 * count1 + price2 * count2 + count3 * unknown_rate) / (count1 + count2 + count3) = avg_price) : 
  unknown_rate = 275 := by
  sorry

#check unknown_blanket_rate

end NUMINAMATH_CALUDE_unknown_blanket_rate_l3727_372757


namespace NUMINAMATH_CALUDE_partnership_profit_l3727_372782

/-- Calculates the total profit of a partnership given investments and one partner's share -/
def calculate_total_profit (investment_a investment_b investment_c c_share : ℕ) : ℕ :=
  let total_parts := investment_a / investment_c + investment_b / investment_c + 1
  total_parts * c_share

/-- Proves that given the investments and C's share, the total profit is 252000 -/
theorem partnership_profit (investment_a investment_b investment_c c_share : ℕ) 
  (h1 : investment_a = 8000)
  (h2 : investment_b = 4000)
  (h3 : investment_c = 2000)
  (h4 : c_share = 36000) :
  calculate_total_profit investment_a investment_b investment_c c_share = 252000 := by
  sorry

#eval calculate_total_profit 8000 4000 2000 36000

end NUMINAMATH_CALUDE_partnership_profit_l3727_372782


namespace NUMINAMATH_CALUDE_original_price_of_discounted_dress_l3727_372763

/-- Proves that given a 30% discount on a dress that results in a final price of $35, the original price of the dress was $50. -/
theorem original_price_of_discounted_dress (discount_percentage : ℝ) (final_price : ℝ) : 
  discount_percentage = 30 →
  final_price = 35 →
  (1 - discount_percentage / 100) * 50 = final_price :=
by sorry

end NUMINAMATH_CALUDE_original_price_of_discounted_dress_l3727_372763


namespace NUMINAMATH_CALUDE_total_lives_in_game_game_lives_proof_l3727_372754

theorem total_lives_in_game (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) : ℕ :=
  (initial_players + additional_players) * lives_per_player

theorem game_lives_proof :
  total_lives_in_game 4 5 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_in_game_game_lives_proof_l3727_372754


namespace NUMINAMATH_CALUDE_alien_mineral_collection_l3727_372717

/-- Converts a base-6 number represented as a list of digits to its base-10 equivalent -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The base-6 representation of the number -/
def alienCollection : List Nat := [5, 3, 2]

theorem alien_mineral_collection :
  base6ToBase10 alienCollection = 95 := by
  sorry

end NUMINAMATH_CALUDE_alien_mineral_collection_l3727_372717


namespace NUMINAMATH_CALUDE_remove_parentheses_l3727_372703

theorem remove_parentheses (a : ℝ) : -(2*a - 1) = -2*a + 1 := by
  sorry

end NUMINAMATH_CALUDE_remove_parentheses_l3727_372703


namespace NUMINAMATH_CALUDE_log_equation_solution_l3727_372771

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 5 → x = 3^(10/3) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3727_372771


namespace NUMINAMATH_CALUDE_trillion_scientific_notation_l3727_372732

theorem trillion_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), a = 1 ∧ n = 12 ∧ 1000000000000 = a * (10 : ℝ) ^ n :=
by
  sorry

end NUMINAMATH_CALUDE_trillion_scientific_notation_l3727_372732


namespace NUMINAMATH_CALUDE_number_of_sides_interior_angle_measure_l3727_372714

/-- 
A regular polygon where the sum of interior angles is 4 times the sum of exterior angles.
-/
structure RegularPolygon where
  n : ℕ  -- number of sides
  sum_interior_angles : ℝ
  sum_exterior_angles : ℝ
  h1 : sum_interior_angles = (n - 2) * 180
  h2 : sum_exterior_angles = 360
  h3 : sum_interior_angles = 4 * sum_exterior_angles

/-- The number of sides of the regular polygon is 10. -/
theorem number_of_sides (p : RegularPolygon) : p.n = 10 := by
  sorry

/-- The measure of each interior angle of the regular polygon is 144°. -/
theorem interior_angle_measure (p : RegularPolygon) : 
  (p.n - 2) * 180 / p.n = 144 := by
  sorry

end NUMINAMATH_CALUDE_number_of_sides_interior_angle_measure_l3727_372714


namespace NUMINAMATH_CALUDE_curvilinear_trapezoid_area_l3727_372780

-- Define the bounds of the trapezoid
def lower_bound : Real := -1
def upper_bound : Real := 2

-- Define the parabola function
def f (x : Real) : Real := 9 - x^2

-- State the theorem
theorem curvilinear_trapezoid_area :
  ∫ x in lower_bound..upper_bound, f x = 24 := by
  sorry

end NUMINAMATH_CALUDE_curvilinear_trapezoid_area_l3727_372780


namespace NUMINAMATH_CALUDE_jonessas_take_home_pay_l3727_372798

/-- Calculates the take-home pay given the total pay and tax rate -/
def takeHomePay (totalPay : ℝ) (taxRate : ℝ) : ℝ :=
  totalPay * (1 - taxRate)

/-- Proves that Jonessa's take-home pay is $450 -/
theorem jonessas_take_home_pay :
  let totalPay : ℝ := 500
  let taxRate : ℝ := 0.1
  takeHomePay totalPay taxRate = 450 := by sorry

end NUMINAMATH_CALUDE_jonessas_take_home_pay_l3727_372798


namespace NUMINAMATH_CALUDE_triangle_angle_c_l3727_372761

theorem triangle_angle_c (A B C : Real) (a b c : Real) :
  -- Conditions
  a + b + c = Real.sqrt 2 + 1 →  -- Perimeter condition
  (1/2) * a * b * Real.sin C = (1/6) * Real.sin C →  -- Area condition
  Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C →  -- Sine relation
  -- Conclusion
  C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l3727_372761


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l3727_372797

theorem geometric_progression_fourth_term 
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) 
  (h1 : a = Real.sqrt 2) 
  (h2 : b = (2 : ℝ) ^ (1/4))
  (h3 : c = (2 : ℝ) ^ (1/6))
  (h4 : b/a = c/b) -- geometric progression condition
  : d = (2 : ℝ) ^ (1/8) := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l3727_372797


namespace NUMINAMATH_CALUDE_subtract_3a_from_expression_l3727_372712

variable (a : ℝ)

theorem subtract_3a_from_expression : (9 * a^2 - 3 * a + 8) - 3 * a = 9 * a^2 + 8 := by
  sorry

end NUMINAMATH_CALUDE_subtract_3a_from_expression_l3727_372712


namespace NUMINAMATH_CALUDE_prism_faces_l3727_372749

/-- Represents a prism with n-sided polygonal bases -/
structure Prism where
  n : ℕ
  vertices : ℕ := 2 * n
  edges : ℕ := 3 * n
  faces : ℕ := n + 2

/-- Theorem: A prism with 40 as the sum of its vertices and edges has 10 faces -/
theorem prism_faces (p : Prism) (h : p.vertices + p.edges = 40) : p.faces = 10 := by
  sorry


end NUMINAMATH_CALUDE_prism_faces_l3727_372749


namespace NUMINAMATH_CALUDE_responses_always_match_l3727_372730

-- Define the types of inhabitants
inductive Inhabitant : Type
| Knight : Inhabitant
| Liar : Inhabitant

-- Define a function to represent an inhabitant's response
def responds (a b : Inhabitant) : Prop :=
  match a, b with
  | Inhabitant.Knight, Inhabitant.Knight => true
  | Inhabitant.Knight, Inhabitant.Liar => false
  | Inhabitant.Liar, Inhabitant.Knight => true
  | Inhabitant.Liar, Inhabitant.Liar => true

-- Theorem: The responses of two inhabitants about each other are always the same
theorem responses_always_match (a b : Inhabitant) :
  responds a b = responds b a :=
sorry

end NUMINAMATH_CALUDE_responses_always_match_l3727_372730


namespace NUMINAMATH_CALUDE_power_multiplication_l3727_372776

theorem power_multiplication (x : ℝ) : (x^5) * (x^2) = x^7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3727_372776


namespace NUMINAMATH_CALUDE_last_flip_heads_prob_2010_l3727_372747

/-- A coin that comes up the same as the last flip 2/3 of the time and opposite 1/3 of the time -/
structure BiasedCoin where
  same_prob : ℚ
  diff_prob : ℚ
  prob_sum_one : same_prob + diff_prob = 1
  same_prob_val : same_prob = 2/3
  diff_prob_val : diff_prob = 1/3

/-- The probability of the last flip being heads after n flips, given the first flip was heads -/
def last_flip_heads_prob (coin : BiasedCoin) (n : ℕ) : ℚ :=
  (3^n + 1) / (2 * 3^n)

/-- The theorem statement -/
theorem last_flip_heads_prob_2010 (coin : BiasedCoin) :
  last_flip_heads_prob coin 2010 = (3^2010 + 1) / (2 * 3^2010) := by
  sorry

end NUMINAMATH_CALUDE_last_flip_heads_prob_2010_l3727_372747


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l3727_372710

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l3727_372710


namespace NUMINAMATH_CALUDE_distinct_cube_models_count_l3727_372736

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of colors available -/
def available_colors : ℕ := 8

/-- The number of rotational symmetries of a cube -/
def cube_rotations : ℕ := 24

/-- The number of distinct models of cubes with differently colored vertices -/
def distinct_cube_models : ℕ := Nat.factorial available_colors / cube_rotations

theorem distinct_cube_models_count :
  distinct_cube_models = 1680 := by sorry

end NUMINAMATH_CALUDE_distinct_cube_models_count_l3727_372736


namespace NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l3727_372708

/-- 
Given:
- Sandy attempts 30 sums
- Sandy obtains 45 marks in total
- Sandy got 21 sums correct
- Sandy loses 2 marks for each incorrect sum

Prove that Sandy gets 3 marks for each correct sum
-/
theorem sandy_marks_per_correct_sum 
  (total_sums : ℕ) 
  (total_marks : ℕ) 
  (correct_sums : ℕ) 
  (penalty_per_incorrect : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 45)
  (h3 : correct_sums = 21)
  (h4 : penalty_per_incorrect = 2) :
  (total_marks + penalty_per_incorrect * (total_sums - correct_sums)) / correct_sums = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l3727_372708


namespace NUMINAMATH_CALUDE_log_equation_solution_l3727_372772

theorem log_equation_solution (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 6 → x = 117649 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3727_372772


namespace NUMINAMATH_CALUDE_pizza_cut_area_theorem_l3727_372786

/-- Represents a circular pizza -/
structure Pizza where
  area : ℝ
  radius : ℝ

/-- Represents a cut on the pizza -/
structure Cut where
  distance_from_center : ℝ

/-- Theorem: Given a circular pizza with area 4 μ² cut into 4 parts by two perpendicular
    straight cuts each at a distance of 50 cm from the center, the sum of the areas of
    two opposite pieces is equal to 1.5 μ² -/
theorem pizza_cut_area_theorem (p : Pizza) (c : Cut) :
  p.area = 4 →
  c.distance_from_center = 0.5 →
  ∃ (piece1 piece2 : ℝ), piece1 + piece2 = 1.5 ∧ 
    piece1 + piece2 = (p.area - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_pizza_cut_area_theorem_l3727_372786


namespace NUMINAMATH_CALUDE_total_servings_in_block_l3727_372760

/-- Represents the number of calories in one serving of cheese -/
def calories_per_serving : ℕ := 110

/-- Represents the number of servings already eaten -/
def servings_eaten : ℕ := 5

/-- Represents the number of calories remaining in the block -/
def calories_remaining : ℕ := 1210

/-- Theorem stating that the total number of servings in the block is 16 -/
theorem total_servings_in_block : 
  (calories_per_serving * servings_eaten + calories_remaining) / calories_per_serving = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_servings_in_block_l3727_372760


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3727_372704

/-- Given a line and a circle that intersect at two points with a specific distance between them, 
    prove that the slope of the line has a specific value. -/
theorem line_circle_intersection (m : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (∀ x y : ℝ, m * x + y + 3 * m - Real.sqrt 3 = 0 → (x, y) ∈ {(x, y) | m * x + y + 3 * m - Real.sqrt 3 = 0}) ∧ 
    (∀ x y : ℝ, x^2 + y^2 = 12 → (x, y) ∈ {(x, y) | x^2 + y^2 = 12}) ∧
    A ∈ {(x, y) | m * x + y + 3 * m - Real.sqrt 3 = 0} ∧
    A ∈ {(x, y) | x^2 + y^2 = 12} ∧
    B ∈ {(x, y) | m * x + y + 3 * m - Real.sqrt 3 = 0} ∧
    B ∈ {(x, y) | x^2 + y^2 = 12} ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12) →
  m = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3727_372704


namespace NUMINAMATH_CALUDE_rare_card_cost_proof_l3727_372753

/-- The cost of each rare card in Tom's deck -/
def rare_card_cost : ℝ := 1

/-- The number of rare cards in Tom's deck -/
def num_rare_cards : ℕ := 19

/-- The number of uncommon cards in Tom's deck -/
def num_uncommon_cards : ℕ := 11

/-- The number of common cards in Tom's deck -/
def num_common_cards : ℕ := 30

/-- The cost of each uncommon card -/
def uncommon_card_cost : ℝ := 0.50

/-- The cost of each common card -/
def common_card_cost : ℝ := 0.25

/-- The total cost of Tom's deck -/
def total_deck_cost : ℝ := 32

theorem rare_card_cost_proof :
  rare_card_cost * num_rare_cards +
  uncommon_card_cost * num_uncommon_cards +
  common_card_cost * num_common_cards = total_deck_cost :=
by sorry

end NUMINAMATH_CALUDE_rare_card_cost_proof_l3727_372753


namespace NUMINAMATH_CALUDE_fuel_station_problem_l3727_372719

/-- Represents the problem of calculating the number of mini-vans filled up at a fuel station. -/
theorem fuel_station_problem (service_cost : ℝ) (fuel_cost_per_liter : ℝ) (total_cost : ℝ) 
  (minivan_tank : ℝ) (truck_tank : ℝ) (num_trucks : ℕ) :
  service_cost = 2.20 →
  fuel_cost_per_liter = 0.70 →
  total_cost = 347.7 →
  minivan_tank = 65 →
  truck_tank = minivan_tank * 2.2 →
  num_trucks = 2 →
  ∃ (num_minivans : ℕ), 
    (num_minivans : ℝ) * (service_cost + fuel_cost_per_liter * minivan_tank) + 
    (num_trucks : ℝ) * (service_cost + fuel_cost_per_liter * truck_tank) = total_cost ∧
    num_minivans = 3 :=
by sorry

end NUMINAMATH_CALUDE_fuel_station_problem_l3727_372719


namespace NUMINAMATH_CALUDE_elenas_car_rental_cost_l3727_372781

/-- Calculates the total cost of a car rental given the daily rate, mileage rate, number of days, and miles driven. -/
def carRentalCost (dailyRate : ℚ) (mileageRate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  dailyRate * days + mileageRate * miles

/-- Proves that the total cost of Elena's car rental is $275. -/
theorem elenas_car_rental_cost :
  carRentalCost 30 0.25 5 500 = 275 := by
  sorry

end NUMINAMATH_CALUDE_elenas_car_rental_cost_l3727_372781


namespace NUMINAMATH_CALUDE_part_one_part_two_l3727_372718

-- Define the line l: y = k(x-n)
def line (k n x : ℝ) : ℝ := k * (x - n)

-- Define the parabola y^2 = 4x
def parabola (x : ℝ) : ℝ := 4 * x

-- Define the intersection points
structure Point where
  x : ℝ
  y : ℝ

-- Theorem for part (I)
theorem part_one (k n : ℝ) (A B : Point) (h1 : A.y = line k n A.x) (h2 : B.y = line k n B.x)
  (h3 : A.y^2 = parabola A.x) (h4 : B.y^2 = parabola B.x) (h5 : A.x * B.x ≠ 0)
  (h6 : line k n 1 = 0) : A.x * B.x = 1 := by sorry

-- Theorem for part (II)
theorem part_two (k n : ℝ) (A B : Point) (h1 : A.y = line k n A.x) (h2 : B.y = line k n B.x)
  (h3 : A.y^2 = parabola A.x) (h4 : B.y^2 = parabola B.x) (h5 : A.x * B.x ≠ 0)
  (h6 : A.x * B.x + A.y * B.y = 0) : n = 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3727_372718


namespace NUMINAMATH_CALUDE_function_comparison_and_maximum_l3727_372744

def f (x : ℝ) := abs (x - 1)
def g (x : ℝ) := -x^2 + 6*x - 5

theorem function_comparison_and_maximum :
  (∀ x : ℝ, g x ≥ f x ↔ x ∈ Set.Icc 1 4) ∧
  (∃ M : ℝ, M = 9/4 ∧ ∀ x : ℝ, g x - f x ≤ M) :=
sorry

end NUMINAMATH_CALUDE_function_comparison_and_maximum_l3727_372744


namespace NUMINAMATH_CALUDE_smallest_integer_x_zero_satisfies_inequality_zero_is_smallest_l3727_372784

theorem smallest_integer_x (x : ℤ) : (3 - 2 * x^2 < 21) → x ≥ 0 :=
by sorry

theorem zero_satisfies_inequality : 3 - 2 * 0^2 < 21 :=
by sorry

theorem zero_is_smallest (x : ℤ) :
  x < 0 → ¬(3 - 2 * x^2 < 21) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_x_zero_satisfies_inequality_zero_is_smallest_l3727_372784


namespace NUMINAMATH_CALUDE_sum_of_squares_of_exponents_992_l3727_372790

-- Define a function to express a number as a sum of distinct powers of 2
def expressAsPowersOfTwo (n : ℕ) : List ℕ := sorry

-- Define a function to calculate the sum of squares of a list of numbers
def sumOfSquares (l : List ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_squares_of_exponents_992 :
  sumOfSquares (expressAsPowersOfTwo 992) = 255 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_exponents_992_l3727_372790


namespace NUMINAMATH_CALUDE_min_value_theorem_l3727_372702

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / x + 1 / y = 5) :
  3 * x + 4 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 1 / y₀ = 5 ∧ 3 * x₀ + 4 * y₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3727_372702


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l3727_372783

def total_players : ℕ := 18
def lineup_size : ℕ := 8
def triplets : ℕ := 3
def twins : ℕ := 2

def remaining_players : ℕ := total_players - (triplets + twins)
def players_to_choose : ℕ := lineup_size - (triplets + twins)

theorem starting_lineup_combinations : 
  Nat.choose remaining_players players_to_choose = 286 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l3727_372783


namespace NUMINAMATH_CALUDE_sum_of_multiples_l3727_372700

def smallest_two_digit_multiple_of_5 : ℕ := 10

def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l3727_372700


namespace NUMINAMATH_CALUDE_class_size_ratio_l3727_372796

theorem class_size_ratio : 
  let finley_class_size : ℕ := 24
  let johnson_class_size : ℕ := 22
  let half_finley_class_size : ℕ := finley_class_size / 2
  (johnson_class_size : ℚ) / (half_finley_class_size : ℚ) = 11 / 6 :=
by sorry

end NUMINAMATH_CALUDE_class_size_ratio_l3727_372796


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_value_l3727_372727

def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a3_value
  (a : ℕ → ℤ)
  (h_geometric : is_geometric_sequence a)
  (h_product : a 2 * a 5 = -32)
  (h_sum : a 3 + a 4 = 4)
  (h_integer_ratio : ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r) :
  a 3 = -4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_value_l3727_372727


namespace NUMINAMATH_CALUDE_inequality_statement_not_always_true_l3727_372768

theorem inequality_statement_not_always_true :
  ¬ (∀ a b c : ℝ, a < b → a * c^2 < b * c^2) :=
sorry

end NUMINAMATH_CALUDE_inequality_statement_not_always_true_l3727_372768


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3727_372709

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∃ (x y : ℝ), 2*x + 2*y = r ∧ x^2 + y^2 = 2*r) →
  (∀ (x y : ℝ), 2*x + 2*y = r → x^2 + y^2 ≥ 2*r) →
  r = 16 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3727_372709


namespace NUMINAMATH_CALUDE_complex_modulus_l3727_372779

theorem complex_modulus (z : ℂ) (h : z * (1 - Complex.I) = 2 - 3 * Complex.I) :
  Complex.abs z = Real.sqrt 26 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3727_372779


namespace NUMINAMATH_CALUDE_carl_needs_sixty_more_bags_l3727_372701

/-- The number of additional gift bags Carl needs to make for his open house -/
def additional_bags_needed (guaranteed_visitors : ℕ) (possible_visitors : ℕ) (extravagant_bags : ℕ) (average_bags : ℕ) : ℕ :=
  (guaranteed_visitors + possible_visitors) - (extravagant_bags + average_bags)

/-- Theorem stating that Carl needs to make 60 more gift bags -/
theorem carl_needs_sixty_more_bags :
  additional_bags_needed 50 40 10 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_carl_needs_sixty_more_bags_l3727_372701


namespace NUMINAMATH_CALUDE_logical_equivalence_l3727_372707

theorem logical_equivalence (P Q : Prop) :
  (¬P → ¬Q) ↔ (Q → P) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l3727_372707


namespace NUMINAMATH_CALUDE_remainder_3_pow_2023_mod_5_l3727_372752

theorem remainder_3_pow_2023_mod_5 : 3^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2023_mod_5_l3727_372752


namespace NUMINAMATH_CALUDE_inscribed_shape_perimeter_lower_bound_l3727_372733

/-- A shape inscribed in a circle -/
structure InscribedShape where
  -- The radius of the circumscribed circle
  radius : ℝ
  -- The perimeter of the shape
  perimeter : ℝ
  -- Predicate indicating if the center of the circle is inside or on the boundary of the shape
  center_inside : Prop

/-- Theorem: The perimeter of a shape inscribed in a circle is at least 4 times the radius
    if the center of the circle is inside or on the boundary of the shape -/
theorem inscribed_shape_perimeter_lower_bound
  (shape : InscribedShape)
  (h : shape.center_inside) :
  shape.perimeter ≥ 4 * shape.radius :=
sorry

end NUMINAMATH_CALUDE_inscribed_shape_perimeter_lower_bound_l3727_372733


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l3727_372758

/-- The parabolas y = (x + 1)² and x + 4 = (y - 3)² intersect at four points that lie on a circle --/
theorem intersection_points_on_circle (x y : ℝ) : 
  (y = (x + 1)^2 ∧ x + 4 = (y - 3)^2) →
  ∃ (center : ℝ × ℝ), (x - center.1)^2 + (y - center.2)^2 = 13/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l3727_372758


namespace NUMINAMATH_CALUDE_reflection_equivalence_l3727_372769

-- Define the shape type
inductive Shape
  | OriginalL
  | InvertedL
  | UpsideDownRotatedL
  | VerticallyFlippedL
  | HorizontallyMirroredL
  | UnalteredL

-- Define the reflection operation
def reflectAcrossDiagonal (s : Shape) : Shape :=
  match s with
  | Shape.OriginalL => Shape.HorizontallyMirroredL
  | _ => s  -- For completeness, though we only care about OriginalL

-- State the theorem
theorem reflection_equivalence :
  reflectAcrossDiagonal Shape.OriginalL = Shape.HorizontallyMirroredL :=
by sorry

end NUMINAMATH_CALUDE_reflection_equivalence_l3727_372769


namespace NUMINAMATH_CALUDE_identical_balls_distribution_seven_balls_four_boxes_l3727_372740

theorem identical_balls_distribution (n m : ℕ) (hn : n ≥ m) :
  (Nat.choose (n + m - 1) (m - 1) : ℕ) = (Nat.choose (n - 1) (m - 1) : ℕ) := by
  sorry

theorem seven_balls_four_boxes :
  (Nat.choose 6 3 : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_identical_balls_distribution_seven_balls_four_boxes_l3727_372740


namespace NUMINAMATH_CALUDE_division_result_l3727_372713

theorem division_result : (2014 : ℕ) / (2 * 2 + 2 * 3 + 3 * 3) = 106 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l3727_372713


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3727_372734

-- Problem 1
theorem problem_1 (x y : ℝ) :
  3 * x^2 * (-3 * x * y)^2 - x^2 * (x^2 * y^2 - 2 * x) = 26 * x^4 * y^2 + 2 * x^3 :=
by sorry

-- Problem 2
theorem problem_2 (a b c : ℝ) :
  -2 * (-a^2 * b * c)^2 * (1/2) * a * (b * c)^3 - (-a * b * c)^3 * (-a * b * c)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3727_372734


namespace NUMINAMATH_CALUDE_number_system_generalization_l3727_372737

-- Define the number systems
inductive NumberSystem
| Natural
| Integer
| Rational
| Real
| Complex

-- Define the basic operations
inductive Operation
| Addition
| Subtraction
| Multiplication
| Division
| SquareRoot

-- Define a function to check if an operation is executable in a given number system
def is_executable (op : Operation) (ns : NumberSystem) : Prop :=
  match op, ns with
  | Operation.Subtraction, NumberSystem.Natural => false
  | Operation.Division, NumberSystem.Integer => false
  | Operation.SquareRoot, NumberSystem.Rational => false
  | Operation.SquareRoot, NumberSystem.Real => false
  | _, _ => true

-- Define the theorem
theorem number_system_generalization (op : Operation) :
  ∃ ns : NumberSystem, is_executable op ns :=
sorry

end NUMINAMATH_CALUDE_number_system_generalization_l3727_372737


namespace NUMINAMATH_CALUDE_finite_gcd_lcm_process_terminates_l3727_372746

theorem finite_gcd_lcm_process_terminates 
  (n : ℕ) 
  (a : Fin n → ℕ+) : 
  ∃ (k : ℕ), ∀ (j k : Fin n), j < k → (a j).val ∣ (a k).val :=
sorry

end NUMINAMATH_CALUDE_finite_gcd_lcm_process_terminates_l3727_372746


namespace NUMINAMATH_CALUDE_debby_messages_before_noon_l3727_372765

/-- The number of text messages Debby received before noon -/
def messages_before_noon : ℕ := sorry

/-- The number of text messages Debby received after noon -/
def messages_after_noon : ℕ := 18

/-- The total number of text messages Debby received -/
def total_messages : ℕ := 39

/-- Theorem stating that Debby received 21 text messages before noon -/
theorem debby_messages_before_noon :
  messages_before_noon = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_debby_messages_before_noon_l3727_372765


namespace NUMINAMATH_CALUDE_geometric_progression_sum_not_end_20_l3727_372742

/-- Given a, b, c form a geometric progression, prove that a^3 + b^3 + c^3 - 3abc cannot end with 20 -/
theorem geometric_progression_sum_not_end_20 
  (a b c : ℤ) 
  (h_geom : ∃ (q : ℚ), b = a * q ∧ c = b * q) : 
  ¬ (∃ (k : ℤ), a^3 + b^3 + c^3 - 3*a*b*c = 100*k + 20) := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_not_end_20_l3727_372742


namespace NUMINAMATH_CALUDE_ice_cream_volume_l3727_372787

/-- The volume of ice cream in a cone with hemisphere and cylinder topping -/
theorem ice_cream_volume (cone_height : Real) (cone_radius : Real) (cylinder_height : Real) :
  cone_height = 12 →
  cone_radius = 3 →
  cylinder_height = 2 →
  (1/3 * Real.pi * cone_radius^2 * cone_height) + 
  (2/3 * Real.pi * cone_radius^3) + 
  (Real.pi * cone_radius^2 * cylinder_height) = 72 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l3727_372787


namespace NUMINAMATH_CALUDE_point_on_line_l3727_372725

/-- A point represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let p1 : Point := ⟨0, 4⟩
  let p2 : Point := ⟨-6, 1⟩
  let p3 : Point := ⟨6, 7⟩
  collinear p1 p2 p3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l3727_372725


namespace NUMINAMATH_CALUDE_expression_simplification_l3727_372793

theorem expression_simplification (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 6) * 5 + (5 - 2 / 4) * (8 * p - 12) = -19 * p - 39 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3727_372793


namespace NUMINAMATH_CALUDE_min_n_for_sum_greater_than_1020_l3727_372743

def sequence_term (n : ℕ) : ℕ := 2^n - 1

def sequence_sum (n : ℕ) : ℕ := 2^(n+1) - 2 - n

theorem min_n_for_sum_greater_than_1020 :
  (∀ k < 10, sequence_sum k ≤ 1020) ∧ (sequence_sum 10 > 1020) := by sorry

end NUMINAMATH_CALUDE_min_n_for_sum_greater_than_1020_l3727_372743


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3727_372764

/-- Given that a and b are inversely proportional, their sum is 40, and their modified difference is 10, prove that b equals 75 when a equals 4. -/
theorem inverse_proportion_problem (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : a + b = 40) (h3 : a - 2*b = 10) : 
  a = 4 → b = 75 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3727_372764


namespace NUMINAMATH_CALUDE_magic_square_a_plus_b_l3727_372748

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (w y a b z : ℕ)
  (magic_sum : ℕ)
  (top_row : 19 + w + 23 = magic_sum)
  (middle_row : 22 + y + a = magic_sum)
  (bottom_row : b + 18 + z = magic_sum)
  (left_column : 19 + 22 + b = magic_sum)
  (middle_column : w + y + 18 = magic_sum)
  (right_column : 23 + a + z = magic_sum)
  (main_diagonal : 19 + y + z = magic_sum)
  (secondary_diagonal : 23 + y + b = magic_sum)

/-- The sum of a and b in the magic square is 23 -/
theorem magic_square_a_plus_b (ms : MagicSquare) : ms.a + ms.b = 23 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_a_plus_b_l3727_372748


namespace NUMINAMATH_CALUDE_profit_maximizing_price_optimal_selling_price_is_14_l3727_372738

/-- Profit function given price increase -/
def profit (x : ℝ) : ℝ :=
  (100 - 10 * x) * ((10 + x) - 8)

/-- The price increase that maximizes profit -/
def optimal_price_increase : ℝ := 4

theorem profit_maximizing_price :
  optimal_price_increase = 4 ∧
  ∀ x : ℝ, profit x ≤ profit optimal_price_increase :=
sorry

/-- The optimal selling price -/
def optimal_selling_price : ℝ :=
  10 + optimal_price_increase

theorem optimal_selling_price_is_14 :
  optimal_selling_price = 14 :=
sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_optimal_selling_price_is_14_l3727_372738


namespace NUMINAMATH_CALUDE_triangle_inequality_cube_l3727_372788

/-- Triangle inequality for sides a, b, c -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- The inequality to be proven -/
theorem triangle_inequality_cube (a b c : ℝ) (h : is_triangle a b c) :
  a^3 + b^3 + c^3 + 4*a*b*c ≤ 9/32 * (a + b + c)^3 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_cube_l3727_372788


namespace NUMINAMATH_CALUDE_quadratic_equation_rational_roots_l3727_372723

theorem quadratic_equation_rational_roots (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (∃ x y : ℚ, x^2 + p^2 * x + q^3 = 0 ∧ y^2 + p^2 * y + q^3 = 0 ∧ x ≠ y) ↔ 
  (p = 3 ∧ q = 2 ∧ 
   ∃ x y : ℚ, x = -1 ∧ y = -8 ∧ 
   x^2 + p^2 * x + q^3 = 0 ∧ y^2 + p^2 * y + q^3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_rational_roots_l3727_372723


namespace NUMINAMATH_CALUDE_functional_inequality_solution_l3727_372726

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x + y + z) + f x ≥ f (x + y) + f (x + z)

/-- The main theorem statement -/
theorem functional_inequality_solution
    (f : ℝ → ℝ)
    (h_diff : Differentiable ℝ f)
    (h_ineq : SatisfiesInequality f) :
    ∃ a b : ℝ, ∀ x, f x = a * x + b :=
  sorry

end NUMINAMATH_CALUDE_functional_inequality_solution_l3727_372726


namespace NUMINAMATH_CALUDE_min_value_expression_l3727_372724

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ 3 * Real.sqrt 2 ∧
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) = 3 * Real.sqrt 2 ↔ y = x * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3727_372724


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l3727_372715

/-- The number of points on the circle -/
def n : ℕ := 7

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The number of ways to select k chords from the total chords -/
def total_selections : ℕ := total_chords.choose k

/-- The number of ways to select k points from n points -/
def convex_quads : ℕ := n.choose k

/-- The probability of forming a convex quadrilateral -/
def probability : ℚ := convex_quads / total_selections

theorem convex_quadrilateral_probability :
  probability = 1 / 171 :=
sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l3727_372715
