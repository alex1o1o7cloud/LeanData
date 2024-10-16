import Mathlib

namespace NUMINAMATH_CALUDE_max_value_reciprocal_sum_l2720_272055

theorem max_value_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1) 
  (hx : a^x = 3) (hy : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  (∀ x' y' : ℝ, a^x' = 3 → b^y' = 3 → 1/x' + 1/y' ≤ 1) ∧ 
  (∃ x'' y'' : ℝ, a^x'' = 3 ∧ b^y'' = 3 ∧ 1/x'' + 1/y'' = 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_reciprocal_sum_l2720_272055


namespace NUMINAMATH_CALUDE_raghu_investment_l2720_272047

/-- Proves that Raghu's investment is 2000 given the problem conditions --/
theorem raghu_investment (raghu trishul vishal : ℝ) : 
  trishul = raghu * 0.9 →
  vishal = trishul * 1.1 →
  raghu + trishul + vishal = 5780 →
  raghu = 2000 := by
sorry

end NUMINAMATH_CALUDE_raghu_investment_l2720_272047


namespace NUMINAMATH_CALUDE_surface_area_of_stacked_cubes_l2720_272015

/-- The surface area of a cube formed by stacking smaller cubes -/
theorem surface_area_of_stacked_cubes (n : Nat) (side_length : Real) :
  n > 0 →
  side_length > 0 →
  n = 27 →
  side_length = 3 →
  let large_cube_side := (n ^ (1 / 3 : Real)) * side_length
  6 * large_cube_side ^ 2 = 486 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_stacked_cubes_l2720_272015


namespace NUMINAMATH_CALUDE_inequality_proof_l2720_272092

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^3 + b^3 + a + b ≥ 4*a*b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2720_272092


namespace NUMINAMATH_CALUDE_min_distance_to_2i_l2720_272037

theorem min_distance_to_2i (z : ℂ) (h : Complex.abs (z^2 - 1) = Complex.abs (z * (z - Complex.I))) :
  ∃ (w : ℂ), Complex.abs (w - 2 * Complex.I) = 1 ∧ 
  ∀ (z : ℂ), Complex.abs (z - 2 * Complex.I) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_min_distance_to_2i_l2720_272037


namespace NUMINAMATH_CALUDE_fractional_equation_root_l2720_272026

theorem fractional_equation_root (m : ℝ) : 
  (∃ x : ℝ, x ≠ 4 ∧ (3 / (x - 4) + (x + m) / (4 - x) = 1)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l2720_272026


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2720_272032

theorem min_value_of_expression (x : ℚ) : (2*x - 5)^2 + 18 ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2720_272032


namespace NUMINAMATH_CALUDE_unique_solution_is_identity_l2720_272094

open Set
open Function
open Real

/-- The functional equation that f must satisfy for all positive real numbers x, y, z -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y z, x > 0 → y > 0 → z > 0 →
    (z + 1) * f (x + y) = f (x * f z + y) + f (y * f z + x)

/-- The main theorem stating that the only function satisfying the equation is the identity function -/
theorem unique_solution_is_identity :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x > 0) →
    satisfies_equation f →
    ∀ x, x > 0 → f x = x :=
by sorry


end NUMINAMATH_CALUDE_unique_solution_is_identity_l2720_272094


namespace NUMINAMATH_CALUDE_triangle_median_and_altitude_l2720_272019

structure Point where
  x : ℝ
  y : ℝ

def Triangle (A B C : Point) := True

def isMedian (l : ℝ → ℝ → Prop) (A B C : Point) : Prop :=
  ∃ D : Point, D.x = (B.x + C.x) / 2 ∧ D.y = (B.y + C.y) / 2 ∧ l D.x D.y ∧ l A.x A.y

def isAltitude (l : ℝ → ℝ → Prop) (A B C : Point) : Prop :=
  ∀ x y : ℝ, l x y → (x - A.x) * (C.x - A.x) + (y - A.y) * (C.y - A.y) = 0

theorem triangle_median_and_altitude 
  (A B C : Point)
  (h_triangle : Triangle A B C)
  (h_A : A.x = 1 ∧ A.y = 3)
  (h_B : B.x = 5 ∧ B.y = 1)
  (h_C : C.x = -1 ∧ C.y = -1) :
  (isMedian (fun x y => 3 * x + y - 6 = 0) A B C) ∧
  (isAltitude (fun x y => x + 2 * y - 7 = 0) B A C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_median_and_altitude_l2720_272019


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2720_272067

theorem quadratic_root_range (a : ℝ) (x₁ x₂ : ℝ) :
  (∃ x, x^2 + a*x - 2 = 0) ∧ 
  (x₁^2 + a*x₁ - 2 = 0) ∧ 
  (x₂^2 + a*x₂ - 2 = 0) ∧ 
  (x₁ < 1) ∧ (1 < x₂) →
  a < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2720_272067


namespace NUMINAMATH_CALUDE_tan_difference_l2720_272003

theorem tan_difference (α β : Real) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4/3) :
  Real.tan (α - β) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_l2720_272003


namespace NUMINAMATH_CALUDE_airplane_trip_people_count_l2720_272071

/-- Represents the airplane trip scenario --/
structure AirplaneTrip where
  bagsPerPerson : ℕ
  weightPerBag : ℕ
  currentCapacity : ℕ
  additionalBags : ℕ

/-- Calculate the number of people on the trip --/
def numberOfPeople (trip : AirplaneTrip) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the number of people on the trip --/
theorem airplane_trip_people_count :
  let trip := AirplaneTrip.mk 5 50 6000 90
  numberOfPeople trip = 42 := by
  sorry

end NUMINAMATH_CALUDE_airplane_trip_people_count_l2720_272071


namespace NUMINAMATH_CALUDE_vector_expression_simplification_l2720_272087

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

theorem vector_expression_simplification (a b : V) :
  (2/3 : ℝ) • ((4 • a - 3 • b) + (1/3 : ℝ) • b - (1/4 : ℝ) • (6 • a - 7 • b)) =
  (5/3 : ℝ) • a - (11/18 : ℝ) • b := by sorry

end NUMINAMATH_CALUDE_vector_expression_simplification_l2720_272087


namespace NUMINAMATH_CALUDE_product_sum_bounds_l2720_272079

def pairProductSum (pairs : List (ℕ × ℕ)) : ℕ :=
  (pairs.map (λ (a, b) => a * b)).sum

theorem product_sum_bounds :
  ∀ (pairs : List (ℕ × ℕ)),
    pairs.length = 50 ∧
    (pairs.map Prod.fst ++ pairs.map Prod.snd).toFinset = Finset.range 100
    →
    85850 ≤ pairProductSum pairs ∧ pairProductSum pairs ≤ 169150 :=
by sorry

end NUMINAMATH_CALUDE_product_sum_bounds_l2720_272079


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l2720_272065

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ (3 * x^2 + 1 = 4) ↔ (x = -1 ∧ y = -4) ∨ (x = 1 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l2720_272065


namespace NUMINAMATH_CALUDE_perfect_square_sum_l2720_272041

theorem perfect_square_sum (n : ℝ) (h : n > 2) :
  ∃ m : ℝ, ∃ k : ℝ, n^2 + m^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l2720_272041


namespace NUMINAMATH_CALUDE_student_marks_l2720_272002

theorem student_marks (total_marks passing_percentage failing_margin : ℕ) 
  (h1 : total_marks = 440)
  (h2 : passing_percentage = 50)
  (h3 : failing_margin = 20) : 
  (total_marks * passing_percentage / 100 - failing_margin : ℕ) = 200 :=
by sorry

end NUMINAMATH_CALUDE_student_marks_l2720_272002


namespace NUMINAMATH_CALUDE_optimal_revenue_model_depends_on_factors_l2720_272007

/-- Represents the revenue model for a movie --/
inductive RevenueModel
  | Forever
  | Rental

/-- Represents various economic factors --/
structure EconomicFactors where
  immediateRevenue : ℝ
  longTermRevenuePotential : ℝ
  customerPriceSensitivity : ℝ
  administrativeCosts : ℝ
  piracyRisks : ℝ

/-- Calculates the overall economic value of a revenue model --/
def economicValue (model : RevenueModel) (factors : EconomicFactors) : ℝ :=
  sorry

/-- The theorem stating that the optimal revenue model depends on economic factors --/
theorem optimal_revenue_model_depends_on_factors
  (factors : EconomicFactors) :
  ∃ (model : RevenueModel),
    ∀ (other : RevenueModel),
      economicValue model factors ≥ economicValue other factors :=
  sorry

end NUMINAMATH_CALUDE_optimal_revenue_model_depends_on_factors_l2720_272007


namespace NUMINAMATH_CALUDE_litter_patrol_problem_l2720_272056

/-- The Litter Patrol problem -/
theorem litter_patrol_problem (total_litter aluminum_cans : ℕ) 
  (h1 : total_litter = 18)
  (h2 : aluminum_cans = 8)
  (h3 : total_litter = aluminum_cans + glass_bottles) :
  glass_bottles = 10 := by
  sorry

end NUMINAMATH_CALUDE_litter_patrol_problem_l2720_272056


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l2720_272020

/-- 
If the quadratic equation x^2 + kx - 3 = 0 has 1 as a root, 
then k = 2.
-/
theorem quadratic_root_implies_k (k : ℝ) : 
  (1 : ℝ)^2 + k*(1 : ℝ) - 3 = 0 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l2720_272020


namespace NUMINAMATH_CALUDE_prime_factorization_sum_l2720_272062

theorem prime_factorization_sum (a b c d : ℕ) : 
  2^a * 3^b * 5^c * 11^d = 14850 → 3*a + 2*b + 4*c + 6*d = 23 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_sum_l2720_272062


namespace NUMINAMATH_CALUDE_sqrt_of_four_l2720_272035

theorem sqrt_of_four : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_four_l2720_272035


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l2720_272064

/-- The amount of oil leaked before repairs, in liters -/
def oil_leaked_before : ℕ := 6522

/-- The amount of oil leaked during repairs, in liters -/
def oil_leaked_during : ℕ := 5165

/-- The total amount of oil leaked, in liters -/
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem oil_leak_calculation :
  total_oil_leaked = 11687 :=
by sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l2720_272064


namespace NUMINAMATH_CALUDE_football_team_throwers_l2720_272031

/-- Proves the number of throwers on a football team given specific conditions -/
theorem football_team_throwers 
  (total_players : ℕ) 
  (total_right_handed : ℕ) 
  (h_total : total_players = 70)
  (h_right : total_right_handed = 62)
  (h_throwers_right : ∀ t : ℕ, t ≤ total_players → t ≤ total_right_handed)
  (h_non_throwers_division : ∀ n : ℕ, n < total_players → 
    3 * (total_players - n) = 2 * (total_right_handed - n) + (total_players - total_right_handed)) :
  ∃ throwers : ℕ, throwers = 46 ∧ throwers ≤ total_players ∧ throwers ≤ total_right_handed :=
sorry

end NUMINAMATH_CALUDE_football_team_throwers_l2720_272031


namespace NUMINAMATH_CALUDE_repeating_decimal_inequality_l2720_272095

/-- Represents a repeating decimal with non-repeating part P and repeating part Q -/
structure RepeatingDecimal where
  P : ℕ  -- non-repeating part
  Q : ℕ  -- repeating part
  r : ℕ  -- number of digits in P
  s : ℕ  -- number of digits in Q

/-- The value of the repeating decimal as a real number -/
noncomputable def decimal_value (D : RepeatingDecimal) : ℝ :=
  sorry

/-- Statement: The equation 10^r(10^s + 1)D = Q(P + 1) is false for repeating decimals -/
theorem repeating_decimal_inequality (D : RepeatingDecimal) :
  (10^D.r * (10^D.s + 1)) * (decimal_value D) ≠ D.Q * (D.P + 1) :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_inequality_l2720_272095


namespace NUMINAMATH_CALUDE_gala_trees_count_l2720_272082

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  fuji : ℕ
  gala : ℕ
  cross_pollinated : ℕ

/-- The conditions of the orchard problem -/
def orchard_conditions (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧
  o.fuji + o.cross_pollinated = 204 ∧
  o.fuji = 3 * o.total / 4 ∧
  o.total = o.fuji + o.gala + o.cross_pollinated

theorem gala_trees_count (o : Orchard) (h : orchard_conditions o) : o.gala = 60 := by
  sorry

end NUMINAMATH_CALUDE_gala_trees_count_l2720_272082


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2720_272080

/-- The solution set of the inequality -x^2 - x + 6 > 0 is the open interval (-3, 2) -/
theorem solution_set_quadratic_inequality :
  {x : ℝ | -x^2 - x + 6 > 0} = Set.Ioo (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2720_272080


namespace NUMINAMATH_CALUDE_trajectory_curve_intersection_range_l2720_272028

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the moving point M and its projection N on AB
def M : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (x, y)
def N : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (x, 0)

-- Define vectors
def vec_MN (m : ℝ × ℝ) : ℝ × ℝ := (0, -(m.2))
def vec_AN (n : ℝ × ℝ) : ℝ × ℝ := (n.1 + 1, 0)
def vec_BN (n : ℝ × ℝ) : ℝ × ℝ := (n.1 - 1, 0)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the condition for point M
def condition (m : ℝ × ℝ) : Prop :=
  let n := N m
  (vec_MN m).1^2 + (vec_MN m).2^2 = dot_product (vec_AN n) (vec_BN n)

-- Define the trajectory curve E
def curve_E (p : ℝ × ℝ) : Prop := p.1^2 - p.2^2 = 1

-- Define the line l
def line_l (k : ℝ) (p : ℝ × ℝ) : Prop := p.2 = k * p.1 - 1

-- Theorem statements
theorem trajectory_curve : ∀ m : ℝ × ℝ, condition m ↔ curve_E m := by sorry

theorem intersection_range : ∀ k : ℝ,
  (∃ p : ℝ × ℝ, curve_E p ∧ line_l k p) ↔ -Real.sqrt 2 ≤ k ∧ k ≤ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_trajectory_curve_intersection_range_l2720_272028


namespace NUMINAMATH_CALUDE_family_size_l2720_272042

def total_spent : ℕ := 119
def adult_ticket_price : ℕ := 21
def child_ticket_price : ℕ := 14
def adult_tickets_purchased : ℕ := 4

theorem family_size :
  ∃ (child_tickets : ℕ),
    adult_tickets_purchased * adult_ticket_price + child_tickets * child_ticket_price = total_spent ∧
    adult_tickets_purchased + child_tickets = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_family_size_l2720_272042


namespace NUMINAMATH_CALUDE_two_solutions_only_l2720_272066

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem two_solutions_only : 
  {k : ℕ | k > 0 ∧ digit_product k = (25 * k) / 8 - 211} = {72, 88} :=
by sorry

end NUMINAMATH_CALUDE_two_solutions_only_l2720_272066


namespace NUMINAMATH_CALUDE_lisa_to_total_ratio_l2720_272088

def total_earnings : ℝ := 60

def lisa_earnings (l : ℝ) : Prop := 
  ∃ (j t : ℝ), l + j + t = total_earnings ∧ t = l / 2 ∧ l = t + 15

theorem lisa_to_total_ratio : 
  ∀ l : ℝ, lisa_earnings l → l / total_earnings = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_lisa_to_total_ratio_l2720_272088


namespace NUMINAMATH_CALUDE_work_completion_time_l2720_272085

/-- Given workers x and y, where:
    - y takes 30 days to complete the entire work
    - x works for 8 days
    - y finishes the remaining work in 24 days
    Prove that x takes 40 days to complete the work alone. -/
theorem work_completion_time (x y : ℝ) (h1 : y > 0) : 
  (30 : ℝ) * y = 1 →  -- y completes the work in 30 days
  (8 : ℝ) * x + (24 : ℝ) * y = 1 →  -- x works for 8 days, then y finishes in 24 days
  x = (1 : ℝ) / 40 :=  -- x completes the work in 40 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2720_272085


namespace NUMINAMATH_CALUDE_parallel_line_equation_l2720_272039

/-- Given two parallel lines with a distance of 2 between them, where one line has the equation 5x - 12y + 6 = 0, prove that the equation of the other line is either 5x - 12y + 32 = 0 or 5x - 12y - 20 = 0 -/
theorem parallel_line_equation (x y : ℝ) :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 5 * x - 12 * y + 6 = 0
  let l : ℝ → ℝ → Prop := λ x y ↦ 5 * x - 12 * y + 32 = 0 ∨ 5 * x - 12 * y - 20 = 0
  let parallel : Prop := ∃ (k : ℝ), k ≠ 0 ∧ ∀ x y, l₁ x y ↔ l x y
  let distance : ℝ := 2
  parallel → (∀ x y, l x y ↔ (5 * x - 12 * y + 32 = 0 ∨ 5 * x - 12 * y - 20 = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l2720_272039


namespace NUMINAMATH_CALUDE_probability_one_black_one_white_l2720_272060

/-- The probability of selecting one black ball and one white ball from a jar containing 6 black balls and 2 white balls when picking two balls at the same time. -/
theorem probability_one_black_one_white (black_balls : ℕ) (white_balls : ℕ) 
  (h1 : black_balls = 6) (h2 : white_balls = 2) :
  (black_balls * white_balls : ℚ) / (Nat.choose (black_balls + white_balls) 2) = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_black_one_white_l2720_272060


namespace NUMINAMATH_CALUDE_cubic_roots_l2720_272006

def f (x : ℝ) : ℝ := x^3 - 4*x^2 - 7*x + 10

theorem cubic_roots :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 5) ∧
  (f 1 = 0) ∧ (f (-2) = 0) ∧ (f 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_l2720_272006


namespace NUMINAMATH_CALUDE_smallest_cube_ending_528_l2720_272044

theorem smallest_cube_ending_528 :
  ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 528 ∧ ∀ (m : ℕ), m > 0 ∧ m^3 % 1000 = 528 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_528_l2720_272044


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l2720_272097

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let t : ℕ := 2^n + n
  let r : ℕ := 3^t - t
  r = 177136 := by sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l2720_272097


namespace NUMINAMATH_CALUDE_margie_change_l2720_272004

/-- The change received when buying apples -/
def change_received (num_apples : ℕ) (cost_per_apple : ℚ) (amount_paid : ℚ) : ℚ :=
  amount_paid - (num_apples : ℚ) * cost_per_apple

/-- Theorem: Margie's change when buying apples -/
theorem margie_change : 
  let num_apples : ℕ := 3
  let cost_per_apple : ℚ := 50 / 100
  let amount_paid : ℚ := 5
  change_received num_apples cost_per_apple amount_paid = 7 / 2 := by
sorry

end NUMINAMATH_CALUDE_margie_change_l2720_272004


namespace NUMINAMATH_CALUDE_sum_of_squares_l2720_272034

theorem sum_of_squares (x y z : ℝ) (h_positive : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2720_272034


namespace NUMINAMATH_CALUDE_modulo_congruence_unique_solution_l2720_272011

theorem modulo_congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 25 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_unique_solution_l2720_272011


namespace NUMINAMATH_CALUDE_solve_grocery_cost_l2720_272009

def grocery_cost_problem (initial_amount : ℝ) (sister_fraction : ℝ) (remaining_amount : ℝ) : Prop :=
  let amount_to_sister := initial_amount * sister_fraction
  let amount_after_giving := initial_amount - amount_to_sister
  let grocery_cost := amount_after_giving - remaining_amount
  grocery_cost = 40

theorem solve_grocery_cost :
  grocery_cost_problem 100 (1/4) 35 := by
  sorry

end NUMINAMATH_CALUDE_solve_grocery_cost_l2720_272009


namespace NUMINAMATH_CALUDE_bread_pieces_theorem_l2720_272027

/-- Number of pieces after tearing a slice of bread in half twice -/
def pieces_per_slice : ℕ := 4

/-- Number of initial bread slices -/
def initial_slices : ℕ := 2

/-- Total number of bread pieces after tearing -/
def total_pieces : ℕ := initial_slices * pieces_per_slice

theorem bread_pieces_theorem : total_pieces = 8 := by
  sorry

end NUMINAMATH_CALUDE_bread_pieces_theorem_l2720_272027


namespace NUMINAMATH_CALUDE_ramon_age_l2720_272033

/-- Ramon's age problem -/
theorem ramon_age (loui_age : ℕ) (ramon_future_age : ℕ) : 
  loui_age = 23 →
  ramon_future_age = 2 * loui_age →
  ramon_future_age - 20 = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_ramon_age_l2720_272033


namespace NUMINAMATH_CALUDE_ellipse_k_range_correct_l2720_272089

/-- The range of k for which the equation x²/(k-2) + y²/(3-k) = 1 represents an ellipse -/
def ellipse_k_range : Set ℝ :=
  {k : ℝ | (2 < k ∧ k < 5/2) ∨ (5/2 < k ∧ k < 3)}

/-- The equation of the ellipse -/
def is_ellipse (k : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (k - 2) + y^2 / (3 - k) = 1 → 
    (k - 2 > 0 ∧ 3 - k > 0 ∧ k - 2 ≠ 3 - k)

theorem ellipse_k_range_correct :
  ∀ k : ℝ, is_ellipse k ↔ k ∈ ellipse_k_range :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_correct_l2720_272089


namespace NUMINAMATH_CALUDE_sum_of_ages_is_fifty_l2720_272022

/-- The sum of ages of 5 children born at intervals of 3 years, with the youngest being 4 years old -/
def sum_of_ages : ℕ :=
  let youngest_age := 4
  let interval := 3
  let num_children := 5
  List.range num_children
    |>.map (fun i => youngest_age + i * interval)
    |>.sum

theorem sum_of_ages_is_fifty :
  sum_of_ages = 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_is_fifty_l2720_272022


namespace NUMINAMATH_CALUDE_fib_mod_5_periodic_fib_10_mod_5_fib_50_mod_5_l2720_272054

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fib_mod_5_periodic (n : ℕ) : fib n % 5 = fib (n % 20) % 5 := sorry

theorem fib_10_mod_5 : fib 10 % 5 = 0 := sorry

theorem fib_50_mod_5 : fib 50 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_mod_5_periodic_fib_10_mod_5_fib_50_mod_5_l2720_272054


namespace NUMINAMATH_CALUDE_expression_value_l2720_272078

theorem expression_value (x : ℤ) (h : x = -2) : 4 * x - 5 = -13 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2720_272078


namespace NUMINAMATH_CALUDE_equation_solution_l2720_272038

theorem equation_solution : 
  let f (x : ℝ) := (x^2 + 3*x - 4)^2 + (2*x^2 - 7*x + 6)^2 - (3*x^2 - 4*x + 2)^2
  ∀ x : ℝ, f x = 0 ↔ x = -4 ∨ x = 1 ∨ x = 3/2 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2720_272038


namespace NUMINAMATH_CALUDE_min_basketballs_is_two_l2720_272081

/-- Represents the number of items sold for each type of sporting good. -/
structure ItemsSold where
  frisbees : ℕ
  baseballs : ℕ
  basketballs : ℕ

/-- Checks if the given ItemsSold satisfies all conditions of the problem. -/
def satisfiesConditions (items : ItemsSold) : Prop :=
  items.frisbees + items.baseballs + items.basketballs = 180 ∧
  3 * items.frisbees + 5 * items.baseballs + 10 * items.basketballs = 800 ∧
  items.frisbees > items.baseballs ∧
  items.baseballs > items.basketballs

/-- The minimum number of basketballs that could have been sold. -/
def minBasketballs : ℕ := 2

/-- Theorem stating that the minimum number of basketballs sold is 2. -/
theorem min_basketballs_is_two :
  ∀ items : ItemsSold,
    satisfiesConditions items →
    items.basketballs ≥ minBasketballs :=
by
  sorry

#check min_basketballs_is_two

end NUMINAMATH_CALUDE_min_basketballs_is_two_l2720_272081


namespace NUMINAMATH_CALUDE_unique_solution_l2720_272049

theorem unique_solution (x y z : ℝ) 
  (hx : x > 2) (hy : y > 2) (hz : z > 2)
  (heq : ((x + 3)^2) / (y + z - 3) + ((y + 5)^2) / (z + x - 5) + ((z + 7)^2) / (x + y - 7) = 45) :
  x = 13 ∧ y = 11 ∧ z = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l2720_272049


namespace NUMINAMATH_CALUDE_sequence_closed_form_l2720_272012

def recurrence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = Real.sqrt ((a n + 2 - Real.sqrt (2 - a n)) / 2)

theorem sequence_closed_form (a : ℕ → ℝ) :
  recurrence a ∧ a 0 = Real.sqrt 2 / 2 →
  ∀ n, a n = Real.sqrt 2 * Real.cos (π / 4 + π / (12 * 2^n)) :=
sorry

end NUMINAMATH_CALUDE_sequence_closed_form_l2720_272012


namespace NUMINAMATH_CALUDE_jeremy_pill_count_l2720_272001

/-- Calculates the total number of pills taken over a period of time given dosage information --/
def total_pills (dose_mg : ℕ) (dose_interval_hours : ℕ) (pill_mg : ℕ) (duration_weeks : ℕ) : ℕ :=
  let pills_per_dose := dose_mg / pill_mg
  let doses_per_day := 24 / dose_interval_hours
  let pills_per_day := pills_per_dose * doses_per_day
  let days := duration_weeks * 7
  pills_per_day * days

/-- Proves that Jeremy takes 112 pills in total during his 2-week treatment --/
theorem jeremy_pill_count : total_pills 1000 6 500 2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_pill_count_l2720_272001


namespace NUMINAMATH_CALUDE_quadrupled_bonus_remainder_l2720_272084

/-- Represents the bonus pool and its division among employees -/
structure BonusPool :=
  (total : ℕ)
  (employees : ℕ)
  (remainder : ℕ)

/-- Theorem stating the relationship between the original and quadrupled bonus pools -/
theorem quadrupled_bonus_remainder
  (original : BonusPool)
  (h1 : original.employees = 8)
  (h2 : original.remainder = 5)
  (quadrupled : BonusPool)
  (h3 : quadrupled.employees = original.employees)
  (h4 : quadrupled.total = 4 * original.total) :
  quadrupled.remainder = 4 := by
sorry

end NUMINAMATH_CALUDE_quadrupled_bonus_remainder_l2720_272084


namespace NUMINAMATH_CALUDE_expression_equality_l2720_272025

theorem expression_equality (x : ℝ) : 3 * x * (21 - (x + 3) * x - 3) = 54 * x - 3 * x^3 + 9 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2720_272025


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2720_272069

theorem quadratic_equation_roots (a : ℝ) : 
  ((a + 1) * (-1)^2 + (-1) - 1 = 0) → 
  (a = 1 ∧ ∃ x : ℝ, x ≠ -1 ∧ (2 * x^2 + x - 1 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2720_272069


namespace NUMINAMATH_CALUDE_track_length_is_600_l2720_272017

/-- Represents a circular running track with two runners -/
structure CircularTrack where
  length : ℝ
  runner1_speed : ℝ
  runner2_speed : ℝ

/-- The conditions of the problem -/
def track_conditions (t : CircularTrack) : Prop :=
  ∃ (first_meeting second_meeting : ℝ),
    first_meeting > 0 ∧
    second_meeting > first_meeting ∧
    first_meeting * t.runner2_speed = 120 ∧
    (second_meeting - first_meeting) * t.runner1_speed = 180 ∧
    first_meeting * t.runner1_speed + 120 = t.length / 2 ∧
    t.runner1_speed > 0 ∧
    t.runner2_speed > 0

/-- The theorem to be proved -/
theorem track_length_is_600 (t : CircularTrack) :
  track_conditions t → t.length = 600 := by
  sorry

end NUMINAMATH_CALUDE_track_length_is_600_l2720_272017


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2720_272074

theorem geometric_series_sum : 
  let series := [2, 6, 18, 54, 162, 486, 1458, 4374]
  series.sum = 6560 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2720_272074


namespace NUMINAMATH_CALUDE_abc_cba_divisibility_l2720_272073

theorem abc_cba_divisibility (a : ℕ) (h : a ≤ 7) :
  ∃ k : ℕ, 100 * a + 10 * (a + 1) + (a + 2) + 100 * (a + 2) + 10 * (a + 1) + a = 212 * k := by
  sorry

end NUMINAMATH_CALUDE_abc_cba_divisibility_l2720_272073


namespace NUMINAMATH_CALUDE_power_six_times_three_six_l2720_272063

theorem power_six_times_three_six : 6^6 * 3^6 = 34012224 := by
  sorry

end NUMINAMATH_CALUDE_power_six_times_three_six_l2720_272063


namespace NUMINAMATH_CALUDE_value_of_c_l2720_272052

theorem value_of_c (a b c : ℝ) 
  (h1 : a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1))
  (h2 : 6 * 15 * c = 1) : c = 11 := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l2720_272052


namespace NUMINAMATH_CALUDE_percent_of_x_l2720_272050

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 50 + x / 25) / x * 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_l2720_272050


namespace NUMINAMATH_CALUDE_inequality_solution_l2720_272068

theorem inequality_solution (x : ℝ) : 
  (x^3 - 3*x^2 + 2*x) / (x^2 - 3*x + 2) ≤ 0 ↔ x ≤ 0 ∧ x ≠ 1 ∧ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2720_272068


namespace NUMINAMATH_CALUDE_nearest_multiple_of_11_to_457_l2720_272098

theorem nearest_multiple_of_11_to_457 :
  ∃ (n : ℤ), n % 11 = 0 ∧ 
  ∀ (m : ℤ), m % 11 = 0 → |n - 457| ≤ |m - 457| ∧
  n = 462 := by
  sorry

end NUMINAMATH_CALUDE_nearest_multiple_of_11_to_457_l2720_272098


namespace NUMINAMATH_CALUDE_wall_volume_calculation_l2720_272000

/-- Represents the dimensions of a wall -/
structure WallDimensions where
  width : ℝ
  height : ℝ
  length : ℝ

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ := w.width * w.height * w.length

/-- Theorem stating the volume of the wall under given conditions -/
theorem wall_volume_calculation :
  ∃ (w : WallDimensions),
    w.width = 8 ∧
    w.height = 6 * w.width ∧
    w.length = 7 * w.height ∧
    wallVolume w = 128512 := by
  sorry


end NUMINAMATH_CALUDE_wall_volume_calculation_l2720_272000


namespace NUMINAMATH_CALUDE_final_value_is_four_l2720_272010

def program_execution (M : Nat) : Nat :=
  let M1 := M + 1
  let M2 := M1 + 2
  M2

theorem final_value_is_four : program_execution 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_final_value_is_four_l2720_272010


namespace NUMINAMATH_CALUDE_seasonal_work_term_l2720_272046

/-- The established term of work for two seasonal workers -/
theorem seasonal_work_term (a r s : ℝ) (hr : r > 0) (hs : s > r) :
  ∃ x : ℝ, x > 0 ∧
  (x - a) * (s / (x + a)) = (x + a) * (r / (x - a)) ∧
  x = a * (s + r) / (s - r) := by
  sorry

end NUMINAMATH_CALUDE_seasonal_work_term_l2720_272046


namespace NUMINAMATH_CALUDE_symmetry_plane_arrangement_l2720_272061

/-- A symmetry plane of a body. -/
structure SymmetryPlane where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A body with symmetry planes. -/
structure Body where
  symmetry_planes : List SymmetryPlane
  exactly_three_planes : symmetry_planes.length = 3

/-- Angle between two symmetry planes. -/
def angle_between (p1 p2 : SymmetryPlane) : ℝ :=
  sorry

/-- Predicate to check if two planes are perpendicular. -/
def are_perpendicular (p1 p2 : SymmetryPlane) : Prop :=
  angle_between p1 p2 = 90

/-- Predicate to check if two planes intersect at 60 degrees. -/
def intersect_at_60 (p1 p2 : SymmetryPlane) : Prop :=
  angle_between p1 p2 = 60

/-- Theorem stating the possible arrangements of symmetry planes. -/
theorem symmetry_plane_arrangement (b : Body) :
  (∀ (p1 p2 : SymmetryPlane), p1 ∈ b.symmetry_planes → p2 ∈ b.symmetry_planes → p1 ≠ p2 →
    are_perpendicular p1 p2) ∨
  (∀ (p1 p2 : SymmetryPlane), p1 ∈ b.symmetry_planes → p2 ∈ b.symmetry_planes → p1 ≠ p2 →
    intersect_at_60 p1 p2) :=
  sorry

end NUMINAMATH_CALUDE_symmetry_plane_arrangement_l2720_272061


namespace NUMINAMATH_CALUDE_max_value_symmetric_function_l2720_272057

def f (a b x : ℝ) : ℝ := (1 + 2*x) * (x^2 + a*x + b)

theorem max_value_symmetric_function (a b : ℝ) :
  (∀ x : ℝ, f a b (1 - x) = f a b (1 + x)) →
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1 : ℝ) 1 ∧
    ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → f a b x ≤ f a b x₀) →
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a b x₀ = 3 * Real.sqrt 3 / 2 ∧
    ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → f a b x ≤ f a b x₀) :=
by sorry

end NUMINAMATH_CALUDE_max_value_symmetric_function_l2720_272057


namespace NUMINAMATH_CALUDE_library_schedule_lcm_l2720_272040

theorem library_schedule_lcm : Nat.lcm 5 (Nat.lcm 3 (Nat.lcm 9 8)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_library_schedule_lcm_l2720_272040


namespace NUMINAMATH_CALUDE_white_surface_fraction_of_given_cube_l2720_272013

/-- Represents a cube composed of smaller cubes -/
structure CompositeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  white_cube_count : ℕ
  black_cube_count : ℕ

/-- Calculates the fraction of white surface area for a composite cube -/
def white_surface_fraction (c : CompositeCube) : ℚ :=
  -- Implementation details omitted
  0

/-- Theorem stating the fraction of white surface area for the given cube -/
theorem white_surface_fraction_of_given_cube :
  let c : CompositeCube := {
    edge_length := 4,
    small_cube_count := 64,
    white_cube_count := 44,
    black_cube_count := 20
  }
  white_surface_fraction c = 5/6 :=
by sorry

end NUMINAMATH_CALUDE_white_surface_fraction_of_given_cube_l2720_272013


namespace NUMINAMATH_CALUDE_problem1_l2720_272096

theorem problem1 (x y : ℝ) : x^2 * (-2*x*y^2)^3 = -8*x^5*y^6 := by sorry

end NUMINAMATH_CALUDE_problem1_l2720_272096


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l2720_272021

def m : ℕ := 2016^2 + 2^2016

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : 
  m = 2016^2 + 2^2016 → (m^2 + 2^m) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l2720_272021


namespace NUMINAMATH_CALUDE_bookstore_new_releases_l2720_272091

theorem bookstore_new_releases (total_books : ℕ) (total_books_pos : total_books > 0) :
  let historical_fiction := (2 : ℚ) / 5 * total_books
  let other_books := total_books - historical_fiction
  let historical_fiction_new := (2 : ℚ) / 5 * historical_fiction
  let other_new := (1 : ℚ) / 5 * other_books
  let total_new := historical_fiction_new + other_new
  historical_fiction_new / total_new = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_bookstore_new_releases_l2720_272091


namespace NUMINAMATH_CALUDE_inequality_proof_l2720_272090

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x + y ≤ (y^2 / x) + (x^2 / y) := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2720_272090


namespace NUMINAMATH_CALUDE_card_ratio_l2720_272043

/-- Prove that given the conditions in the problem, the ratio of football cards to hockey cards is 4:1 -/
theorem card_ratio (total_cards : ℕ) (hockey_cards : ℕ) (s : ℕ) :
  total_cards = 1750 →
  hockey_cards = 200 →
  total_cards = (s * hockey_cards - 50) + (s * hockey_cards) + hockey_cards →
  (s * hockey_cards) / hockey_cards = 4 :=
by sorry

end NUMINAMATH_CALUDE_card_ratio_l2720_272043


namespace NUMINAMATH_CALUDE_max_value_theorem_l2720_272048

theorem max_value_theorem (x y z : ℝ) (h : 2*x + 3*y + 5*z = 29) :
  Real.sqrt (2*x + 1) + Real.sqrt (3*y + 4) + Real.sqrt (5*z + 6) ≤ 2 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2720_272048


namespace NUMINAMATH_CALUDE_magnitude_of_Z_l2720_272083

theorem magnitude_of_Z (Z : ℂ) (h : (1 - Complex.I) * Z = 1 + Complex.I) : Complex.abs Z = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_Z_l2720_272083


namespace NUMINAMATH_CALUDE_inequality_solution_length_l2720_272018

/-- Given an inequality c ≤ 3x + 5 ≤ d, where the length of the interval of solutions is 15, prove that d - c = 45 -/
theorem inequality_solution_length (c d : ℝ) : 
  (∃ (x : ℝ), c ≤ 3*x + 5 ∧ 3*x + 5 ≤ d) → 
  ((d - 5) / 3 - (c - 5) / 3 = 15) →
  d - c = 45 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_length_l2720_272018


namespace NUMINAMATH_CALUDE_gcd_78_36_l2720_272072

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_78_36_l2720_272072


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l2720_272076

theorem mean_of_remaining_numbers (a b c d : ℝ) :
  (a + b + c + d + 105) / 5 = 92 →
  (a + b + c + d) / 4 = 88.75 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l2720_272076


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2720_272014

/-- The speed of a boat in still water, given its travel distances with and against a stream. -/
theorem boat_speed_in_still_water 
  (along_stream : ℝ) 
  (against_stream : ℝ) 
  (h1 : along_stream = 11) 
  (h2 : against_stream = 7) : 
  ∃ (boat_speed stream_speed : ℝ), 
    boat_speed + stream_speed = along_stream ∧ 
    boat_speed - stream_speed = against_stream ∧ 
    boat_speed = 9 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2720_272014


namespace NUMINAMATH_CALUDE_malcolm_followers_difference_l2720_272024

def malcolm_social_media (instagram_followers facebook_followers : ℕ) : Prop :=
  let twitter_followers := (instagram_followers + facebook_followers) / 2
  let tiktok_followers := 3 * twitter_followers
  ∃ (youtube_followers : ℕ),
    youtube_followers > tiktok_followers ∧
    instagram_followers + facebook_followers + twitter_followers + tiktok_followers + youtube_followers = 3840 ∧
    youtube_followers - tiktok_followers = 510

theorem malcolm_followers_difference :
  malcolm_social_media 240 500 :=
by sorry

end NUMINAMATH_CALUDE_malcolm_followers_difference_l2720_272024


namespace NUMINAMATH_CALUDE_jam_cost_l2720_272053

/-- The cost of jam for N sandwiches given the following conditions:
  * N sandwiches are prepared
  * Each sandwich requires B scoops of peanut butter and J spoonfuls of jam
  * Peanut butter costs 6 cents per scoop
  * Jam costs 7 cents per spoonful
  * Total cost for peanut butter and jam is $3.06
  * B, J, and N are positive integers
  * N > 1
-/
theorem jam_cost (N B J : ℕ) : N > 1 → B > 0 → J > 0 → N * (6 * B + 7 * J) = 306 → N * J * 7 = 238 := by
  sorry

end NUMINAMATH_CALUDE_jam_cost_l2720_272053


namespace NUMINAMATH_CALUDE_square_side_differences_l2720_272023

theorem square_side_differences (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ > a₂) (h₂ : a₂ > a₃) (h₃ : a₃ > a₄)
  (diff₁ : a₁ - a₂ = 11) (diff₂ : a₂ - a₃ = 5) (diff₃ : a₃ - a₄ = 13) :
  a₁ - a₄ = 29 := by
sorry

end NUMINAMATH_CALUDE_square_side_differences_l2720_272023


namespace NUMINAMATH_CALUDE_subset_implies_c_equals_two_l2720_272093

theorem subset_implies_c_equals_two :
  {p : ℝ × ℝ | p.1 + p.2 - 2 = 0 ∧ p.1 - 2*p.2 + 4 = 0} ⊆ {p : ℝ × ℝ | p.2 = 3*p.1 + c} →
  c = 2 :=
by sorry

end NUMINAMATH_CALUDE_subset_implies_c_equals_two_l2720_272093


namespace NUMINAMATH_CALUDE_quadratic_one_root_l2720_272045

theorem quadratic_one_root (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∃! x : ℝ, x^2 + 6*m*x + m - n = 0) →
  (0 < m ∧ m < 1/9 ∧ n = m - 9*m^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l2720_272045


namespace NUMINAMATH_CALUDE_reciprocal_sum_fourths_sixths_l2720_272099

theorem reciprocal_sum_fourths_sixths : (1 / (1/4 + 1/6) : ℚ) = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_fourths_sixths_l2720_272099


namespace NUMINAMATH_CALUDE_first_three_squares_s_3_equals_149_l2720_272077

/-- s(n) is the n-digit number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ :=
  sorry

/-- The first three perfect squares are 1, 4, and 9 -/
theorem first_three_squares : List ℕ := [1, 4, 9]

/-- s(3) is equal to 149 -/
theorem s_3_equals_149 : s 3 = 149 := by
  sorry

end NUMINAMATH_CALUDE_first_three_squares_s_3_equals_149_l2720_272077


namespace NUMINAMATH_CALUDE_base9_addition_theorem_l2720_272086

/-- Converts a base-9 number represented as a list of digits to its decimal (base-10) equivalent -/
def base9ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 9 * acc + d) 0

/-- Converts a decimal (base-10) number to its base-9 representation as a list of digits -/
def decimalToBase9 (n : Nat) : List Nat :=
  if n < 9 then [n]
  else (n % 9) :: decimalToBase9 (n / 9)

/-- The main theorem stating that the sum of the given base-9 numbers equals 1416₉ -/
theorem base9_addition_theorem :
  let a := [3, 4, 6] -- 346₉
  let b := [8, 0, 2] -- 802₉
  let c := [1, 5, 7] -- 157₉
  let result := [1, 4, 1, 6] -- 1416₉
  base9ToDecimal a + base9ToDecimal b + base9ToDecimal c = base9ToDecimal result :=
by sorry


end NUMINAMATH_CALUDE_base9_addition_theorem_l2720_272086


namespace NUMINAMATH_CALUDE_max_cubes_in_box_l2720_272070

/-- The maximum number of cubes that can fit in a rectangular box -/
theorem max_cubes_in_box (box_length box_width box_height cube_volume : ℕ) :
  box_length = 8 →
  box_width = 9 →
  box_height = 12 →
  cube_volume = 27 →
  (box_length * box_width * box_height) / cube_volume = 32 := by
  sorry

#check max_cubes_in_box

end NUMINAMATH_CALUDE_max_cubes_in_box_l2720_272070


namespace NUMINAMATH_CALUDE_min_students_in_both_clubs_l2720_272051

theorem min_students_in_both_clubs 
  (total_students : ℕ) 
  (num_clubs : ℕ) 
  (min_percentage : ℚ) 
  (h1 : total_students = 33) 
  (h2 : num_clubs = 2) 
  (h3 : min_percentage = 7/10) : 
  ∃ (students_in_both : ℕ), 
    students_in_both ≥ 15 ∧ 
    ∀ (n1 n2 : ℕ), 
      n1 ≥ Int.ceil (total_students * min_percentage) → 
      n2 ≥ Int.ceil (total_students * min_percentage) → 
      n1 + n2 - students_in_both ≤ total_students → 
      students_in_both ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_min_students_in_both_clubs_l2720_272051


namespace NUMINAMATH_CALUDE_complex_multiplication_l2720_272036

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2720_272036


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l2720_272008

/-- The function f(x) = x³ - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The function g(x) = x² + a -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The tangent line of f at x = -1 -/
def tangent_line (x : ℝ) : ℝ := 2 * x + 2

theorem tangent_line_intersection (a : ℝ) : 
  (∀ x, tangent_line x = g a x) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l2720_272008


namespace NUMINAMATH_CALUDE_divisor_problem_l2720_272029

theorem divisor_problem (p n : ℕ) (d : Fin 8 → ℕ) : 
  p.Prime → 
  n > 0 →
  (∀ i : Fin 8, d i > 0) →
  (∀ i j : Fin 8, i < j → d i < d j) →
  d 0 = 1 →
  d 7 = p * n →
  (∀ x : ℕ, x ∣ (p * n) ↔ ∃ i : Fin 8, d i = x) →
  d (⟨17 * p - d 2, sorry⟩ : Fin 8) = (d 0 + d 1 + d 2) * (d 2 + d 3 + 13 * p) →
  n = 2021 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l2720_272029


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2720_272016

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {0, 2, 3}
  A ∩ B = {2, 3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2720_272016


namespace NUMINAMATH_CALUDE_probability_12th_roll_last_proof_l2720_272030

/-- The probability of the 12th roll being the last roll when rolling a standard 
    eight-sided die until getting the same number on consecutive rolls -/
def probability_12th_roll_last : ℚ :=
  (7^10 : ℚ) / (8^11 : ℚ)

/-- The number of sides on the standard die -/
def num_sides : ℕ := 8

/-- The number of rolls -/
def num_rolls : ℕ := 12

theorem probability_12th_roll_last_proof :
  probability_12th_roll_last = (7^(num_rolls - 2) : ℚ) / (num_sides^(num_rolls - 1) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_probability_12th_roll_last_proof_l2720_272030


namespace NUMINAMATH_CALUDE_shower_tiles_count_l2720_272005

/-- Calculates the total number of tiles in a 3-sided shower --/
def shower_tiles (sides : ℕ) (width : ℕ) (height : ℕ) : ℕ :=
  sides * width * height

/-- Theorem: The total number of tiles in a 3-sided shower with 8 tiles in width and 20 tiles in height is 480 --/
theorem shower_tiles_count : shower_tiles 3 8 20 = 480 := by
  sorry

end NUMINAMATH_CALUDE_shower_tiles_count_l2720_272005


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2720_272075

theorem inequality_solution_range (b : ℝ) : 
  (∀ x : ℤ, |3 * (x : ℝ) - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) →
  (5 < b ∧ b < 7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2720_272075


namespace NUMINAMATH_CALUDE_goals_tied_in_june_l2720_272058

def ronaldo_goals : List Nat := [2, 9, 14, 8, 7, 11, 12]
def messi_goals : List Nat := [5, 8, 18, 6, 10, 9, 9]

def cumulative_sum (xs : List Nat) : List Nat :=
  List.scanl (·+·) 0 xs

def first_equal_index (xs ys : List Nat) : Option Nat :=
  (List.zip xs ys).findIdx (fun (x, y) => x = y)

def months : List String := ["January", "February", "March", "April", "May", "June", "July"]

theorem goals_tied_in_june :
  first_equal_index (cumulative_sum ronaldo_goals) (cumulative_sum messi_goals) = some 5 :=
by sorry

end NUMINAMATH_CALUDE_goals_tied_in_june_l2720_272058


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2720_272059

theorem quadratic_inequality_condition (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 2 > 0) ↔ -2*Real.sqrt 2 < m ∧ m < 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2720_272059
