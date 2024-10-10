import Mathlib

namespace reciprocal_of_sum_l3764_376421

theorem reciprocal_of_sum : (1 / (1/2 + 1/3) : ℚ) = 6/5 := by sorry

end reciprocal_of_sum_l3764_376421


namespace length_of_AB_l3764_376438

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 3)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ 
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_AB : 
  ∀ A B : ℝ × ℝ, intersection_points A B → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 :=
sorry

end length_of_AB_l3764_376438


namespace arithmetic_calculation_l3764_376448

theorem arithmetic_calculation : 5 * (7 + 3) - 10 * 2 + 36 / 3 = 42 := by
  sorry

end arithmetic_calculation_l3764_376448


namespace power_division_nineteen_l3764_376497

theorem power_division_nineteen : (19 : ℕ)^11 / (19 : ℕ)^8 = 6859 := by sorry

end power_division_nineteen_l3764_376497


namespace divisibility_by_9_52B7_l3764_376412

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def number_52B7 (B : ℕ) : ℕ := 5000 + 200 + B * 10 + 7

theorem divisibility_by_9_52B7 :
  ∀ B : ℕ, B < 10 → (is_divisible_by_9 (number_52B7 B) ↔ B = 4) := by sorry

end divisibility_by_9_52B7_l3764_376412


namespace min_value_sqrt_sum_squares_l3764_376455

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 3) 
  (h2 : m*a + n*b = 3) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 3 := by
  sorry

end min_value_sqrt_sum_squares_l3764_376455


namespace exists_periodic_nonconstant_sequence_l3764_376472

def isPeriodicSequence (x : ℕ → ℤ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ ∀ n, x (n + T) = x n

def satisfiesRecurrence (x : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → x (n + 1) = 2 * x n + 3 * x (n - 1)

def isConstant (x : ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, x m = x n

theorem exists_periodic_nonconstant_sequence :
  ∃ x : ℕ → ℤ, satisfiesRecurrence x ∧ isPeriodicSequence x ∧ ¬isConstant x := by
  sorry

end exists_periodic_nonconstant_sequence_l3764_376472


namespace min_S6_arithmetic_sequence_l3764_376456

/-- Given an arithmetic sequence with common ratio q > 1, where S_n denotes the sum of first n terms,
    and S_4 = 2S_2 + 1, the minimum value of S_6 is 2√3 + 3. -/
theorem min_S6_arithmetic_sequence (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  q > 1 →
  (∀ n, a (n + 1) = a n + q) →
  (∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * q) / 2) →
  S 4 = 2 * S 2 + 1 →
  (∀ s : ℝ, s = S 6 → s ≥ 2 * Real.sqrt 3 + 3) ∧
  ∃ s : ℝ, s = S 6 ∧ s = 2 * Real.sqrt 3 + 3 :=
by sorry


end min_S6_arithmetic_sequence_l3764_376456


namespace magic_square_x_value_l3764_376406

/-- Represents a 3x3 multiplicative magic square --/
structure MagicSquare where
  a11 : ℝ
  a12 : ℝ
  a13 : ℝ
  a21 : ℝ
  a22 : ℝ
  a23 : ℝ
  a31 : ℝ
  a32 : ℝ
  a33 : ℝ
  positive : ∀ i j, (i, j) ∈ [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)] → 
    match (i, j) with
    | (1, 1) => a11 > 0
    | (1, 2) => a12 > 0
    | (1, 3) => a13 > 0
    | (2, 1) => a21 > 0
    | (2, 2) => a22 > 0
    | (2, 3) => a23 > 0
    | (3, 1) => a31 > 0
    | (3, 2) => a32 > 0
    | (3, 3) => a33 > 0
    | _ => False
  magic : a11 * a12 * a13 = a21 * a22 * a23 ∧
          a11 * a12 * a13 = a31 * a32 * a33 ∧
          a11 * a12 * a13 = a11 * a21 * a31 ∧
          a11 * a12 * a13 = a12 * a22 * a32 ∧
          a11 * a12 * a13 = a13 * a23 * a33 ∧
          a11 * a12 * a13 = a11 * a22 * a33 ∧
          a11 * a12 * a13 = a13 * a22 * a31

theorem magic_square_x_value (ms : MagicSquare) 
  (h1 : ms.a11 = 5)
  (h2 : ms.a21 = 4)
  (h3 : ms.a33 = 20) :
  ms.a12 = 100 := by
  sorry

end magic_square_x_value_l3764_376406


namespace no_real_solutions_l3764_376496

theorem no_real_solutions :
  ¬∃ y : ℝ, (3 * y - 4)^2 + 4 = -(y + 3) := by
sorry

end no_real_solutions_l3764_376496


namespace a_3_equals_negative_8_l3764_376400

/-- The sum of the first n terms of a geometric sequence -/
def S (n : ℕ) (x : ℝ) : ℝ := (x^2 + 3*x)*2^n - x + 1

/-- The n-th term of the geometric sequence -/
def a (n : ℕ) (x : ℝ) : ℝ :=
  if n = 1 then S 1 x
  else S n x - S (n-1) x

/-- The common ratio of the geometric sequence -/
def q : ℝ := 2

/-- The value of x that satisfies the given condition -/
def x : ℝ := -1

theorem a_3_equals_negative_8 : a 3 x = -8 := by sorry

end a_3_equals_negative_8_l3764_376400


namespace second_cube_volume_is_64_l3764_376471

-- Define the volume of the first cube
def first_cube_volume : ℝ := 8

-- Define the relationship between the surface areas of the two cubes
def surface_area_ratio : ℝ := 4

-- Theorem statement
theorem second_cube_volume_is_64 :
  let first_side := first_cube_volume ^ (1/3 : ℝ)
  let first_surface_area := 6 * first_side^2
  let second_surface_area := surface_area_ratio * first_surface_area
  let second_side := (second_surface_area / 6) ^ (1/2 : ℝ)
  second_side^3 = 64 := by sorry

end second_cube_volume_is_64_l3764_376471


namespace monotone_cubic_implies_nonneg_a_l3764_376411

/-- A function f : ℝ → ℝ is monotonically increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The cubic function with parameter a -/
def f (a : ℝ) : ℝ → ℝ := λ x => x^3 + a*x

theorem monotone_cubic_implies_nonneg_a :
  ∀ a : ℝ, MonotonicallyIncreasing (f a) → a ≥ 0 :=
by
  sorry

end monotone_cubic_implies_nonneg_a_l3764_376411


namespace velocity_of_point_C_l3764_376428

/-- Given the equation relating distances and time, prove the velocity of point C. -/
theorem velocity_of_point_C 
  (a R L T : ℝ) 
  (x : ℝ) 
  (h : (a * T) / (a * T - R) = (L + x) / x) :
  (x / T) = a * L / R :=
sorry

end velocity_of_point_C_l3764_376428


namespace oxygen_atoms_count_l3764_376446

-- Define the atomic weights
def carbon_weight : ℕ := 12
def hydrogen_weight : ℕ := 1
def oxygen_weight : ℕ := 16

-- Define the number of Carbon and Hydrogen atoms
def carbon_atoms : ℕ := 2
def hydrogen_atoms : ℕ := 4

-- Define the total molecular weight of the compound
def total_weight : ℕ := 60

-- Theorem to prove
theorem oxygen_atoms_count :
  let carbon_hydrogen_weight := carbon_atoms * carbon_weight + hydrogen_atoms * hydrogen_weight
  let oxygen_weight_total := total_weight - carbon_hydrogen_weight
  oxygen_weight_total / oxygen_weight = 2 := by
  sorry

end oxygen_atoms_count_l3764_376446


namespace quilt_sewing_percentage_l3764_376401

theorem quilt_sewing_percentage (total_squares : ℕ) (squares_left : ℕ) : 
  total_squares = 32 → squares_left = 24 → 
  (total_squares - squares_left : ℚ) / total_squares * 100 = 25 := by
sorry

end quilt_sewing_percentage_l3764_376401


namespace raphael_manny_ratio_l3764_376452

/-- Represents the number of lasagna pieces each person eats -/
structure LasagnaPieces where
  manny : ℕ
  lisa : ℕ
  raphael : ℕ
  aaron : ℕ
  kai : ℕ

/-- The properties of the lasagna distribution -/
def LasagnaDistribution (p : LasagnaPieces) : Prop :=
  p.manny = 1 ∧
  p.aaron = 0 ∧
  p.kai = 2 * p.manny ∧
  p.lisa = 2 + (p.raphael - 1) ∧
  p.manny + p.lisa + p.raphael + p.aaron + p.kai = 6

theorem raphael_manny_ratio (p : LasagnaPieces) 
  (h : LasagnaDistribution p) : p.raphael = p.manny := by
  sorry

end raphael_manny_ratio_l3764_376452


namespace sin_cos_range_l3764_376486

open Real

theorem sin_cos_range :
  ∀ y : ℝ, (∃ x : ℝ, sin x + cos x = y) ↔ -Real.sqrt 2 ≤ y ∧ y ≤ Real.sqrt 2 :=
by sorry

end sin_cos_range_l3764_376486


namespace sin_210_degrees_l3764_376431

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end sin_210_degrees_l3764_376431


namespace quadratic_condition_for_x_equals_one_l3764_376415

theorem quadratic_condition_for_x_equals_one :
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧
  ¬(∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) :=
by sorry

end quadratic_condition_for_x_equals_one_l3764_376415


namespace average_goat_price_l3764_376435

/-- Given the number of goats and hens, their total cost, and the average cost of a hen,
    calculate the average cost of a goat. -/
theorem average_goat_price
  (num_goats : ℕ)
  (num_hens : ℕ)
  (total_cost : ℕ)
  (avg_hen_price : ℕ)
  (h1 : num_goats = 5)
  (h2 : num_hens = 10)
  (h3 : total_cost = 2500)
  (h4 : avg_hen_price = 50) :
  (total_cost - num_hens * avg_hen_price) / num_goats = 400 := by
  sorry

#check average_goat_price

end average_goat_price_l3764_376435


namespace triangle_tangent_circles_intersection_l3764_376403

/-- Triangle ABC with side lengths AB=8, BC=9, CA=10 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB_length : dist A B = 8)
  (BC_length : dist B C = 9)
  (CA_length : dist C A = 10)

/-- Circle passing through a point and tangent to a line at another point -/
structure TangentCircle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (passes_through : ℝ × ℝ)
  (tangent_point : ℝ × ℝ)
  (tangent_line : ℝ × ℝ → ℝ × ℝ → Prop)

/-- The intersection point of two circles -/
def CircleIntersection (ω₁ ω₂ : TangentCircle) : ℝ × ℝ := sorry

/-- The theorem to be proved -/
theorem triangle_tangent_circles_intersection
  (abc : Triangle)
  (ω₁ : TangentCircle)
  (ω₂ : TangentCircle)
  (h₁ : ω₁.passes_through = abc.B ∧ ω₁.tangent_point = abc.A ∧ ω₁.tangent_line abc.A abc.C)
  (h₂ : ω₂.passes_through = abc.C ∧ ω₂.tangent_point = abc.A ∧ ω₂.tangent_line abc.A abc.B)
  (K : ℝ × ℝ)
  (hK : K = CircleIntersection ω₁ ω₂ ∧ K ≠ abc.A) :
  dist abc.A K = 10 * Real.sqrt 3 / 3 := by
  sorry

end triangle_tangent_circles_intersection_l3764_376403


namespace swimmers_speed_l3764_376405

/-- Proves that a person's swimming speed in still water is 4 km/h given the conditions -/
theorem swimmers_speed (water_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : water_speed = 2)
  (h2 : distance = 16) (h3 : time = 8) : ∃ v : ℝ, v = 4 ∧ distance = (v - water_speed) * time :=
by
  sorry

end swimmers_speed_l3764_376405


namespace min_S_19_l3764_376451

/-- An arithmetic sequence with its sum properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2
  arithmetic : ∀ n m, a (n + m) - a n = m * (a 2 - a 1)

/-- The minimum value of S_19 given the conditions -/
theorem min_S_19 (seq : ArithmeticSequence) 
  (h1 : seq.S 8 ≤ 6) (h2 : seq.S 11 ≥ 27) : 
  seq.S 19 ≥ 133 := by
  sorry

#check min_S_19

end min_S_19_l3764_376451


namespace power_of_power_l3764_376484

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end power_of_power_l3764_376484


namespace honey_production_optimal_tax_revenue_optimal_l3764_376433

/-- The inverse demand function for honey -/
def inverse_demand (Q : ℝ) : ℝ := 310 - 3 * Q

/-- The production cost per jar of honey -/
def production_cost : ℝ := 10

/-- The profit function without tax -/
def profit (Q : ℝ) : ℝ := (inverse_demand Q) * Q - production_cost * Q

/-- The profit function with tax -/
def profit_with_tax (Q t : ℝ) : ℝ := (inverse_demand Q) * Q - production_cost * Q - t * Q

/-- The tax revenue function -/
def tax_revenue (Q t : ℝ) : ℝ := Q * t

theorem honey_production_optimal (Q : ℝ) :
  profit Q ≤ profit 50 := by sorry

theorem tax_revenue_optimal (t : ℝ) :
  tax_revenue ((310 - t) / 6) t ≤ tax_revenue ((310 - 150) / 6) 150 := by sorry

end honey_production_optimal_tax_revenue_optimal_l3764_376433


namespace inequality_solution_set_l3764_376408

theorem inequality_solution_set (x : ℝ) :
  (x^2 + 1) / ((x - 3) * (x + 2)) ≥ 0 ↔ x ≤ -2 ∨ x ≥ 3 :=
by sorry

end inequality_solution_set_l3764_376408


namespace shoe_rebate_problem_l3764_376423

/-- Calculates the total rebate and quantity discount for a set of shoe purchases --/
def calculate_rebate_and_discount (prices : List ℝ) (rebate_percentages : List ℝ) 
  (discount_threshold_1 : ℝ) (discount_threshold_2 : ℝ) 
  (discount_rate_1 : ℝ) (discount_rate_2 : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct rebate and discount for the given problem --/
theorem shoe_rebate_problem :
  let prices := [28, 35, 40, 45, 50]
  let rebate_percentages := [10, 12, 15, 18, 20]
  let discount_threshold_1 := 200
  let discount_threshold_2 := 250
  let discount_rate_1 := 5
  let discount_rate_2 := 7
  let (total_rebate, quantity_discount) := 
    calculate_rebate_and_discount prices rebate_percentages 
      discount_threshold_1 discount_threshold_2 
      discount_rate_1 discount_rate_2
  total_rebate = 31.1 ∧ quantity_discount = 0 := by
  sorry

end shoe_rebate_problem_l3764_376423


namespace final_staff_count_l3764_376475

/- Define the initial number of staff in each category -/
def initial_doctors : ℕ := 11
def initial_nurses : ℕ := 18
def initial_assistants : ℕ := 9
def initial_interns : ℕ := 6

/- Define the number of staff who quit or are transferred -/
def doctors_quit : ℕ := 5
def nurses_quit : ℕ := 2
def assistants_quit : ℕ := 3
def nurses_transferred : ℕ := 2
def interns_transferred : ℕ := 4

/- Define the number of staff on leave -/
def doctors_on_leave : ℕ := 4
def nurses_on_leave : ℕ := 3

/- Define the number of new staff joining -/
def new_doctors : ℕ := 3
def new_nurses : ℕ := 5

/- Theorem to prove the final staff count -/
theorem final_staff_count :
  (initial_doctors - doctors_quit - doctors_on_leave + new_doctors) +
  (initial_nurses - nurses_quit - nurses_transferred - nurses_on_leave + new_nurses) +
  (initial_assistants - assistants_quit) +
  (initial_interns - interns_transferred) = 29 := by
  sorry

end final_staff_count_l3764_376475


namespace sequence_periodicity_l3764_376488

def is_periodic (a : ℕ → ℤ) : Prop :=
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, a (n + p) = a n

theorem sequence_periodicity (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, n ≥ 2 → (0 : ℝ) ≤ a (n - 1) + ((1 - Real.sqrt 5) / 2) * (a n) + a (n + 1) ∧
                       a (n - 1) + ((1 - Real.sqrt 5) / 2) * (a n) + a (n + 1) < 1) :
  is_periodic a :=
sorry

end sequence_periodicity_l3764_376488


namespace relationship_abc_l3764_376409

theorem relationship_abc : ∀ (a b c : ℕ),
  a = 2^12 → b = 3^8 → c = 7^4 → b > a ∧ a > c := by
  sorry

end relationship_abc_l3764_376409


namespace good_apples_count_l3764_376481

theorem good_apples_count (total_apples unripe_apples : ℕ) 
  (h1 : total_apples = 14) 
  (h2 : unripe_apples = 6) : 
  total_apples - unripe_apples = 8 := by
sorry

end good_apples_count_l3764_376481


namespace total_cost_equals_12_46_l3764_376440

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 249/100

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 187/100

/-- The number of sandwiches -/
def num_sandwiches : ℕ := 2

/-- The number of sodas -/
def num_sodas : ℕ := 4

/-- The total cost of the order -/
def total_cost : ℚ := num_sandwiches * sandwich_cost + num_sodas * soda_cost

theorem total_cost_equals_12_46 : total_cost = 1246/100 := by
  sorry

end total_cost_equals_12_46_l3764_376440


namespace right_triangle_sets_l3764_376410

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  (is_right_triangle 1 2 (Real.sqrt 5)) ∧
  (is_right_triangle (Real.sqrt 2) (Real.sqrt 2) 2) ∧
  (is_right_triangle 13 12 5) ∧
  ¬(is_right_triangle 1 3 (Real.sqrt 7)) := by sorry

end right_triangle_sets_l3764_376410


namespace trigonometric_identities_l3764_376441

theorem trigonometric_identities (α : ℝ) (h1 : α ∈ Set.Ioo (π/2) π) (h2 : Real.sin α = 3/5) :
  (Real.tan (α - π/4) = -7) ∧
  ((Real.sin (2*α) - Real.cos α) / (1 + Real.cos (2*α)) = -1/8) :=
by sorry

end trigonometric_identities_l3764_376441


namespace rectangle_length_proof_l3764_376498

theorem rectangle_length_proof (width : ℝ) (small_area : ℝ) : 
  width = 20 → small_area = 200 → ∃ (length : ℝ), 
    length = 40 ∧ 
    (length / 2) * (width / 2) = small_area := by
  sorry

end rectangle_length_proof_l3764_376498


namespace seed_germination_probability_l3764_376434

/-- The probability of success in a single trial -/
def p : ℝ := 0.9

/-- The probability of failure in a single trial -/
def q : ℝ := 1 - p

/-- The number of trials -/
def n : ℕ := 4

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Probability of exactly k successes in n trials -/
def P (k : ℕ) : ℝ := (binomial n k : ℝ) * p^k * q^(n - k)

theorem seed_germination_probability :
  (P 3 = 0.2916) ∧ (P 3 + P 4 = 0.9477) := by
  sorry

end seed_germination_probability_l3764_376434


namespace share_difference_l3764_376443

/-- Represents the distribution of money among three people -/
structure MoneyDistribution where
  total : ℕ
  ratio_faruk : ℕ
  ratio_vasim : ℕ
  ratio_ranjith : ℕ

/-- Calculates the share of a person given their ratio and the total amount -/
def calculate_share (dist : MoneyDistribution) (ratio : ℕ) : ℕ :=
  dist.total * ratio / (dist.ratio_faruk + dist.ratio_vasim + dist.ratio_ranjith)

theorem share_difference (dist : MoneyDistribution) 
  (h1 : dist.ratio_faruk = 3)
  (h2 : dist.ratio_vasim = 5)
  (h3 : dist.ratio_ranjith = 9)
  (h4 : calculate_share dist dist.ratio_vasim = 1500) :
  calculate_share dist dist.ratio_ranjith - calculate_share dist dist.ratio_faruk = 1800 :=
by sorry

end share_difference_l3764_376443


namespace divisibility_in_sequence_l3764_376453

theorem divisibility_in_sequence (a : ℕ → ℕ) 
  (h : ∀ n ∈ Finset.range 3029, 2 * a (n + 2) = a (n + 1) + 4 * a n) :
  ∃ i ∈ Finset.range 3031, 2^2020 ∣ a i := by
sorry

end divisibility_in_sequence_l3764_376453


namespace mollys_age_l3764_376470

theorem mollys_age (sandy_age molly_age : ℕ) : 
  sandy_age = 42 → 
  sandy_age * 9 = molly_age * 7 → 
  molly_age = 54 := by
sorry

end mollys_age_l3764_376470


namespace complement_event_probability_formula_l3764_376432

/-- The probability of the complement event Ā occurring k times in n trials, 
    given that the probability of event A is p -/
def complementEventProbability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * (1 - p) ^ k * p ^ (n - k)

/-- Theorem stating that the probability of the complement event Ā occurring k times 
    in n trials is equal to ⁽ᵏⁿ)(1-p)ᵏp⁽ⁿ⁻ᵏ⁾, given that the probability of event A is p -/
theorem complement_event_probability_formula (n k : ℕ) (p : ℝ) 
    (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : k ≤ n) : 
  complementEventProbability n k p = Nat.choose n k * (1 - p) ^ k * p ^ (n - k) := by
  sorry

#check complement_event_probability_formula

end complement_event_probability_formula_l3764_376432


namespace polynomial_non_negative_l3764_376464

theorem polynomial_non_negative (x : ℝ) : x^4 - x^3 + 3*x^2 - 2*x + 2 ≥ 0 := by
  sorry

end polynomial_non_negative_l3764_376464


namespace monotone_decreasing_implies_a_bound_l3764_376490

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem monotone_decreasing_implies_a_bound (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y ≤ 4 → f a x > f a y) → a ≤ -3 := by
  sorry

end monotone_decreasing_implies_a_bound_l3764_376490


namespace anthony_final_pet_count_l3764_376492

/-- The number of pets Anthony has after a series of events --/
def final_pet_count (initial_pets : ℕ) : ℕ :=
  let pets_after_loss := initial_pets - (initial_pets * 12 / 100)
  let pets_after_contest := pets_after_loss + 7
  let pets_giving_birth := pets_after_contest / 4
  let new_offspring := pets_giving_birth * 2
  let pets_before_deaths := pets_after_contest + new_offspring
  pets_before_deaths - (pets_before_deaths / 10)

/-- Theorem stating that Anthony ends up with 62 pets --/
theorem anthony_final_pet_count :
  final_pet_count 45 = 62 := by
  sorry

end anthony_final_pet_count_l3764_376492


namespace largest_of_five_consecutive_odd_integers_with_product_110895_l3764_376426

-- Define a function that generates five consecutive odd integers
def fiveConsecutiveOddIntegers (x : ℤ) : List ℤ :=
  [x - 4, x - 2, x, x + 2, x + 4]

-- Theorem statement
theorem largest_of_five_consecutive_odd_integers_with_product_110895 :
  ∃ x : ℤ, 
    (fiveConsecutiveOddIntegers x).prod = 110895 ∧
    (fiveConsecutiveOddIntegers x).all (λ i => i % 2 ≠ 0) ∧
    (fiveConsecutiveOddIntegers x).maximum? = some 17 :=
by sorry

end largest_of_five_consecutive_odd_integers_with_product_110895_l3764_376426


namespace group_transfer_equation_l3764_376460

/-- 
Given two groups of people, with 22 in the first group and 26 in the second group,
this theorem proves the equation for the number of people that should be transferred
from the second group to the first group so that the first group has twice the number
of people as the second group.
-/
theorem group_transfer_equation (x : ℤ) : (22 + x = 2 * (26 - x)) ↔ 
  (22 + x = 2 * (26 - x) ∧ 
   22 + x > 0 ∧ 
   26 - x > 0) := by
  sorry

end group_transfer_equation_l3764_376460


namespace loaf_has_twelve_slices_l3764_376420

/-- Represents a household with bread consumption patterns. -/
structure Household where
  members : ℕ
  breakfast_slices : ℕ
  snack_slices : ℕ
  loaves : ℕ
  days : ℕ

/-- Calculates the number of slices in a loaf of bread for a given household. -/
def slices_per_loaf (h : Household) : ℕ :=
  (h.members * (h.breakfast_slices + h.snack_slices) * h.days) / h.loaves

/-- Theorem stating that for the given household, a loaf of bread contains 12 slices. -/
theorem loaf_has_twelve_slices : 
  slices_per_loaf { members := 4, breakfast_slices := 3, snack_slices := 2, loaves := 5, days := 3 } = 12 := by
  sorry

end loaf_has_twelve_slices_l3764_376420


namespace undefined_slopes_parallel_l3764_376476

-- Define a type for lines
structure Line where
  slope : Option ℝ
  -- Other properties of a line could be added here if needed

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop :=
  (l1.slope = none ∧ l2.slope = none) ∨ (l1.slope ≠ none ∧ l2.slope ≠ none ∧ l1.slope = l2.slope)

-- Define what it means for two lines to be distinct
def distinct (l1 l2 : Line) : Prop :=
  l1 ≠ l2

-- Theorem statement
theorem undefined_slopes_parallel (l1 l2 : Line) :
  distinct l1 l2 → l1.slope = none → l2.slope = none → parallel l1 l2 :=
by
  sorry


end undefined_slopes_parallel_l3764_376476


namespace u_v_cube_sum_l3764_376467

theorem u_v_cube_sum (u v : ℝ) (hu : u > 1) (hv : v > 1)
  (h : Real.log u / Real.log 4 ^ 3 + Real.log v / Real.log 5 ^ 3 + 9 = 
       9 * (Real.log u / Real.log 4) * (Real.log v / Real.log 5)) :
  u^3 + v^3 = 4^(9/2) + 5^(9/2) := by
sorry

end u_v_cube_sum_l3764_376467


namespace range_of_a_for_positive_solutions_l3764_376407

theorem range_of_a_for_positive_solutions (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (1/4)^x + (1/2)^(x-1) + a = 0) ↔ -3 < a ∧ a < 0 := by
  sorry

end range_of_a_for_positive_solutions_l3764_376407


namespace equal_area_triangles_l3764_376442

noncomputable def triangle_area (a b : ℝ) (θ : ℝ) : ℝ := (1/2) * a * b * Real.sin θ

theorem equal_area_triangles (AB AC AD : ℝ) (θ : ℝ) (AE : ℝ) : 
  AB = 4 →
  AC = 5 →
  AD = 2.5 →
  θ = Real.pi / 3 →
  triangle_area AB AC θ = triangle_area AD AE θ →
  AE = 8 :=
by sorry

end equal_area_triangles_l3764_376442


namespace equation_condition_l3764_376436

theorem equation_condition (a d e : ℕ) : 
  (0 < a ∧ a < 10) → (0 < d ∧ d < 10) → (0 < e ∧ e < 10) →
  ((10 * a + d) * (10 * a + e) = 100 * a^2 + 110 * a + d * e ↔ d + e = 11) :=
by sorry

end equation_condition_l3764_376436


namespace fixed_point_on_symmetric_line_l3764_376454

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define symmetry about a point
def symmetric_about (l1 l2 : Line) (p : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ), (y = l1.slope * (x - 4)) →
    ∃ (x' y' : ℝ), (y' = l2.slope * x' + l2.intercept) ∧
      (x + x') / 2 = p.1 ∧ (y + y') / 2 = p.2

-- Define a point being on a line
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

-- Theorem statement
theorem fixed_point_on_symmetric_line (k : ℝ) :
  ∀ (l2 : Line), symmetric_about ⟨k, -4*k⟩ l2 (2, 1) →
    point_on_line (0, 2) l2 := by sorry

end fixed_point_on_symmetric_line_l3764_376454


namespace cube_face_sum_l3764_376437

/-- Represents the six positive integers on the faces of a cube -/
structure CubeFaces where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- The sum of the products of the three numbers adjacent to each vertex -/
def vertexSum (faces : CubeFaces) : ℕ :=
  faces.a * faces.b * faces.c +
  faces.a * faces.e * faces.c +
  faces.a * faces.b * faces.f +
  faces.a * faces.e * faces.f +
  faces.d * faces.b * faces.c +
  faces.d * faces.e * faces.c +
  faces.d * faces.b * faces.f +
  faces.d * faces.e * faces.f

/-- The sum of the numbers on the faces -/
def faceSum (faces : CubeFaces) : ℕ :=
  faces.a + faces.b + faces.c + faces.d + faces.e + faces.f

theorem cube_face_sum (faces : CubeFaces) :
  vertexSum faces = 1386 → faceSum faces = 38 := by
  sorry

end cube_face_sum_l3764_376437


namespace winwin_processing_fee_l3764_376479

/-- Calculates the processing fee for a lottery win -/
def processing_fee (total_win : ℝ) (tax_rate : ℝ) (take_home : ℝ) : ℝ :=
  total_win * (1 - tax_rate) - take_home

/-- Theorem: The processing fee for Winwin's lottery win is $5 -/
theorem winwin_processing_fee :
  processing_fee 50 0.2 35 = 5 := by
  sorry

end winwin_processing_fee_l3764_376479


namespace two_person_subcommittees_from_eight_l3764_376424

theorem two_person_subcommittees_from_eight (n : ℕ) (k : ℕ) : 
  n = 8 → k = 2 → Nat.choose n k = 28 := by
  sorry

end two_person_subcommittees_from_eight_l3764_376424


namespace expand_expression_l3764_376489

theorem expand_expression (x y z : ℝ) : 
  (x + 10) * (3 * y + 5 * z + 15) = 3 * x * y + 5 * x * z + 15 * x + 30 * y + 50 * z + 150 := by
  sorry

end expand_expression_l3764_376489


namespace benjamin_egg_collection_l3764_376491

/-- Proves that Benjamin collects 6 dozen eggs a day given the conditions of the problem -/
theorem benjamin_egg_collection :
  ∀ (benjamin_eggs : ℕ),
  (∃ (carla_eggs trisha_eggs : ℕ),
    carla_eggs = 3 * benjamin_eggs ∧
    trisha_eggs = benjamin_eggs - 4 ∧
    benjamin_eggs + carla_eggs + trisha_eggs = 26) →
  benjamin_eggs = 6 := by
sorry

end benjamin_egg_collection_l3764_376491


namespace inequality_proof_l3764_376439

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := by
  sorry

end inequality_proof_l3764_376439


namespace cab_delay_l3764_376474

theorem cab_delay (usual_time : ℝ) (speed_ratio : ℝ) (h1 : usual_time = 25) (h2 : speed_ratio = 5/6) :
  (usual_time / speed_ratio) - usual_time = 5 :=
by sorry

end cab_delay_l3764_376474


namespace factorial_product_not_perfect_square_l3764_376465

theorem factorial_product_not_perfect_square (n : ℕ) (hn : n ≥ 100) :
  ¬ ∃ m : ℕ, n.factorial * (n + 1).factorial = m ^ 2 := by
  sorry

end factorial_product_not_perfect_square_l3764_376465


namespace sphere_volume_surface_area_equality_l3764_376458

theorem sphere_volume_surface_area_equality (r : ℝ) (h : r > 0) :
  (4 / 3 : ℝ) * Real.pi * r^3 = 36 * Real.pi → 4 * Real.pi * r^2 = 36 * Real.pi :=
by sorry

end sphere_volume_surface_area_equality_l3764_376458


namespace cube_root_of_64_l3764_376487

theorem cube_root_of_64 : (64 : ℝ) ^ (1/3) = 4 := by
  sorry

end cube_root_of_64_l3764_376487


namespace symmetrical_line_over_x_axis_l3764_376422

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

/-- Reflects a line over the X axis --/
def reflect_over_x_axis (l : Line) : Line :=
  { a := l.a,
    b := -l.b,
    c := -l.c,
    eq := sorry }

theorem symmetrical_line_over_x_axis :
  let original_line : Line := { a := 1, b := -2, c := 3, eq := sorry }
  let reflected_line := reflect_over_x_axis original_line
  reflected_line.a = 1 ∧ reflected_line.b = 2 ∧ reflected_line.c = -3 :=
by sorry

end symmetrical_line_over_x_axis_l3764_376422


namespace parallelogram_side_length_l3764_376477

-- Define a parallelogram
structure Parallelogram :=
  (A B C D : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem parallelogram_side_length 
  (ABCD : Parallelogram) 
  (x : ℝ) 
  (h1 : length ABCD.A ABCD.B = x + 3)
  (h2 : length ABCD.B ABCD.C = x - 4)
  (h3 : length ABCD.C ABCD.D = 16) :
  length ABCD.A ABCD.D = 9 := by sorry

end parallelogram_side_length_l3764_376477


namespace toothpick_grid_15_8_l3764_376402

/-- Calculates the number of toothpicks needed for a rectangular grid with diagonals -/
def toothpick_count (height width : ℕ) : ℕ :=
  let horizontal := (height + 1) * width
  let vertical := (width + 1) * height
  let diagonal := height * width
  horizontal + vertical + diagonal

/-- Theorem stating the correct number of toothpicks for a 15x8 grid with diagonals -/
theorem toothpick_grid_15_8 :
  toothpick_count 15 8 = 383 := by
  sorry

end toothpick_grid_15_8_l3764_376402


namespace difference_of_hypotenuse_numbers_l3764_376483

/-- A hypotenuse number is a natural number that can be represented as the sum of two squares of non-negative integers. -/
def is_hypotenuse (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

/-- Any natural number greater than 10 can be represented as the difference of two hypotenuse numbers. -/
theorem difference_of_hypotenuse_numbers (n : ℕ) (h : n > 10) :
  ∃ m₁ m₂ : ℕ, is_hypotenuse m₁ ∧ is_hypotenuse m₂ ∧ n = m₁ - m₂ :=
sorry

end difference_of_hypotenuse_numbers_l3764_376483


namespace white_square_area_l3764_376499

-- Define the cube's edge length
def cube_edge : ℝ := 15

-- Define the total area of blue paint
def blue_paint_area : ℝ := 500

-- Define the number of faces of a cube
def cube_faces : ℕ := 6

-- Theorem statement
theorem white_square_area :
  let face_area := cube_edge ^ 2
  let blue_area_per_face := blue_paint_area / cube_faces
  let white_area_per_face := face_area - blue_area_per_face
  white_area_per_face = 425 / 3 := by
  sorry

end white_square_area_l3764_376499


namespace sum_of_fractions_equals_one_l3764_376466

theorem sum_of_fractions_equals_one 
  (a b c x y z : ℝ) 
  (h1 : 13 * x + b * y + c * z = 0)
  (h2 : a * x + 23 * y + c * z = 0)
  (h3 : a * x + b * y + 42 * z = 0)
  (h4 : a ≠ 13)
  (h5 : x ≠ 0) :
  a / (a - 13) + b / (b - 23) + c / (c - 42) = 1 := by
  sorry

end sum_of_fractions_equals_one_l3764_376466


namespace eugene_payment_l3764_376462

def tshirt_cost : ℕ := 20
def pants_cost : ℕ := 80
def shoes_cost : ℕ := 150
def discount_rate : ℚ := 1/10

def tshirt_quantity : ℕ := 4
def pants_quantity : ℕ := 3
def shoes_quantity : ℕ := 2

def total_cost : ℕ := tshirt_cost * tshirt_quantity + pants_cost * pants_quantity + shoes_cost * shoes_quantity

def discounted_cost : ℚ := (1 - discount_rate) * total_cost

theorem eugene_payment : discounted_cost = 558 := by
  sorry

end eugene_payment_l3764_376462


namespace power_two_ge_square_l3764_376447

theorem power_two_ge_square (n : ℕ) : 2^n ≥ n^2 ↔ n ≠ 3 :=
sorry

end power_two_ge_square_l3764_376447


namespace inequality_and_nonexistence_l3764_376449

theorem inequality_and_nonexistence (x y z : ℝ) :
  (x^2 + 2*y^2 + 3*z^2 ≥ Real.sqrt 3 * (x*y + y*z + z*x)) ∧
  (∀ k > Real.sqrt 3, ∃ x y z : ℝ, x^2 + 2*y^2 + 3*z^2 < k*(x*y + y*z + z*x)) :=
by sorry

end inequality_and_nonexistence_l3764_376449


namespace magazine_subscription_pigeonhole_l3764_376480

theorem magazine_subscription_pigeonhole 
  (total_students : Nat) 
  (subscription_combinations : Nat) 
  (h1 : total_students = 39) 
  (h2 : subscription_combinations = 7) :
  ∃ (combination : Nat), combination ≤ subscription_combinations ∧ 
    (total_students / subscription_combinations + 1 : Nat) ≤ 
      (λ i => (total_students / subscription_combinations : Nat) + 
        if i ≤ (total_students % subscription_combinations) then 1 else 0) combination :=
by
  sorry

end magazine_subscription_pigeonhole_l3764_376480


namespace right_triangle_among_sets_l3764_376482

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_among_sets :
  ¬ is_right_triangle 1 2 3 ∧
  ¬ is_right_triangle 2 3 4 ∧
  is_right_triangle 3 4 5 ∧
  ¬ is_right_triangle 4 5 6 :=
sorry

end right_triangle_among_sets_l3764_376482


namespace shopping_expense_percentage_l3764_376445

theorem shopping_expense_percentage (T : ℝ) (O : ℝ) : 
  T > 0 →
  0.50 * T + 0.20 * T + O * T / 100 = T →
  0.04 * (0.50 * T) + 0 * (0.20 * T) + 0.08 * (O * T / 100) = 0.044 * T →
  O = 30 := by
sorry

end shopping_expense_percentage_l3764_376445


namespace thirtieth_term_of_sequence_l3764_376450

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence : arithmetic_sequence 3 4 30 = 119 := by
  sorry

end thirtieth_term_of_sequence_l3764_376450


namespace bus_students_l3764_376404

theorem bus_students (initial : Real) (got_on : Real) (total : Real) : 
  initial = 10.0 → got_on = 3.0 → total = initial + got_on → total = 13.0 := by
  sorry

end bus_students_l3764_376404


namespace oil_bill_difference_l3764_376485

/-- Given the oil bills for January and February, calculate the difference
    between February's bill in two scenarios. -/
theorem oil_bill_difference (jan_bill : ℝ) (feb_ratio1 feb_ratio2 jan_ratio1 jan_ratio2 : ℚ) :
  jan_bill = 120 →
  feb_ratio1 / jan_ratio1 = 5 / 4 →
  feb_ratio2 / jan_ratio2 = 3 / 2 →
  ∃ (feb_bill1 feb_bill2 : ℝ),
    feb_bill1 / jan_bill = feb_ratio1 / jan_ratio1 ∧
    feb_bill2 / jan_bill = feb_ratio2 / jan_ratio2 ∧
    feb_bill2 - feb_bill1 = 30 :=
by sorry

end oil_bill_difference_l3764_376485


namespace power_function_positive_l3764_376418

theorem power_function_positive (α : ℚ) (x : ℝ) (h : x > 0) : x ^ (α : ℝ) > 0 := by
  sorry

end power_function_positive_l3764_376418


namespace cone_volume_from_half_sector_l3764_376413

/-- The volume of a cone formed by rolling up a half-sector of a circle -/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let slant_height := r
  let base_circumference := r * π
  let base_radius := base_circumference / (2 * π)
  let cone_height := Real.sqrt (slant_height ^ 2 - base_radius ^ 2)
  (1 / 3) * π * base_radius ^ 2 * cone_height = 9 * π * Real.sqrt 3 := by
  sorry

end cone_volume_from_half_sector_l3764_376413


namespace children_ages_l3764_376419

theorem children_ages (total_age_first_birth total_age_third_birth total_age_children : ℕ)
  (h1 : total_age_first_birth = 45)
  (h2 : total_age_third_birth = 70)
  (h3 : total_age_children = 14) :
  ∃ (age1 age2 age3 : ℕ),
    age1 = 8 ∧ age2 = 5 ∧ age3 = 1 ∧
    age1 + age2 + age3 = total_age_children :=
by
  sorry


end children_ages_l3764_376419


namespace waiter_customers_l3764_376469

theorem waiter_customers (initial_customers : ℕ) : 
  (initial_customers - 3 + 39 = 50) → initial_customers = 14 := by
  sorry

end waiter_customers_l3764_376469


namespace sin_product_seventh_pi_l3764_376478

theorem sin_product_seventh_pi : 
  Real.sin (π / 7) * Real.sin (2 * π / 7) * Real.sin (3 * π / 7) = Real.sqrt 13 / 8 := by
  sorry

end sin_product_seventh_pi_l3764_376478


namespace n_div_30_n_squared_cube_n_cubed_square_n_smallest_n_has_three_digits_l3764_376473

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
theorem n_div_30 : 30 ∣ n := sorry

/-- n^2 is a perfect cube -/
theorem n_squared_cube : ∃ k : ℕ, n^2 = k^3 := sorry

/-- n^3 is a perfect square -/
theorem n_cubed_square : ∃ k : ℕ, n^3 = k^2 := sorry

/-- n is the smallest positive integer satisfying the conditions -/
theorem n_smallest : ∀ m : ℕ, m < n → ¬(30 ∣ m ∧ (∃ k : ℕ, m^2 = k^3) ∧ (∃ k : ℕ, m^3 = k^2)) := sorry

/-- The number of digits in n -/
def digits_of_n : ℕ := sorry

/-- Theorem stating that n has 3 digits -/
theorem n_has_three_digits : digits_of_n = 3 := sorry

end n_div_30_n_squared_cube_n_cubed_square_n_smallest_n_has_three_digits_l3764_376473


namespace exists_special_sequence_l3764_376493

/-- A sequence of natural numbers satisfying specific conditions -/
def SpecialSequence (F : ℕ → ℕ) : Prop :=
  (∀ k, ∃ n, F n = k) ∧
  (∀ k, Set.Infinite {n | F n = k}) ∧
  (∀ n ≥ 2, F (F (n^163)) = F (F n) + F (F 361))

/-- There exists a sequence satisfying the SpecialSequence conditions -/
theorem exists_special_sequence : ∃ F, SpecialSequence F := by
  sorry

end exists_special_sequence_l3764_376493


namespace function_property_l3764_376429

/-- A function satisfying f(x) + 3f(1 - x) = 4x^3 for all real x has f(4) = -72.5 -/
theorem function_property (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 3 * f (1 - x) = 4 * x^3) : 
  f 4 = -72.5 := by
  sorry

end function_property_l3764_376429


namespace money_percentage_difference_l3764_376427

/-- The problem statement about Kim, Sal, and Phil's money --/
theorem money_percentage_difference 
  (sal_phil_total : ℝ)
  (kim_money : ℝ)
  (sal_percent_less : ℝ)
  (h1 : sal_phil_total = 1.80)
  (h2 : kim_money = 1.12)
  (h3 : sal_percent_less = 20) :
  let phil_money := sal_phil_total / (2 - sal_percent_less / 100)
  let sal_money := phil_money * (1 - sal_percent_less / 100)
  let percentage_difference := (kim_money - sal_money) / sal_money * 100
  percentage_difference = 40 := by
sorry

end money_percentage_difference_l3764_376427


namespace irrational_among_options_l3764_376444

theorem irrational_among_options : 
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 5 = (a : ℚ) / b) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (3.14 : ℝ) = (a : ℚ) / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (22 : ℚ) / 7 = (a : ℚ) / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 4 = (a : ℚ) / b) :=
by
  sorry

end irrational_among_options_l3764_376444


namespace tan_alpha_plus_pi_fourth_l3764_376430

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin α = 3/5) : 
  Real.tan (α + π/4) = 1/7 := by
  sorry

end tan_alpha_plus_pi_fourth_l3764_376430


namespace dissimilarTerms_eq_distributionWays_l3764_376461

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^10 -/
def dissimilarTerms : ℕ := Nat.choose 13 3

/-- The number of ways to distribute 10 indistinguishable objects into 4 distinguishable boxes -/
def distributionWays : ℕ := Nat.choose 13 3

theorem dissimilarTerms_eq_distributionWays : dissimilarTerms = distributionWays := by
  sorry

end dissimilarTerms_eq_distributionWays_l3764_376461


namespace job_completion_time_l3764_376414

/-- Given that m people can complete a job in d days, 
    prove that (m + r) people can complete the same job in md / (m + r) days. -/
theorem job_completion_time 
  (m d r : ℕ) (m_pos : m > 0) (d_pos : d > 0) (r_pos : r > 0) : 
  let n := (m * d) / (m + r)
  ∃ (W : ℝ), W > 0 ∧ W / (m * d) = W / ((m + r) * n) :=
sorry

end job_completion_time_l3764_376414


namespace partial_week_salary_l3764_376463

/-- Calculates the salary for a partial work week --/
theorem partial_week_salary
  (usual_hours : ℝ)
  (worked_fraction : ℝ)
  (hourly_rate : ℝ)
  (h1 : usual_hours = 40)
  (h2 : worked_fraction = 4/5)
  (h3 : hourly_rate = 15) :
  worked_fraction * usual_hours * hourly_rate = 480 := by
  sorry

#check partial_week_salary

end partial_week_salary_l3764_376463


namespace refrigerator_temp_difference_l3764_376459

/-- The temperature difference between two compartments in a refrigerator -/
def temperature_difference (refrigeration_temp freezer_temp : ℤ) : ℤ :=
  refrigeration_temp - freezer_temp

/-- Theorem stating the temperature difference between specific compartments -/
theorem refrigerator_temp_difference :
  temperature_difference 3 (-10) = 13 := by
  sorry

end refrigerator_temp_difference_l3764_376459


namespace grasshopper_movement_l3764_376417

/-- Represents the possible jump distances of the grasshopper -/
inductive Jump
| large : Jump  -- 36 cm jump
| small : Jump  -- 14 cm jump

/-- Represents the direction of the jump -/
inductive Direction
| left : Direction
| right : Direction

/-- Represents a single jump of the grasshopper -/
structure GrasshopperJump :=
  (distance : Jump)
  (direction : Direction)

/-- The distance covered by a single jump -/
def jumpDistance (j : GrasshopperJump) : ℤ :=
  match j.distance, j.direction with
  | Jump.large, Direction.right => 36
  | Jump.large, Direction.left  => -36
  | Jump.small, Direction.right => 14
  | Jump.small, Direction.left  => -14

/-- The total distance covered by a sequence of jumps -/
def totalDistance (jumps : List GrasshopperJump) : ℤ :=
  jumps.foldl (fun acc j => acc + jumpDistance j) 0

/-- Predicate to check if a distance is reachable by the grasshopper -/
def isReachable (d : ℤ) : Prop :=
  ∃ (jumps : List GrasshopperJump), totalDistance jumps = d

theorem grasshopper_movement :
  (¬ isReachable 3) ∧ (isReachable 2) ∧ (isReachable 1234) := by sorry

end grasshopper_movement_l3764_376417


namespace theater_admission_revenue_l3764_376425

/-- Calculates the total amount collected from theater admissions --/
def total_amount_collected (adult_price child_price : ℚ) (total_attendance children_attendance : ℕ) : ℚ :=
  let adults_attendance := total_attendance - children_attendance
  let adult_revenue := adult_price * adults_attendance
  let child_revenue := child_price * children_attendance
  adult_revenue + child_revenue

/-- Theorem stating that the total amount collected is $140 given the specified conditions --/
theorem theater_admission_revenue :
  total_amount_collected (60/100) (25/100) 280 80 = 140 := by
  sorry

end theater_admission_revenue_l3764_376425


namespace milk_conversion_rate_l3764_376468

/-- The number of ounces in a gallon of milk -/
def ounces_per_gallon : ℕ := sorry

/-- The initial amount of milk in gallons -/
def initial_gallons : ℕ := 3

/-- The amount of milk consumed in ounces -/
def consumed_ounces : ℕ := 13

/-- The remaining amount of milk in ounces -/
def remaining_ounces : ℕ := 371

theorem milk_conversion_rate :
  ounces_per_gallon = 128 :=
by sorry

end milk_conversion_rate_l3764_376468


namespace only_seven_satisfies_inequality_l3764_376494

theorem only_seven_satisfies_inequality :
  ∃! (n : ℤ), (3 : ℚ) / 10 < (n : ℚ) / 20 ∧ (n : ℚ) / 20 < 2 / 5 :=
by
  sorry

end only_seven_satisfies_inequality_l3764_376494


namespace representation_2015_l3764_376416

theorem representation_2015 : ∃ (a b c : ℤ), 
  a + b + c = 2015 ∧ 
  Nat.Prime a.natAbs ∧ 
  ∃ (k : ℤ), b = 3 * k ∧
  400 < c ∧ c < 500 ∧
  ¬∃ (m : ℤ), c = 3 * m := by
  sorry

end representation_2015_l3764_376416


namespace number_exceeding_percentage_l3764_376457

theorem number_exceeding_percentage : ∃ x : ℝ, x = 75 ∧ x = 0.16 * x + 63 := by
  sorry

end number_exceeding_percentage_l3764_376457


namespace minimizing_integral_minimizing_function_achieves_minimum_minimizing_function_integral_one_l3764_376495

noncomputable def minimizing_function (x : ℝ) : ℝ := 6 / (Real.pi * (x^2 + x + 1))

theorem minimizing_integral 
  (f : ℝ → ℝ) 
  (hf_continuous : Continuous f) 
  (hf_integral : ∫ x in (0:ℝ)..1, f x = 1) :
  ∫ x in (0:ℝ)..1, (x^2 + x + 1) * (f x)^2 ≥ 6 / Real.pi :=
sorry

theorem minimizing_function_achieves_minimum :
  ∫ x in (0:ℝ)..1, (x^2 + x + 1) * (minimizing_function x)^2 = 6 / Real.pi :=
sorry

theorem minimizing_function_integral_one :
  ∫ x in (0:ℝ)..1, minimizing_function x = 1 :=
sorry

end minimizing_integral_minimizing_function_achieves_minimum_minimizing_function_integral_one_l3764_376495
