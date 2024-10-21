import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_abs_sum_l1343_134387

theorem min_value_abs_sum : 
  (∀ x : ℝ, |2*x - 1| + |2*x - 2| ≥ 1) ∧ (∃ x : ℝ, |2*x - 1| + |2*x - 2| = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_abs_sum_l1343_134387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relationship_l1343_134341

theorem order_relationship (x : ℝ) (hx : x ∈ Set.Ioo (-1/2 : ℝ) 0) :
  Real.cos ((x + 1) * π) < Real.sin (Real.cos (x * π)) ∧ 
  Real.sin (Real.cos (x * π)) < Real.cos (Real.sin (x * π)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relationship_l1343_134341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_always_60_always_loss_of_40_l1343_134304

/-- Represents the outcome of a bet -/
inductive BetOutcome
  | Win
  | Loss
deriving BEq, Repr

/-- Calculates the new amount after a bet -/
def betResult (amount : ℚ) (outcome : BetOutcome) (isFirstWin : Bool) : ℚ :=
  match outcome with
  | BetOutcome.Win => if isFirstWin then amount * (1 + 3/5) else amount * (1 + 1/2)
  | BetOutcome.Loss => amount * (1 - 1/2)

/-- Calculates the final amount after all bets -/
def finalAmount (bets : List BetOutcome) : ℚ :=
  let initialAmount : ℚ := 100
  let rec loop (remaining : List BetOutcome) (current : ℚ) (firstWinUsed : Bool) : ℚ :=
    match remaining with
    | [] => current
    | bet :: rest =>
        let newAmount := betResult current bet (bet == BetOutcome.Win && !firstWinUsed)
        loop rest newAmount (firstWinUsed || bet == BetOutcome.Win)
  loop bets initialAmount false

theorem final_amount_always_60 :
  ∀ (bets : List BetOutcome),
    (bets.length = 4) →
    (bets.count BetOutcome.Win = 2) →
    (bets.count BetOutcome.Loss = 2) →
    finalAmount bets = 60 := by
  sorry

theorem always_loss_of_40 :
  ∀ (bets : List BetOutcome),
    (bets.length = 4) →
    (bets.count BetOutcome.Win = 2) →
    (bets.count BetOutcome.Loss = 2) →
    100 - finalAmount bets = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_always_60_always_loss_of_40_l1343_134304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_cone_l1343_134350

/-- The volume of a cone formed by cutting a cylinder to maximize the cone's volume,
    given that the cut-off portion's volume is 3.6 cubic meters larger than the cone's volume. -/
theorem max_volume_cone (cone_volume : ℝ) : 
  (cone_volume > 0) →  -- Ensure positive volume
  (2 * cone_volume - cone_volume = 3.6) →  -- Condition from the problem
  (cone_volume = 3.6) := by
  intro h1 h2
  -- Proof steps would go here
  sorry

#check max_volume_cone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_cone_l1343_134350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thinnest_nanotube_scientific_notation_l1343_134392

/-- Represents the diameter of the world's thinnest carbon nanotube in meters -/
noncomputable def thinnest_nanotube_diameter : ℝ := 0.0000000005

/-- Represents the scientific notation of the thinnest carbon nanotube diameter -/
noncomputable def thinnest_nanotube_scientific : ℝ := 5 * (10 ^ (-10 : ℤ))

/-- Theorem stating that the thinnest carbon nanotube diameter in meters 
    is equal to its scientific notation representation -/
theorem thinnest_nanotube_scientific_notation : 
  thinnest_nanotube_diameter = thinnest_nanotube_scientific :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thinnest_nanotube_scientific_notation_l1343_134392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_book_cost_l1343_134394

-- Define the exchange rate
noncomputable def exchange_rate : ℝ := 0.82

-- Define the book price in GBP
noncomputable def book_price_gbp : ℝ := 30

-- Function to convert GBP to USD
noncomputable def gbp_to_usd (gbp : ℝ) : ℝ := gbp / exchange_rate

-- Function to round to the nearest hundredth
noncomputable def round_to_hundredth (x : ℝ) : ℝ := 
  ⌊x * 100 + 0.5⌋ / 100

-- Theorem statement
theorem alice_book_cost :
  round_to_hundredth (gbp_to_usd book_price_gbp) = 36.59 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_book_cost_l1343_134394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_a_l1343_134356

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sin (7 * Real.pi / 6 - 2 * x) - 1

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem min_side_a (abc : Triangle) (h1 : f abc.A = 1/2) 
  (h2 : abc.b^2 + abc.c^2 - abc.a^2 - abc.b * abc.c * Real.cos abc.A = 4) : 
  abc.a ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_a_l1343_134356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_constant_inequality_l1343_134346

theorem largest_constant_inequality :
  (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    Real.sqrt (a/(b+c+d+e)) + Real.sqrt (b/(a+c+d+e)) + Real.sqrt (c/(a+b+d+e)) +
    Real.sqrt (d/(a+b+c+e)) + Real.sqrt (e/(a+b+c+d)) > 2) ∧
  (∀ m : ℝ, (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    Real.sqrt (a/(b+c+d+e)) + Real.sqrt (b/(a+c+d+e)) + Real.sqrt (c/(a+b+d+e)) +
    Real.sqrt (d/(a+b+c+e)) + Real.sqrt (e/(a+b+c+d)) > m) → m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_constant_inequality_l1343_134346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1343_134380

-- Define the function f(x) using a recursive definition
def f_aux : Nat → ℝ → ℝ
| 0, x => 2009
| 1, x => -2008*x + 2009
| (n+2), x => x^(n+2) - (n+2)*x^(n+1) + f_aux n x

def f (x : ℝ) : ℝ := f_aux 2008 x

-- State the theorem
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ 1005 :=
sorry

-- Additional lemma to help with the proof
lemma f_aux_min_value (n : Nat) :
  ∀ x : ℝ, f_aux n x ≥ (n/2 + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1343_134380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_l1343_134371

noncomputable section

-- Define the point in rectangular coordinates
def x : ℝ := -3
noncomputable def y : ℝ := 3 * Real.sqrt 3
def z : ℝ := 6

-- Define the spherical coordinates
noncomputable def ρ : ℝ := 6 * Real.sqrt 2
noncomputable def θ : ℝ := 11 * Real.pi / 6
noncomputable def φ : ℝ := Real.pi / 4

-- State the theorem
theorem rectangular_to_spherical :
  (x, y, z) = (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ) ∧
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_l1343_134371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l1343_134382

/-- The function g(n) defined as (1/4) * n^2 * (n+1)^2 -/
noncomputable def g (n : ℝ) : ℝ := (1/4) * n^2 * (n+1)^2

/-- Theorem stating that g(s) - g(s-1) = s^3 for any real number s -/
theorem g_difference (s : ℝ) : g s - g (s-1) = s^3 := by
  -- Expand the definition of g
  unfold g
  -- Simplify the expression
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l1343_134382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_perfect_square_s_one_eq_one_l1343_134357

/-- s(n) is a function that returns a n-digit number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ := sorry

/-- The first perfect square is 1 -/
theorem first_perfect_square : (1 : ℕ) ^ 2 = 1 := by
  rfl

/-- s(1) is equal to 1 -/
theorem s_one_eq_one : s 1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_perfect_square_s_one_eq_one_l1343_134357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_length_implies_t_range_l1343_134375

-- Define the inequality system
def inequality_system (x t : ℝ) : Prop :=
  1 ≤ 12 / (x + 3) ∧ 12 / (x + 3) ≤ 3 ∧ x^2 + 3*t*x - 4*t^2 < 0

-- Define the solution set of the inequality system
def solution_set (t : ℝ) : Set ℝ :=
  {x : ℝ | inequality_system x t}

-- Define the length of an interval
def interval_length (a b : ℝ) : ℝ := b - a

-- Theorem statement
theorem solution_set_length_implies_t_range :
  ∀ t : ℝ, (∃ a b c d : ℝ, a < b ∧ c < d ∧ 
    solution_set t = Set.Icc a b ∪ Set.Icc c d ∧
    interval_length a b + interval_length c d = 8) →
  t ∈ Set.Iic (-9/4) ∪ Set.Ici 9 :=
by sorry

#check solution_set_length_implies_t_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_length_implies_t_range_l1343_134375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_gym_cost_l1343_134366

-- Define the gym membership and personal training costs
def gym1_monthly_cost : ℚ := 10
def gym1_signup_fee : ℚ := 50
def gym2_monthly_cost : ℚ := 3 * gym1_monthly_cost
def gym2_signup_fee : ℚ := 4 * gym2_monthly_cost
def gym1_training_cost : ℚ := 25
def gym2_training_cost : ℚ := 45

-- Define the discount rate and number of weeks in a year
def discount_rate : ℚ := 1/10
def weeks_in_year : ℚ := 52

-- Define the total cost function
noncomputable def total_cost : ℚ :=
  -- Gym 1 costs
  (gym1_monthly_cost * 12 + 
   gym1_signup_fee * (1 - discount_rate) +
   gym1_training_cost * weeks_in_year) +
  -- Gym 2 costs
  (gym2_monthly_cost * 12 + 
   gym2_signup_fee * (1 - discount_rate) +
   gym2_training_cost * weeks_in_year)

-- Theorem statement
theorem total_gym_cost : total_cost = 4273 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_gym_cost_l1343_134366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_truncated_cone_l1343_134393

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  bottom_radius : ℝ
  top_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  bottom_tangent : sphere_radius ≤ bottom_radius
  top_tangent : sphere_radius ≤ top_radius
  lateral_tangent : True  -- This condition is implied by the geometry but not easily expressible

/-- The radius of the sphere in a specific truncated cone configuration -/
noncomputable def sphere_radius_in_specific_cone : ℝ := 4 * Real.sqrt 5

/-- Theorem stating that the radius of the sphere in the specific truncated cone is 4√5 -/
theorem sphere_radius_in_truncated_cone 
  (cone : TruncatedConeWithSphere)
  (h_bottom : cone.bottom_radius = 20)
  (h_top : cone.top_radius = 4)
  (h_height : cone.height = 15) :
  cone.sphere_radius = sphere_radius_in_specific_cone := by
  sorry

#check sphere_radius_in_truncated_cone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_truncated_cone_l1343_134393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_perimeter_l1343_134395

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  R : ℝ
  α : ℝ
  h_positive : R > 0
  h_acute : 0 < α ∧ α < π / 2

/-- The perimeter of the isosceles trapezoid -/
noncomputable def perimeter (t : IsoscelesTrapezoid) : ℝ :=
  8 * t.R / Real.sin t.α

/-- Theorem: The perimeter of an isosceles trapezoid circumscribed around a circle 
    with radius R and acute base angle α is equal to 8R / sin(α) -/
theorem isosceles_trapezoid_perimeter (t : IsoscelesTrapezoid) : 
  perimeter t = 8 * t.R / Real.sin t.α :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_perimeter_l1343_134395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_l1343_134330

/-- The distance from the center of a sphere to the plane of a triangle on its surface -/
noncomputable def distance_to_triangle (radius : ℝ) (side1 side2 side3 : ℝ) : ℝ :=
  let s := (side1 + side2 + side3) / 2
  let area := Real.sqrt (s * (s - side1) * (s - side2) * (s - side3))
  let circumradius := (side1 * side2 * side3) / (4 * area)
  Real.sqrt (radius^2 - circumradius^2)

theorem sphere_triangle_distance (x y z : ℕ) :
  (∀ p : ℕ, Nat.Prime p → y % p^2 ≠ 0) →
  Nat.Coprime x z →
  x > 0 ∧ y > 0 ∧ z > 0 →
  distance_to_triangle 25 17 18 19 = (x * Real.sqrt y) / z →
  x + y + z = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_l1343_134330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_board_probability_l1343_134370

noncomputable section

/-- The side length of each equilateral triangle on the dart board -/
def triangle_side_length : ℝ := 4

/-- The area of one equilateral triangle on the dart board -/
def triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side_length^2

/-- The side length of the central square on the dart board -/
def square_side_length : ℝ := triangle_side_length * (Real.sqrt 3 / 2)

/-- The area of the central square on the dart board -/
def square_area : ℝ := square_side_length^2

/-- The total area of the dart board -/
def total_area : ℝ := square_area + 4 * triangle_area

/-- The probability of a dart landing within the central square -/
def landing_probability : ℝ := square_area / total_area

theorem dart_board_probability :
  landing_probability = 12 / (12 + 16 * Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_board_probability_l1343_134370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_production_proof_l1343_134398

/-- The time (in hours) it takes A and B to complete the batch together -/
noncomputable def joint_time : ℝ := 8

/-- The time (in hours) it takes A to complete the batch alone -/
noncomputable def a_alone_time : ℝ := 12

/-- The time (in hours) A and B work together before A leaves -/
noncomputable def initial_work_time : ℝ := 2 + 2/5

/-- The number of additional parts B produces after A leaves -/
def b_additional_parts : ℕ := 420

/-- The total number of parts B produces -/
def b_total_parts : ℕ := 480

theorem b_production_proof :
  ∃ (total_parts : ℕ),
    (total_parts : ℝ) / joint_time = total_parts / a_alone_time + (total_parts - (total_parts : ℝ) / a_alone_time * initial_work_time) / (joint_time - initial_work_time) ∧
    (total_parts : ℝ) / joint_time * initial_work_time + b_additional_parts = total_parts ∧
    b_total_parts = total_parts - (total_parts : ℝ) / a_alone_time * initial_work_time := by
  sorry

#eval b_total_parts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_production_proof_l1343_134398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l1343_134347

theorem max_value_expression (a b c d : ℕ) : 
  a ∈ ({1, 2, 5, 6} : Set ℕ) → 
  b ∈ ({1, 2, 5, 6} : Set ℕ) → 
  c ∈ ({1, 2, 5, 6} : Set ℕ) → 
  d ∈ ({1, 2, 5, 6} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  (∀ x y z w : ℕ, x ∈ ({1, 2, 5, 6} : Set ℕ) → y ∈ ({1, 2, 5, 6} : Set ℕ) → 
               z ∈ ({1, 2, 5, 6} : Set ℕ) → w ∈ ({1, 2, 5, 6} : Set ℕ) →
               x ≠ y → x ≠ z → x ≠ w → y ≠ z → y ≠ w → z ≠ w →
               (x - y)^2 + (z * w) ≤ (a - b)^2 + (c * d)) →
  (a - b)^2 + (c * d) = 35 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l1343_134347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_ratio_l1343_134342

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The angle between the asymptotes of a hyperbola -/
noncomputable def asymptote_angle (h : Hyperbola) : ℝ := 2 * Real.arctan (h.b / h.a)

theorem hyperbola_asymptote_angle_ratio (h : Hyperbola) 
  (h_angle : asymptote_angle h = π / 4) : h.a / h.b = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_ratio_l1343_134342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_net_effect_on_revenue_l1343_134314

/-- Calculates the net effect on revenue when price is reduced and sales increase -/
theorem net_effect_on_revenue 
  (original_price : ℝ) 
  (original_sales : ℝ) 
  (price_reduction_percent : ℝ) 
  (sales_increase_percent : ℝ) 
  (h1 : price_reduction_percent = 40) 
  (h2 : sales_increase_percent = 80) : 
  (1 - price_reduction_percent / 100) * (1 + sales_increase_percent / 100) * 
  (original_price * original_sales) = 
  1.08 * (original_price * original_sales) := by
  sorry

#check net_effect_on_revenue

end NUMINAMATH_CALUDE_ERRORFEEDBACK_net_effect_on_revenue_l1343_134314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_equals_cube_root_l1343_134368

theorem square_root_equals_cube_root (x : ℝ) :
  Real.sqrt x = x ^ (1/3) → x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_equals_cube_root_l1343_134368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1343_134300

-- Define the function g(x) = x^2 + 1 for x < 0
def g (x : ℝ) : ℝ := x^2 + 1

-- Define the symmetry condition
def symmetric_about_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, y = g x ∧ x < 0 → f y = x

-- State the theorem
theorem domain_of_f 
  (f : ℝ → ℝ) 
  (h : symmetric_about_y_eq_x f g) : 
  Set.range f = Set.Ioi 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1343_134300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1343_134384

noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (3, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem trajectory_equation (x y : ℝ) :
  let M : ℝ × ℝ := (x, y)
  distance M O = (1/2) * distance M A →
  x^2 + y^2 + 2*x - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1343_134384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_l1343_134335

theorem seating_arrangements (n : ℕ) (k : ℕ) (h : n = 10 ∧ k = 3) :
  (Nat.factorial n : ℕ) - Nat.factorial (n - k) * Nat.factorial k = 3386880 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_l1343_134335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1343_134389

noncomputable def f (x : ℝ) : ℝ := (x^3 - 3*x^2 + 3*x - 1) / (x^2 - 1)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -1 ∨ (-1 < x ∧ x < 1) ∨ 1 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1343_134389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_a5_a16_l1343_134396

/-- An arithmetic sequence with positive terms -/
structure PositiveArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_positive : ∀ n, a n > 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : PositiveArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (seq.a 1 + seq.a n)

theorem max_product_a5_a16 (seq : PositiveArithmeticSequence) 
    (h_sum : sum_n seq 20 = 100) :
    (seq.a 5 * seq.a 16 ≤ 25) ∧ 
    (∃ seq' : PositiveArithmeticSequence, sum_n seq' 20 = 100 ∧ seq'.a 5 * seq'.a 16 = 25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_a5_a16_l1343_134396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_honest_dwarf_count_l1343_134310

/-- The number of dwarfs who answered "yes" to each question -/
structure YesAnswers where
  q1 : ℕ
  q2 : ℕ
  q3 : ℕ
  q4 : ℕ

/-- Properties of the dwarf kingdom -/
class DwarfKingdom where
  total_dwarfs : ℕ
  yes_answers : YesAnswers
  honest_yes_count : ℕ → ℕ
  lying_yes_count : ℕ → ℕ

/-- The specific dwarf kingdom in the problem -/
def problem_kingdom : DwarfKingdom where
  total_dwarfs := 100
  yes_answers := { q1 := 40, q2 := 50, q3 := 70, q4 := 100 }
  honest_yes_count := λ n ↦ if n ≤ 3 then 1 else 0
  lying_yes_count := λ n ↦ if n ≤ 3 then 2 else 0

theorem honest_dwarf_count (k : DwarfKingdom) : 
  k.total_dwarfs = k.yes_answers.q4 →
  (∀ n, n ≤ 3 → k.honest_yes_count n = 1) →
  (∀ n, n ≤ 3 → k.lying_yes_count n = 2) →
  k.yes_answers.q1 + k.yes_answers.q2 + k.yes_answers.q3 = 
    k.total_dwarfs + 2 * (k.total_dwarfs - (k.total_dwarfs - k.yes_answers.q1 - k.yes_answers.q2 - k.yes_answers.q3)) →
  k.total_dwarfs - k.yes_answers.q1 - k.yes_answers.q2 - k.yes_answers.q3 = 40 := by
  sorry

#check honest_dwarf_count problem_kingdom

end NUMINAMATH_CALUDE_ERRORFEEDBACK_honest_dwarf_count_l1343_134310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_betty_reaches_abel_time_l1343_134386

-- Define the variables and constants
noncomputable def total_distance : ℝ := 25
noncomputable def rate_of_approach : ℝ := 2
noncomputable def initial_time : ℝ := 10

-- Define Abel's and Betty's speeds
noncomputable def betty_speed : ℝ := rate_of_approach * 60 / 3
noncomputable def abel_speed : ℝ := 2 * betty_speed

-- Define the theorem
theorem betty_reaches_abel_time :
  let distance_covered := rate_of_approach * initial_time
  let remaining_distance := total_distance - distance_covered
  let remaining_time := remaining_distance / betty_speed
  initial_time + remaining_time = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_betty_reaches_abel_time_l1343_134386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x₀_l1343_134331

/-- The natural logarithm function -/
noncomputable def f (x : ℝ) : ℝ := Real.log x

/-- The point of tangency -/
def x₀ : ℝ := 1

/-- The equation of the tangent line to f at x₀ -/
def tangent_line (x : ℝ) : ℝ := x - 1

theorem tangent_line_at_x₀ :
  ∀ x : ℝ, (deriv f x₀) * (x - x₀) + f x₀ = tangent_line x :=
by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x₀_l1343_134331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l1343_134397

noncomputable def normalDistribution (μ σ : ℝ) (x : ℝ) : ℝ := 
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ)^2) / (2 * σ^2))

theorem normal_distribution_probability 
  (X : ℝ → ℝ) (a : ℝ) (h₁ : a > 0) 
  (h₂ : ∀ x, X x = normalDistribution 100 a x) 
  (h₃ : ∫ x in Set.Iio 90, X x = 1/10) :
  ∫ x in Set.Icc 100 110, X x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l1343_134397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_axis_length_is_2_sqrt_2_l1343_134322

/-- An ellipse with equation x²/(10-m) + y²/(m-2) = 1, foci on y-axis, and focal distance 4 -/
structure Ellipse where
  m : ℝ
  eq : ∀ (x y : ℝ), x^2 / (10 - m) + y^2 / (m - 2) = 1
  foci_on_y : True  -- This is a placeholder for the condition that foci are on y-axis
  focal_distance : ℝ
  focal_distance_eq : focal_distance = 4

/-- The length of the minor axis of the ellipse -/
noncomputable def minor_axis_length (e : Ellipse) : ℝ := 2 * Real.sqrt (10 - e.m)

/-- Theorem stating that the length of the minor axis is 2√2 -/
theorem minor_axis_length_is_2_sqrt_2 (e : Ellipse) : minor_axis_length e = 2 * Real.sqrt 2 := by
  sorry

#check minor_axis_length_is_2_sqrt_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_axis_length_is_2_sqrt_2_l1343_134322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1343_134311

-- Define the points and shapes
variable (A B C D E F G H : ℝ × ℝ)
variable (ABCD : Set (ℝ × ℝ))
variable (EHGF : Set (ℝ × ℝ))

-- Define the conditions
def is_parallelogram (S : Set (ℝ × ℝ)) : Prop := sorry

def is_midpoint (M X Y : ℝ × ℝ) : Prop := sorry

def is_intersection (P : ℝ × ℝ) (L1 L2 : Set (ℝ × ℝ)) : Prop := sorry

noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem parallelogram_area 
  (h1 : is_parallelogram ABCD)
  (h2 : is_midpoint E C D)
  (h3 : is_intersection F (Set.insert A (Set.singleton E)) (Set.insert B (Set.singleton D)))
  (h4 : is_intersection H (Set.insert A (Set.singleton C)) (Set.insert B (Set.singleton E)))
  (h5 : is_intersection G (Set.insert A (Set.singleton C)) (Set.insert B (Set.singleton D)))
  (h6 : area EHGF = 15) :
  area ABCD = 90 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1343_134311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moon_permutations_l1343_134325

/- Define the function to calculate permutations of a multiset -/
def multiset_permutations (n : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial n / (repetitions.foldl (fun acc x => acc * Nat.factorial x) 1)

/- Theorem statement -/
theorem moon_permutations :
  let total_letters : ℕ := 4
  let repeated_letter_count : ℕ := 2
  let single_letter_count : ℕ := 1
  multiset_permutations total_letters [repeated_letter_count, single_letter_count, single_letter_count] = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moon_permutations_l1343_134325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_creates_bisector_l1343_134378

/-- An angle drawn on a transparent sheet --/
structure AngleOnSheet where
  sides : Set (ℝ × ℝ)
  vertex_inaccessible : Bool

/-- A folded sheet with a crease --/
structure FoldedSheet where
  original : AngleOnSheet
  crease : Set (ℝ × ℝ)
  sides_coincide : Bool

/-- Definition of an angle bisector --/
def is_angle_bisector (line : Set (ℝ × ℝ)) (angle : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- The theorem stating that folding the sheet so the sides coincide creates a bisector --/
theorem fold_creates_bisector (sheet : AngleOnSheet) (folded : FoldedSheet) :
  folded.original = sheet →
  folded.sides_coincide = true →
  is_angle_bisector folded.crease sheet.sides := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_creates_bisector_l1343_134378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F₁M_equation_l1343_134369

noncomputable section

-- Define the points F₁ and F₂
def F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
def F₂ : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the curve C
def C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  Real.sqrt ((x + Real.sqrt 3)^2 + y^2) + Real.sqrt ((x - Real.sqrt 3)^2 + y^2) = 4

-- Define the parallel condition
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂ ∨ m₁ = -m₂

-- Define the circle condition
def circle_condition (M N : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  (x₁ + x₂)^2 + (y₁ + y₂ - 4)^2 = (x₁ - x₂)^2 + (y₁ - y₂)^2

-- Theorem statement
theorem F₁M_equation (M N : ℝ × ℝ) (m : ℝ) :
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  y₁ > 0 ∧ y₂ > 0 ∧
  C x₁ y₁ ∧ C x₂ y₂ ∧
  x₁ = m * y₁ - Real.sqrt 3 ∧
  parallel m ((x₂ + Real.sqrt 3) / y₂) ∧
  circle_condition M N →
  m = Real.sqrt (4 * Real.sqrt 10 - 2) / 4 ∨
  m = -Real.sqrt (4 * Real.sqrt 10 - 2) / 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F₁M_equation_l1343_134369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_permutation_sum_l1343_134334

def permutation_sum (p : Equiv.Perm (Fin 12)) : ℚ :=
  |p.toFun 0 - p.toFun 1| + |p.toFun 2 - p.toFun 3| + |p.toFun 4 - p.toFun 5| + 
  |p.toFun 6 - p.toFun 7| + |p.toFun 8 - p.toFun 9| + |p.toFun 10 - p.toFun 11|

theorem average_permutation_sum : 
  (Finset.sum (Finset.univ : Finset (Equiv.Perm (Fin 12))) permutation_sum) / 
  (Finset.card (Finset.univ : Finset (Equiv.Perm (Fin 12)))) = 143 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_permutation_sum_l1343_134334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coth_squared_integral_l1343_134308

-- Define the hyperbolic cotangent function
noncomputable def coth (x : ℝ) : ℝ := Real.cosh x / Real.sinh x

-- State the theorem
theorem coth_squared_integral (x : ℝ) :
  deriv (fun x => x - coth x) = fun x => coth x ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coth_squared_integral_l1343_134308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_intersection_l1343_134339

-- Define arithmetic sequences a_n and b_n
def arithmetic_sequence (a : ℕ) (d : ℕ) : ℕ → ℕ
  | 0 => a
  | n + 1 => arithmetic_sequence a d n + d

-- Define sets A and B
def A (a d : ℕ) : Set ℕ := {n | ∃ k, n = arithmetic_sequence a d k}
def B (b e : ℕ) : Set ℕ := {n | ∃ k, n = arithmetic_sequence b e k}

-- Define set C as the intersection of A and B
def C (a b d e : ℕ) : Set ℕ := A a d ∩ B b e

-- Define what it means for a set of natural numbers to be an arithmetic sequence
def IsArithmeticSequence (S : Set ℕ) : Prop :=
  ∃ a d : ℕ, ∀ n ∈ S, ∃ k : ℕ, n = a + k * d

-- State the theorem
theorem arithmetic_sequence_intersection (a d e : ℕ) (hd : d > 0) (he : e > 0) :
  IsArithmeticSequence (C a a d e) ∧
  ¬ (∀ (b : ℕ), IsArithmeticSequence (C a b d e) → a = b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_intersection_l1343_134339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_course_choice_gender_relationship_l1343_134312

/-- Represents a 2x2 contingency table -/
structure ContingencyTable where
  boys_calligraphy : ℕ
  boys_paper_cutting : ℕ
  girls_calligraphy : ℕ
  girls_paper_cutting : ℕ
  total : ℕ

/-- Calculates the K² value for a given contingency table -/
noncomputable def calculate_k_squared (table : ContingencyTable) : ℝ :=
  let a := table.boys_calligraphy
  let b := table.boys_paper_cutting
  let c := table.girls_calligraphy
  let d := table.girls_paper_cutting
  let n := table.total
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for 95% confidence -/
def critical_value : ℝ := 3.841

/-- Theorem stating that the K² value for the given contingency table
    is greater than the critical value, indicating a significant
    relationship between course choice and gender -/
theorem course_choice_gender_relationship
  (table : ContingencyTable)
  (h1 : table.boys_calligraphy = 40)
  (h2 : table.boys_paper_cutting = 10)
  (h3 : table.girls_calligraphy = 30)
  (h4 : table.girls_paper_cutting = 20)
  (h5 : table.total = 100) :
  calculate_k_squared table > critical_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_course_choice_gender_relationship_l1343_134312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1343_134306

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then Real.log (-x) + 3 * x
  else Real.log x - 3 * x

-- State the theorem
theorem tangent_line_equation :
  (∀ x, f (-x) = f x) →  -- f is even
  f 1 = -3 →             -- the point (1, -3) is on the curve
  (∀ x, x < 0 → f x = Real.log (-x) + 3 * x) →  -- definition for x < 0
  ∃ m b, ∀ x, m * x + b = 2 * x + 1 ∧  -- equation of tangent line
         m = (deriv f 1) ∧             -- slope is the derivative at x = 1
         -3 = m * 1 + b                -- point (1, -3) satisfies the equation
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1343_134306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1343_134376

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  point : ℝ × ℝ
  eq : (point.1^2 / a^2) + (point.2^2 / b^2) = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

theorem ellipse_eccentricity_range (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : c^2 = a^2 - b^2) (h4 : ∀ P : Ellipse a b, 2*c^2 ≤ (a^2) ∧ (a^2) ≤ 3*c^2) :
  Real.sqrt 3 / 3 ≤ eccentricity a b ∧ eccentricity a b ≤ Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1343_134376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l1343_134377

/-- Given two vectors a and b in ℝ², prove that if a = (1, 2) and b = (-1, lambda) are perpendicular,
    then lambda = 1/2. -/
theorem perpendicular_vectors_lambda (a b : ℝ × ℝ) (lambda : ℝ) :
  a = (1, 2) →
  b = (-1, lambda) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  lambda = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l1343_134377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coconut_grove_solution_l1343_134360

/-- Represents the number of trees in the coconut grove that yield 120 nuts per year -/
def x : ℕ := sorry

/-- The number of trees yielding 40 nuts per year -/
def low_yield_trees : ℕ := x + 2

/-- The number of trees yielding 120 nuts per year -/
def medium_yield_trees : ℕ := x

/-- The number of trees yielding 180 nuts per year -/
def high_yield_trees : ℕ := x - 2

/-- The total number of nuts produced by all trees -/
def total_nuts : ℕ := 40 * low_yield_trees + 120 * medium_yield_trees + 180 * high_yield_trees

/-- The total number of trees in the grove -/
def total_trees : ℕ := low_yield_trees + medium_yield_trees + high_yield_trees

/-- The average yield per tree is 100 nuts -/
axiom average_yield : total_nuts = 100 * total_trees

/-- Theorem stating that x equals 7 -/
theorem coconut_grove_solution : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coconut_grove_solution_l1343_134360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_with_square_divisor_l1343_134367

theorem least_k_with_square_divisor (n : ℕ) (hn : n > 1) :
  let X := Finset.range (n^2 + 1) \ {0}
  ∃ k : ℕ, k = n^2 - n + 1 ∧
    (∀ S : Finset ℕ, S ⊆ X → S.card ≥ k →
      ∃ x y, x ∈ S ∧ y ∈ S ∧ x^2 ∣ y) ∧
    (∀ t : ℕ, t < k →
      ∃ T : Finset ℕ, T ⊆ X ∧ T.card = t ∧
        ∀ x y, x ∈ T → y ∈ T → ¬(x^2 ∣ y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_with_square_divisor_l1343_134367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1343_134332

theorem trigonometric_identities (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.cos (α + π / 6) = 1 / 3) : 
  Real.sin α = (2 * Real.sqrt 6 - 1) / 6 ∧ 
  Real.sin (2 * α + 5 * π / 6) = -7 / 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1343_134332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1343_134399

theorem problem_solution (a b c : ℤ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_eq1 : a^2 - b^2 - c^2 + a*b + b*c = 3003)
  (h_eq2 : a^2 + 4*b^2 + 4*c^2 - 4*a*b - 3*a*c - 3*b*c = -4004) :
  a = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1343_134399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_exponential_conversion_l1343_134365

noncomputable def complex_exponential_to_rectangular (sqrt3 : ℝ) (θ : ℝ) : ℂ :=
  (sqrt3 : ℂ) * Complex.exp (θ * Complex.I)

theorem complex_exponential_conversion :
  complex_exponential_to_rectangular (Real.sqrt 3) (13 * Real.pi / 6) = 
    Complex.mk (3 / 2) (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_exponential_conversion_l1343_134365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_prices_max_desk_lamps_l1343_134309

/- Define the unit prices and quantities -/
def desk_lamp_price : ℕ → ℕ := sorry
def flashlight_price : ℕ → ℕ := sorry
def desk_lamps_quantity : ℕ → ℕ := sorry
def flashlights_quantity : ℕ → ℕ := sorry

/- Define the conditions -/
axiom price_difference (x : ℕ) : desk_lamp_price x = flashlight_price x + 50
axiom equal_spending (x : ℕ) : desk_lamp_price x * desk_lamps_quantity x = flashlight_price x * flashlights_quantity x
axiom desk_lamp_spending (x : ℕ) : desk_lamp_price x * desk_lamps_quantity x = 240
axiom flashlight_spending (x : ℕ) : flashlight_price x * flashlights_quantity x = 90

/- Define the promotion conditions -/
def free_flashlight (a : ℕ) : ℕ := a
def required_flashlights (a : ℕ) : ℕ := 2 * a + 8
def total_cost (a : ℕ) : ℕ := 80 * a + 30 * (required_flashlights a - free_flashlight a)

/- State the theorems to be proved -/
theorem correct_prices : ∃ x : ℕ, desk_lamp_price x = 80 ∧ flashlight_price x = 30 := by
  sorry

theorem max_desk_lamps : ∀ a : ℕ, total_cost a ≤ 2440 → a ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_prices_max_desk_lamps_l1343_134309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purchase_savings_theorem_l1343_134344

/-- Calculates the percentage saved in a purchase transaction -/
noncomputable def percentage_saved (spent : ℝ) (saved : ℝ) : ℝ :=
  (saved / (spent + saved)) * 100

/-- Theorem stating that for a purchase of $20 with $2.75 saved, 
    the percentage saved is approximately 12.09% -/
theorem purchase_savings_theorem (ε : ℝ) (h_ε : ε > 0) :
  ∃ (δ : ℝ), abs (percentage_saved 20 2.75 - 12.09) < δ ∧ δ < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_purchase_savings_theorem_l1343_134344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_irrational_with_rational_output_l1343_134326

/-- A quadratic polynomial with rational coefficients -/
def QuadraticPolynomial (b c : ℚ) : ℝ → ℝ :=
  fun x => x^2 + (b : ℝ) * x + (c : ℝ)

/-- Theorem: For any quadratic polynomial with rational coefficients,
    there exists an irrational input that produces a rational output -/
theorem exists_irrational_with_rational_output (b c : ℚ) :
  ∃ x : ℝ, Irrational x ∧ ∃ q : ℚ, QuadraticPolynomial b c x = q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_irrational_with_rational_output_l1343_134326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1343_134301

noncomputable def P (n : ℕ) (x : ℝ) : ℝ := (x^(n+1) - 1) / (x - 1)

theorem inequality_holds (x : ℝ) (h : x > 0) :
  P 20 x * P 21 (x^2) ≤ P 20 (x^2) * P 22 x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1343_134301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_in_range_l1343_134352

/-- Represents the number of students in a class -/
def total_students : ℕ := 22

/-- Represents the number of students who scored in the 70%-79% range -/
def students_in_range : ℕ := 6

/-- Calculates the percentage of students in the 70%-79% range -/
def percentage : ℚ := (students_in_range : ℚ) / (total_students : ℚ) * 100

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ := (q + 1/2).floor

theorem percentage_in_range : round_to_nearest percentage = 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_in_range_l1343_134352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_of_specific_line_l1343_134383

/-- The line equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ := -l.c / l.a

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ := -l.c / l.b

/-- The sum of x-intercept and y-intercept of a line -/
noncomputable def intercept_sum (l : Line) : ℝ := x_intercept l + y_intercept l

theorem intercept_sum_of_specific_line :
  let l : Line := { a := 3, b := -4, c := -12 }
  intercept_sum l = 1 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_of_specific_line_l1343_134383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_day_is_monday_l1343_134390

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving BEq, Repr

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

-- Theorem statement
theorem twelfth_day_is_monday 
  (first_day : DayOfWeek)
  (h1 : first_day ≠ DayOfWeek.Friday)
  (h2 : ∃ (last_day : DayOfWeek), last_day ≠ DayOfWeek.Friday ∧ 
        (∃ (n : Nat), n > 28 ∧ n ≤ 31 ∧ advanceDay first_day (n - 1) = last_day))
  (h3 : (List.filter (λ d => d == DayOfWeek.Friday) 
        (List.map (λ i => advanceDay first_day i) (List.range 31))).length = 5) :
  advanceDay first_day 11 = DayOfWeek.Monday := by
  sorry

#eval advanceDay DayOfWeek.Thursday 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_day_is_monday_l1343_134390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1343_134385

-- Define the hyperbola
def is_hyperbola (a b : ℝ) (h : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, h x y ↔ x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 16 * x

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop :=
  x = -4

-- Define the asymptote passing through (√3, 3)
def asymptote_passes_through (a b : ℝ) : Prop :=
  3 = (b / a) * Real.sqrt 3

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) (h : ℝ → ℝ → Prop)
  (ha : a > 0)
  (hb : b > 0)
  (h_hyperbola : is_hyperbola a b h)
  (h_focus : ∃ x y, h x y ∧ directrix x)
  (h_asymptote : asymptote_passes_through a b) :
  a^2 = 4 ∧ b^2 = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1343_134385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_theorem_l1343_134305

theorem election_votes_theorem (total_votes : ℕ) : 
  (0.7 * (total_votes : ℝ) - 0.3 * (total_votes : ℝ) = 200) → total_votes = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_theorem_l1343_134305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_ratings_inequality_l1343_134361

-- Define judge_rating as a variable
variable (judge_rating : ℕ → ℕ → Bool)

theorem competition_ratings_inequality (a b k : ℕ) : 
  b ≥ 3 → 
  Odd b → 
  (∀ (i j : ℕ), i < b → j < b → i ≠ j → 
    (∃ (same_ratings : ℕ), same_ratings ≤ k ∧ 
      (∀ (p : ℕ), p < a → 
        (judge_rating i p = judge_rating j p) → same_ratings > 0))) →
  (k : ℚ) / a ≥ (b - 1 : ℚ) / (2 * b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_ratings_inequality_l1343_134361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequalities_l1343_134321

theorem trigonometric_inequalities :
  (Real.tan 1 > (3 : ℝ) / 2) ∧ (Real.log (Real.cos 1) < Real.sin (Real.cos 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequalities_l1343_134321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_2θ_f_range_l1343_134362

noncomputable section

def θ : ℝ := sorry
def a : ℝ × ℝ := (Real.cos θ - 2 * Real.sin θ, 2)
def b : ℝ × ℝ := (Real.sin θ, 1)

def f (θ : ℝ) : ℝ := (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2

theorem parallel_vectors_tan_2θ :
  (∃ (k : ℝ), a = k • b) →
  Real.tan (2 * θ) = 8 / 15 := by sorry

theorem f_range :
  θ ∈ Set.Icc 0 (Real.pi / 2) →
  Set.range f = Set.Icc 2 ((5 + Real.sqrt 2) / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_2θ_f_range_l1343_134362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l1343_134329

def mySequence (b : ℕ → ℕ) : Prop :=
  b 1 = 2010 ∧ 
  b 2 = 2012 ∧ 
  ∀ n : ℕ, n ≥ 1 → b n + b (n + 1) + b (n + 2) = 2 * n + 3

theorem sequence_1000th_term (b : ℕ → ℕ) (h : mySequence b) : b 1000 = 2674 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l1343_134329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonian_circle_l1343_134324

/-- The Apollonian circle theorem -/
theorem apollonian_circle 
  (A B P : ℝ × ℝ) 
  (hA : A = (-4, 2)) 
  (hB : B = (2, 2)) 
  (h_ratio : dist P A / dist P B = 2) : 
  (P.1 - 4)^2 + (P.2 - 2)^2 = 16 := by
  sorry

where
  dist (p q : ℝ × ℝ) : ℝ := 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonian_circle_l1343_134324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_l1343_134316

/-- The capacity of a tank given its emptying rates and inlet flow rate -/
theorem tank_capacity 
  (empty_time : ℝ) 
  (inlet_rate : ℝ) 
  (combined_empty_time : ℝ) 
  (h1 : empty_time = 6) 
  (h2 : inlet_rate = 3.5) 
  (h3 : combined_empty_time = 8) : 
  (combined_empty_time * (inlet_rate * 60 - combined_empty_time * (inlet_rate * 60 - combined_empty_time * (inlet_rate * 60 - combined_empty_time * (inlet_rate * 60 - combined_empty_time * (inlet_rate * 60 / empty_time)) / empty_time) / empty_time) / empty_time) / empty_time) = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_l1343_134316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_formula_l1343_134359

/-- A pyramid with a rectangular base and specific height -/
structure Pyramid where
  a : ℝ
  base_length : ℝ := a
  base_width : ℝ := 2 * a
  height : ℝ := a
  height_midpoint : height = a

/-- The radius of the circumscribed sphere around the pyramid -/
noncomputable def circumscribed_sphere_radius (p : Pyramid) : ℝ :=
  (p.a * Real.sqrt 89) / 8

/-- Theorem: The radius of the circumscribed sphere around the pyramid is (a * √89) / 8 -/
theorem circumscribed_sphere_radius_formula (p : Pyramid) :
  circumscribed_sphere_radius p = (p.a * Real.sqrt 89) / 8 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_formula_l1343_134359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1343_134315

/-- Circle C in the Cartesian plane -/
def circle_C : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 16}

/-- Line l passing through (1,2) with inclination angle π/6 -/
def line_l : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p = (1 + Real.sqrt 3 / 2 * t, 2 + 1 / 2 * t)}

/-- Point P on the line l -/
def point_P : ℝ × ℝ := (1, 2)

/-- Theorem: The product of distances from P to intersection points of l and C is 11 -/
theorem intersection_distance_product : 
  ∃ A B : ℝ × ℝ, A ∈ circle_C ∩ line_l ∧ B ∈ circle_C ∩ line_l ∧ 
  ((point_P.1 - A.1)^2 + (point_P.2 - A.2)^2) * 
  ((point_P.1 - B.1)^2 + (point_P.2 - B.2)^2) = 121 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1343_134315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_eq_neg_one_f_is_decreasing_l1343_134379

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 / (2^x + 1) + m

-- Part 1: Prove that if f is an odd function, then m = -1
theorem odd_function_implies_m_eq_neg_one (m : ℝ) :
  (∀ x : ℝ, f m (-x) = -(f m x)) → m = -1 :=
by
  sorry

-- Part 2: Prove that f is decreasing for any real m
theorem f_is_decreasing (m : ℝ) :
  ∀ x y : ℝ, x < y → f m x > f m y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_eq_neg_one_f_is_decreasing_l1343_134379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l1343_134338

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + 4*x + y^2 - 5 = 0

-- Define the point M
def M : ℝ × ℝ := (1, 1)

-- Define the two tangent lines
def tangent1 (x : ℝ) : Prop := x = 1
def tangent2 (x y : ℝ) : Prop := 4*x + 3*y - 7 = 0

-- Theorem statement
theorem tangent_lines_to_circle :
  (∃ (x y : ℝ), circle_equation x y ∧ (tangent1 x ∨ tangent2 x y) ∧ (x, y) = M) ∧
  (∀ (x y : ℝ), circle_equation x y → (tangent1 x ∨ tangent2 x y) → (x, y) = M) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l1343_134338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l1343_134348

/-- Represents the travel details between two cities -/
structure TravelDetails where
  distance : ℝ
  time : ℝ

/-- Calculates the average speed given travel details -/
noncomputable def averageSpeed (travel : TravelDetails) : ℝ :=
  travel.distance / travel.time

theorem distance_A_to_B (x : ℝ) : 
  let eddy : TravelDetails := { distance := x, time := 3 }
  let freddy : TravelDetails := { distance := 300, time := 4 }
  averageSpeed eddy / averageSpeed freddy = 2.4 →
  x = 540 := by
  sorry

#check distance_A_to_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l1343_134348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_3_8_digit_difference_l1343_134319

/-- The number of digits in the base-b representation of a positive integer n -/
noncomputable def num_digits (n : ℕ) (b : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.floor (Real.log (n : ℝ) / Real.log (b : ℝ)) + 1

/-- The difference in the number of digits between base-3 and base-8 representations of 987 -/
theorem base_3_8_digit_difference :
  num_digits 987 3 - num_digits 987 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_3_8_digit_difference_l1343_134319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l1343_134327

theorem power_of_three (y : ℕ) (h : (3 : ℝ)^y = 81) : (3 : ℝ)^(y+3) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l1343_134327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_l1343_134345

-- Define the type for polynomials with real coefficients
def RealPolynomial := Polynomial ℝ

-- Define the condition for a, b, and m
def Condition (a b : ℝ) (m : ℕ+) : Prop :=
  a^(m : ℕ) = b^2

-- Define the property that P, Q, and R must satisfy
def SatisfiesProperty (P Q : RealPolynomial) (R : ℝ → ℝ → ℝ) (m : ℕ+) : Prop :=
  ∀ (a b : ℝ), Condition a b m → (P.eval (R a b) = a ∧ Q.eval (R a b) = b)

-- The main theorem
theorem unique_m :
  ∃! (m : ℕ+), ∃ (P Q : RealPolynomial) (R : ℝ → ℝ → ℝ), SatisfiesProperty P Q R m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_l1343_134345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1343_134372

def a : Fin 3 → ℝ := ![-4, 2, 4]
def b : Fin 3 → ℝ := ![-6, 3, -2]

theorem vector_properties :
  let norm_a := Real.sqrt (a 0^2 + a 1^2 + a 2^2)
  let norm_b := Real.sqrt (b 0^2 + b 1^2 + b 2^2)
  let dot_product := a 0 * b 0 + a 1 * b 1 + a 2 * b 2
  norm_a = 6 ∧ (dot_product / (norm_a * norm_b) = 11 / 21) := by
  sorry

#check vector_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1343_134372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1343_134388

theorem negation_of_proposition : 
  (¬ (∀ x : ℝ, 2 * x^2 + 1 > 0)) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1343_134388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l1343_134318

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ),
    (x^2 + a*x + b = 0 ↔ x = -3 - Complex.I * Real.sqrt 7 ∨ x = -3 + Complex.I * Real.sqrt 7) ∧
    (a = 6 ∧ b = 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l1343_134318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1343_134354

-- Define the power function as noncomputable
noncomputable def powerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_through_point (α : ℝ) :
  powerFunction α (Real.sqrt 2) = 2 → powerFunction α 9 = 81 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1343_134354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_85_factorial_l1343_134381

/-- The last two nonzero digits of a natural number -/
def lastTwoNonzeroDigits (n : ℕ) : ℕ := sorry

/-- 85 factorial -/
def factorial85 : ℕ := Nat.factorial 85

theorem last_two_nonzero_digits_of_85_factorial :
  lastTwoNonzeroDigits factorial85 = 68 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_85_factorial_l1343_134381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_time_to_height_l1343_134317

/-- The time taken for a rider on a Ferris wheel to reach a specified height -/
theorem ferris_wheel_time_to_height 
  (radius : ℝ) 
  (revolution_time : ℝ) 
  (height : ℝ) 
  (h_radius : radius = 30) 
  (h_rev_time : revolution_time = 90) 
  (h_height : height = 15) :
  let time_to_height := 
    (2 * Real.pi / 3) * (revolution_time / (2 * Real.pi))
  time_to_height = 30 := by
  sorry

#check ferris_wheel_time_to_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_time_to_height_l1343_134317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_motion_equation_l1343_134336

/-- The equation of motion for a weight supported by a helical spring -/
theorem spring_motion_equation 
  (b : ℝ) -- initial displacement
  (p : ℝ) -- constant related to spring properties
  (s : ℝ → ℝ) -- position function
  (h1 : ∀ t, (deriv (deriv s)) t = -p^2 * s t) -- acceleration equation
  (h2 : s 0 = b) -- initial position
  (h3 : (deriv s) 0 = 0) -- initial velocity
  : ∀ t, s t = b * Real.cos (p * t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_motion_equation_l1343_134336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_fsinA_lt_fcosB_l1343_134302

-- Define a structure for our function f
structure OddDecreasingFunction (α : Type*) [LinearOrder α] [Neg α] where
  f : α → α
  odd : ∀ x, f (-x) = -f x
  decreasing : ∀ x y, x ≤ y → f x ≥ f y

-- Define the theorem
theorem not_always_fsinA_lt_fcosB 
  (f : OddDecreasingFunction ℝ) 
  (A B : ℝ) 
  (acute_A : 0 < A ∧ A < Real.pi/2) 
  (acute_B : 0 < B ∧ B < Real.pi/2) 
  (triangle : A + B < Real.pi) :
  ¬(∀ A B, f.f (Real.sin A) < f.f (Real.cos B)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_fsinA_lt_fcosB_l1343_134302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1343_134343

noncomputable def H (n : ℕ) : ℝ := (Finset.range n).sum (λ k => 1 / (k + 1 : ℝ))

theorem inequality_proof (m : ℕ+) (a : Fin m → ℕ+) :
  (Finset.range m).sum (λ i => H (a i)) ≤ Real.sqrt (Real.pi^2 / 3) *
    Real.sqrt ((Finset.range m).sum (λ i => (i + 1 : ℝ) * (a i))) := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1343_134343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_perimeter_at_max_area_l1343_134374

-- Define a structure for triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  t.a / (Real.cos t.C * Real.sin t.B) = t.b / Real.sin t.B + t.c / Real.cos t.C

-- Define the expression to be maximized in part 1
noncomputable def expression_to_maximize (t : Triangle) : ℝ :=
  Real.sin (t.A + t.B) + Real.sin t.A * Real.cos t.A + Real.cos (t.A - t.B)

-- Theorem for part 1
theorem max_value_theorem (t : Triangle) (h : given_condition t) :
  ∃ (max_val : ℝ), max_val = 5/2 ∧ 
  ∀ (t' : Triangle), given_condition t' → expression_to_maximize t' ≤ max_val := by
  sorry

-- Theorem for part 2
theorem perimeter_at_max_area (t : Triangle) (h1 : given_condition t) (h2 : t.b = Real.sqrt 2) :
  ∃ (max_perimeter : ℝ), max_perimeter = 2 * Real.sqrt (2 + Real.sqrt 2) + Real.sqrt 2 ∧
  ∀ (t' : Triangle), given_condition t' → t'.b = Real.sqrt 2 → 
    t.a * t.c * Real.sin t.B ≤ t'.a * t'.c * Real.sin t'.B →
    t.a + t.b + t.c = max_perimeter := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_perimeter_at_max_area_l1343_134374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l1343_134358

/-- The parabola defined by y = (x-1)^2 + 3 has its vertex at (1,3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = (x - 1)^2 + 3 → (1, 3) = (x, y) ↔ ∀ z : ℝ, y ≤ (z - 1)^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l1343_134358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_property_l1343_134307

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

noncomputable def M (a : ℝ) : ℝ := max (f a 1) (f a 2)

noncomputable def m (a : ℝ) : ℝ := min (f a 1) (f a 2)

theorem exponential_function_property (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  M a = 2 * m a → a = 1/2 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_property_l1343_134307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l1343_134320

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the left branch of the hyperbola -/
def isOnLeftBranch (h : Hyperbola) (p : Point) : Prop :=
  p.x < 0 ∧ p.x^2 - p.y^2 / (24 : ℝ) = 1

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1 / 2 : ℝ) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem hyperbola_triangle_area 
  (h : Hyperbola)
  (f1 f2 p : Point)
  (h_eq : h.a^2 - h.b^2 / (24 : ℝ) = 1)
  (h_focal : distance f1 f2 = 2 * Real.sqrt (h.a^2 + h.b^2))
  (h_left : isOnLeftBranch h p)
  (h_dist : distance p f1 = 3/5 * distance f1 f2) :
  triangleArea p f1 f2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l1343_134320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l1343_134323

/-- The hyperbola equation -/
def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / m = 1

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + 2*x + y^2 = 3

/-- The asymptote of the hyperbola -/
def asymptote (m : ℝ) (x y : ℝ) : Prop :=
  4*x = m.sqrt*y ∨ 4*x = -m.sqrt*y

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The main theorem -/
theorem hyperbola_circle_intersection (m : ℝ) :
  ∃ (x1 y1 x2 y2 : ℝ),
    hyperbola m x1 y1 ∧ hyperbola m x2 y2 ∧
    circle_eq x1 y1 ∧ circle_eq x2 y2 ∧
    asymptote m x1 y1 ∧ asymptote m x2 y2 ∧
    distance x1 y1 x2 y2 = 8 * Real.sqrt 5 / 5 →
    m = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l1343_134323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_eight_eq_zero_l1343_134328

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + x - 2

-- Define the properties of g
noncomputable def g : ℝ → ℝ := sorry

-- g is a cubic polynomial
axiom g_cubic : ∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d

-- g(0) = 2
axiom g_zero : g 0 = 2

-- The roots of g are the cubes of the roots of f
axiom g_roots : ∀ x : ℝ, f x = 0 → g (x^3) = 0

-- Theorem to prove
theorem g_eight_eq_zero : g 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_eight_eq_zero_l1343_134328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1343_134355

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x + Real.pi / 3)

theorem function_properties (ω : ℝ) (h₁ : ω > 0) (h₂ : ∀ x, f ω (x + Real.pi) = f ω x) :
  ω = 1 ∧
  (∃ x₀ ∈ Set.Icc (-Real.pi/6) (Real.pi/2), ∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/2), f ω x ≤ f ω x₀) ∧
  (∃ x₀ ∈ Set.Icc (-Real.pi/6) (Real.pi/2), f ω x₀ = 1 - Real.sqrt 3 / 2) ∧
  (∃ x₁ ∈ Set.Icc (-Real.pi/6) (Real.pi/2), ∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/2), f ω x ≥ f ω x₁) ∧
  (∃ x₁ ∈ Set.Icc (-Real.pi/6) (Real.pi/2), f ω x₁ = -Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1343_134355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_extension_l1343_134349

/-- Represents a non-Hookian spring system -/
structure NonHookianSpring where
  k : ℝ  -- Spring constant
  m : ℝ  -- Mass
  g : ℝ  -- Gravitational acceleration

/-- The force exerted by the spring as a function of displacement -/
noncomputable def spring_force (s : NonHookianSpring) (x : ℝ) : ℝ := -s.k * x^2

/-- The potential energy stored in the spring -/
noncomputable def spring_potential_energy (s : NonHookianSpring) (x : ℝ) : ℝ := 
  -(1/3) * s.k * x^3

/-- The change in gravitational potential energy -/
noncomputable def gravitational_potential_energy (s : NonHookianSpring) (x : ℝ) : ℝ := 
  s.m * s.g * x * (1/2)  -- sin(30°) = 1/2

/-- Theorem: Maximum extension of the spring -/
theorem max_extension (s : NonHookianSpring) :
  ∃ x : ℝ, x > 0 ∧ 
    spring_potential_energy s x = gravitational_potential_energy s x ∧
    x = Real.sqrt ((3 * s.m * s.g) / (2 * s.k)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_extension_l1343_134349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_one_l1343_134303

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then 2*x - 3
  else if x ≤ 1 then 1
  else 2*x - 3

-- Define the set of x values
def S : Set ℝ := Set.Icc (-1 : ℝ) 2 ∪ {5/2}

-- Theorem statement
theorem f_composition_equals_one (x : ℝ) : 
  f (f x) = 1 ↔ x ∈ S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_one_l1343_134303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1343_134391

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  let S₁ := (Real.sqrt 3 / 4) * t.a^2
  let S₂ := (Real.sqrt 3 / 4) * t.b^2
  let S₃ := (Real.sqrt 3 / 4) * t.c^2
  S₁ - S₂ + S₃ = Real.sqrt 3 / 2 ∧
  Real.sin t.B = 1 / 3 ∧
  Real.sin t.A * Real.sin t.C = Real.sqrt 2 / 3

-- Define the theorem
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  (1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 2 / 8) ∧ (t.b = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1343_134391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1343_134340

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.cos (π/2 + α) = 2 * Real.sqrt 2 / 3) 
  (h2 : α ∈ Set.Ioo (π/2) (3*π/2)) : 
  Real.tan α = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1343_134340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_y_measure_l1343_134337

-- Define the geometric configuration
structure GeometricConfig where
  m : Set ℝ × Set ℝ  -- Representing a line as a pair of sets of real numbers
  n : Set ℝ × Set ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  y : ℝ

-- Define a function to check if two lines are parallel
def parallel (l1 l2 : Set ℝ × Set ℝ) : Prop :=
  sorry  -- The actual implementation would go here

-- Define the theorem
theorem angle_y_measure (config : GeometricConfig)
  (h1 : parallel config.m config.n)
  (h2 : config.angle1 = 40)
  (h3 : config.angle2 = 90)
  (h4 : config.angle3 = 40) :
  config.y = 80 := by
  sorry  -- The proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_y_measure_l1343_134337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_max_and_min_l1343_134364

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/a) * x^2 - 4*x + 1

noncomputable def M (a : ℝ) : ℝ := ⨆ (x : ℝ) (h : x ∈ Set.Icc 0 1), f a x

noncomputable def m (a : ℝ) : ℝ := ⨅ (x : ℝ) (h : x ∈ Set.Icc 0 1), f a x

theorem difference_of_max_and_min (a : ℝ) (ha : a ≠ 0) :
  M a - m a = if a < 0 ∨ a > 1/2 then
                4 - 1/a
              else if 0 < a ∧ a ≤ 1/4 then
                1/a + 4*a - 4
              else if 1/4 < a ∧ a ≤ 1/2 then
                4*a
              else
                0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_max_and_min_l1343_134364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_properties_l1343_134313

noncomputable section

variable (m a : ℝ)

-- Define the cone's properties
def surface_area (r l : ℝ) : ℝ := Real.pi * r * (r + l)
def volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- State the main theorem
theorem cone_properties (m_pos : m > 0) (a_pos : a > 0) :
  ∃ (r h : ℝ),
    -- Surface area condition
    surface_area r (Real.sqrt (h^2 + r^2)) = Real.pi * m^2 ∧
    -- Volume condition
    volume r h = (1/3) * Real.pi * a^3 ∧
    -- Radius formula
    r = (1/2) * Real.sqrt ((m^3 + Real.sqrt (m^6 - 8*a^6)) / m) ∧
    -- Height formula
    h = (m * (m^3 - Real.sqrt (m^6 - 8*a^6))) / (2*a^3) ∧
    -- Minimum surface area condition
    (∀ (r' h' : ℝ), volume r' h' = volume r h → surface_area r' (Real.sqrt (h'^2 + r'^2)) ≥ surface_area r (Real.sqrt (h^2 + r^2))) ∧
    -- Maximum volume condition
    (∀ (r' h' : ℝ), surface_area r' (Real.sqrt (h'^2 + r'^2)) = surface_area r (Real.sqrt (h^2 + r^2)) → volume r' h' ≤ volume r h) ∧
    -- Angle between slant height and axis
    Real.arctan (Real.sqrt 2 / 4) = Real.arctan (r / h) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_properties_l1343_134313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_same_monotonicity_l1343_134363

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, (deriv^[2] f) x ≠ 0) ∧ 
  (∀ x, f (f x + x^3) = 2)

/-- Theorem stating the range of k values -/
theorem k_range_for_same_monotonicity 
  (f : ℝ → ℝ) 
  (hf : SpecialFunction f) 
  (k : ℝ) :
  (∀ x y, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → x < y → (f x < f y ↔ (f x - k*x < f y - k*y))) →
  k ≥ 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_same_monotonicity_l1343_134363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_between_powers_of_three_l1343_134373

theorem count_even_between_powers_of_three : 
  (Finset.filter (fun x => Even x ∧ 3^2 < x ∧ x < 3^3) (Finset.range (3^3 - 3^2 + 1))).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_between_powers_of_three_l1343_134373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_interval_l1343_134353

def is_repeating_decimal (q : ℚ) (period : ℕ) : Prop :=
  ∃ k : ℕ, (q * (10^period - 1)).isInt ∧ ∀ m < period, ¬(q * (10^m - 1)).isInt

theorem repeating_decimal_interval (n : ℕ) :
  n < 500 →
  is_repeating_decimal (1 / n) 5 →
  is_repeating_decimal (1 / (n + 3)) 3 →
  201 ≤ n ∧ n ≤ 300 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_interval_l1343_134353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_base_angle_is_45_degrees_l1343_134351

-- Define a regular square pyramid
structure RegularSquarePyramid where
  edge_length : ℝ
  is_regular : edge_length > 0

-- Define the angle between lateral edge and base
noncomputable def lateral_base_angle (p : RegularSquarePyramid) : ℝ := 
  Real.arctan 1

-- Theorem statement
theorem lateral_base_angle_is_45_degrees (p : RegularSquarePyramid) 
  (h : p.edge_length = 2) : 
  lateral_base_angle p = π / 4 := by
  sorry

#check lateral_base_angle_is_45_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_base_angle_is_45_degrees_l1343_134351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_condition_l1343_134333

/-- The equation |x^2 - 2ax + 1| = a has exactly three distinct real roots if and only if a = (1 + √5) / 2 -/
theorem three_roots_condition (a : ℝ) : 
  (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, |x^2 - 2*a*x + 1| = a) ↔ 
  a = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_condition_l1343_134333
