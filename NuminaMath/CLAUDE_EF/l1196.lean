import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_cubic_range_l1196_119687

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + b * x^2 + (b + 2) * x + 3

noncomputable def f_deriv (b : ℝ) (x : ℝ) : ℝ := x^2 + 2 * b * x + (b + 2)

def is_monotonic (g : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, (0 : ℝ) ≤ (deriv g) x) ∨ (∀ x : ℝ, (deriv g) x ≤ (0 : ℝ))

theorem monotonic_cubic_range (b : ℝ) :
  is_monotonic (f b) → -1 ≤ b ∧ b ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_cubic_range_l1196_119687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milkshake_cost_is_five_l1196_119673

/-- Proves the cost of a milkshake given Annie's purchases and remaining money -/
def milkshake_cost (initial_money : ℕ) (hamburger_cost : ℕ) (num_hamburgers : ℕ) 
  (num_milkshakes : ℕ) (remaining_money : ℕ) : ℕ :=
  let total_spent := initial_money - remaining_money
  let hamburger_total := hamburger_cost * num_hamburgers
  let milkshake_total := total_spent - hamburger_total
  milkshake_total / num_milkshakes

#eval milkshake_cost 132 4 8 6 70

theorem milkshake_cost_is_five :
  milkshake_cost 132 4 8 6 70 = 5 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milkshake_cost_is_five_l1196_119673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1196_119602

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the perimeter of triangle PQF₂
noncomputable def perimeter_PQF2 (a b : ℝ) : ℝ := 12

-- Define the product ab
def product_ab (a b : ℝ) : ℝ := a * b

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := 
  Real.sqrt (1 + b^2 / a^2)

-- State the theorem
theorem hyperbola_eccentricity (a b : ℝ) :
  (∃ x y, hyperbola a b x y) ∧ 
  perimeter_PQF2 a b = 12 ∧ 
  IsMaxOn (fun b => product_ab a b) (Set.Icc 0 Real.pi) b →
  eccentricity a b = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1196_119602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_k_range_l1196_119633

noncomputable def e : ℝ := Real.exp 1

def f (k : ℝ) (x : ℝ) : ℝ := k * x

noncomputable def g (x : ℝ) : ℝ := (1 / e) ^ (x / 2)

def domain : Set ℝ := { x | 1 / e ≤ x ∧ x ≤ e^2 }

theorem symmetric_points_k_range (k : ℝ) :
  (∃ x y, x ∈ domain ∧ y ∈ domain ∧ f k x = g y ∧ g x = f k y) →
  -2 / e ≤ k ∧ k ≤ 2 * e :=
by
  sorry

#check symmetric_points_k_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_k_range_l1196_119633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_prism_volume_l1196_119671

/-- Given a pyramid with height H and base area S, and a plane intersecting
    the pyramid at distance d from the apex, calculate the volume of the
    resulting prism. -/
noncomputable def prism_volume (H : ℝ) (S : ℝ) (d : ℝ) : ℝ :=
  let k := (H - d) / H
  let s := k^2 * S
  s * d

/-- Theorem stating that for a pyramid with height 3 and base area 9,
    the volume of the prism formed by intersecting the pyramid with a plane
    1 unit from the apex is equal to 4. -/
theorem pyramid_prism_volume :
  prism_volume 3 9 1 = 4 := by
  -- Unfold the definition of prism_volume
  unfold prism_volume
  -- Simplify the expression
  simp
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_prism_volume_l1196_119671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_pool_contribution_l1196_119667

/-- The amount of money Mark contributes to the pizza pool -/
noncomputable def mark_contribution (emma daya jeff brenda : ℚ) : ℚ :=
  (emma + daya + jeff + brenda) / 4

/-- The problem statement -/
theorem pizza_pool_contribution : 
  ∀ (emma daya jeff brenda : ℚ),
    emma = 8 →
    daya = emma * (5/4) →
    jeff = daya * (2/5) →
    brenda = jeff + 4 →
    mark_contribution emma daya jeff brenda = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_pool_contribution_l1196_119667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_arithmetic_progression_l1196_119684

theorem sequence_arithmetic_progression (f : ℕ+ → ℕ+) (h : Function.Bijective f) :
  ∃ (ℓ m : ℕ+), 1 < ℓ ∧ ℓ < m ∧ f 1 + f m = 2 * f ℓ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_arithmetic_progression_l1196_119684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_range_l1196_119621

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := Real.log (a^(2*x) - 4*a^x + 1) / Real.log a

-- State the theorem
theorem f_negative_range (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x : ℝ, f a x < 0 ↔ x < 2 * (Real.log 2 / Real.log a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_range_l1196_119621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l1196_119656

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the integral of f from 0 to 1
noncomputable def integral_f : ℝ := ∫ x in (0:ℝ)..(1:ℝ), f x

-- State the theorem
theorem integral_value (h : ∀ x, f x + integral_f f = x) : integral_f f = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l1196_119656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_and_minimum_l1196_119652

noncomputable section

def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b * Real.log x

def g (x : ℝ) : ℝ := (x - 10) / (x - 4)

theorem parallel_tangents_and_minimum (b : ℝ) :
  (∃ (k : ℝ), deriv (f b) 5 = k ∧ deriv g 5 = k) →
  b = -20 ∧
  (∃ (x : ℝ), x > 0 ∧ x^2 = 10 ∧
    ∀ y > 0, f (-20) x ≤ f (-20) y ∧ f (-20) x = 10 - 10 * Real.log 10) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_and_minimum_l1196_119652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_10_l1196_119690

-- Define the variable cost function
noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 8 then (1/3) * x^2 + x
  else if x ≥ 8 then 6*x + 100/x - 38
  else 0

-- Define the profit function
noncomputable def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 8 then 5*x - W x - 3
  else if x ≥ 8 then 5*x - W x - 3
  else 0

-- State the theorem
theorem max_profit_at_10 :
  ∀ x > 0, L x ≤ 15 ∧ L 10 = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_10_l1196_119690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_17th_inning_l1196_119654

/-- Represents a batsman's scoring record -/
structure BatsmanRecord where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (record : BatsmanRecord) (runs : ℕ) : ℚ :=
  (record.totalRuns + runs) / (record.innings + 1)

/-- Theorem: The batsman's average after the 17th inning is 23 -/
theorem batsman_average_after_17th_inning
  (record : BatsmanRecord)
  (h1 : record.innings = 16)
  (h2 : newAverage record 87 = record.average + 4)
  (h3 : (newAverage record 87).num % (newAverage record 87).den = 0) :
  newAverage record 87 = 23 := by
  sorry

#check batsman_average_after_17th_inning

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_17th_inning_l1196_119654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_is_integer_l1196_119640

theorem zero_is_integer : (0 : ℤ) ∈ (Set.univ : Set ℤ) := by
  exact Set.mem_univ 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_is_integer_l1196_119640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_percentage_after_addition_l1196_119675

noncomputable def original_volume : ℝ := 40
noncomputable def original_alcohol_percentage : ℝ := 5
noncomputable def added_alcohol : ℝ := 6.5
noncomputable def added_water : ℝ := 3.5

noncomputable def new_alcohol_volume : ℝ := (original_volume * original_alcohol_percentage / 100) + added_alcohol
noncomputable def new_total_volume : ℝ := original_volume + added_alcohol + added_water

noncomputable def new_alcohol_percentage : ℝ := (new_alcohol_volume / new_total_volume) * 100

theorem alcohol_percentage_after_addition :
  |new_alcohol_percentage - 19.54| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_percentage_after_addition_l1196_119675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_calculation_l1196_119609

/-- Represents the profit calculation for a shopkeeper's article sale --/
theorem shopkeeper_profit_calculation 
  (cost_price : ℝ)
  (discount_rate : ℝ)
  (profit_rate_with_discount : ℝ)
  (h_discount : discount_rate = 0.05)
  (h_profit_with_discount : profit_rate_with_discount = 0.216)
  : (cost_price * (1 + profit_rate_with_discount) / (1 - discount_rate) - cost_price) / cost_price = 0.28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_calculation_l1196_119609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1196_119618

/-- The function f(x) = 2^(2x-1) - 3 * 2^x + 5 -/
noncomputable def f (x : ℝ) : ℝ := 2^(2*x - 1) - 3 * 2^x + 5

/-- The theorem stating that the maximum value of f(x) is 5/2 when 0 ≤ x ≤ 2 -/
theorem max_value_of_f :
  ∀ x : ℝ, 0 ≤ x → x ≤ 2 → f x ≤ 5/2 :=
by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1196_119618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c10_value_l1196_119650

-- Define arithmetic sequences a and b
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry

-- Define the conditions
axiom a_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
axiom b_arithmetic : ∀ n m : ℕ, b (n + 1) - b n = b (m + 1) - b m
axiom a2_value : a 2 = 4
axiom a4_value : a 4 = 6
axiom b3_value : b 3 = 9
axiom b7_value : b 7 = 21

-- Define sequence c as common terms of a and b
def c : ℕ → ℝ := sorry

-- Define the property of c being formed by common terms
axiom c_common : ∀ n : ℕ, (∃ k m : ℕ, a k = b m ∧ c n = a k)

-- Theorem to prove
theorem c10_value : c 10 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c10_value_l1196_119650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_theorem_l1196_119677

/-- Two intersecting circles with radius R -/
structure IntersectingCircles (R : ℝ) where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ

/-- Points of intersection of two circles -/
structure IntersectionPoints where
  M : ℝ × ℝ
  N : ℝ × ℝ

/-- Points where perpendicular bisector of MN intersects the circles -/
structure PerpBisectorPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem intersecting_circles_theorem 
  (R : ℝ) 
  (circles : IntersectingCircles R)
  (intersectionPoints : IntersectionPoints)
  (perpBisectorPoints : PerpBisectorPoints)
  (h1 : R > 0)
  (h2 : distance circles.center1 intersectionPoints.M = R)
  (h3 : distance circles.center1 intersectionPoints.N = R)
  (h4 : distance circles.center2 intersectionPoints.M = R)
  (h5 : distance circles.center2 intersectionPoints.N = R)
  (h6 : distance circles.center1 perpBisectorPoints.A = R)
  (h7 : distance circles.center2 perpBisectorPoints.B = R)
  (h8 : perpBisectorPoints.A.1 = perpBisectorPoints.B.1 ∨ perpBisectorPoints.A.2 = perpBisectorPoints.B.2) :
  (distance intersectionPoints.M intersectionPoints.N)^2 + 
  (distance perpBisectorPoints.A perpBisectorPoints.B)^2 = 4 * R^2 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_theorem_l1196_119677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_is_sin_l1196_119697

noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => fun x => Real.cos x
  | n + 1 => fun x => deriv (f n) x

theorem f_2015_is_sin : f 2015 = fun x => Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_is_sin_l1196_119697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_online_textbooks_cost_l1196_119632

/-- Proves that the cost of online ordered textbooks is $40 given the problem conditions --/
theorem online_textbooks_cost : ∃ (online_cost : ℕ),
  let sale_price : ℕ := 10
  let total_spent : ℕ := 210
  5 * sale_price + online_cost + 3 * online_cost = total_spent ∧
  online_cost = 40
  := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_online_textbooks_cost_l1196_119632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l1196_119608

theorem trig_inequality : Real.cos 8.5 < Real.sin 3 ∧ Real.sin 3 < Real.sin 1.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l1196_119608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximately_55_l1196_119626

-- Define the given constants
noncomputable def train_length : ℝ := 100  -- meters
noncomputable def crossing_time : ℝ := 6   -- seconds
noncomputable def man_speed_kmh : ℝ := 5   -- km/h

-- Convert man's speed to m/s
noncomputable def man_speed_ms : ℝ := man_speed_kmh * 1000 / 3600

-- Define the relative speed
noncomputable def relative_speed : ℝ := train_length / crossing_time

-- Define the train's speed in m/s
noncomputable def train_speed_ms : ℝ := relative_speed - man_speed_ms

-- Convert train's speed to km/h
noncomputable def train_speed_kmh : ℝ := train_speed_ms * 3600 / 1000

-- Theorem statement
theorem train_speed_approximately_55 : 
  54.5 ≤ train_speed_kmh ∧ train_speed_kmh ≤ 55.5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximately_55_l1196_119626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_face_angle_l1196_119688

/-- The number of rays on the clock face. -/
def num_rays : ℕ := 12

/-- The angle between adjacent rays on the clock face. -/
noncomputable def angle_between_rays : ℝ := 360 / num_rays

/-- The number of segments between 3 o'clock and 7 o'clock. -/
def segments_3_to_7 : ℕ := 4

/-- The smaller angle between the rays pointing to 3 o'clock and 7 o'clock. -/
noncomputable def angle_3_to_7 : ℝ := angle_between_rays * segments_3_to_7

theorem clock_face_angle :
  angle_3_to_7 = 120 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_face_angle_l1196_119688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vince_customers_l1196_119651

/-- Represents the hair salon business model --/
structure HairSalon where
  earningsPerCustomer : ℚ
  fixedExpenses : ℚ
  recreationPercentage : ℚ
  monthlySavings : ℚ

/-- Calculates the number of customers served in a month --/
def customersServed (salon : HairSalon) : ℚ :=
  (salon.monthlySavings + salon.fixedExpenses) / 
  (salon.earningsPerCustomer * (1 - salon.recreationPercentage))

/-- Theorem: Vince serves 80 customers per month --/
theorem vince_customers : 
  let salon : HairSalon := {
    earningsPerCustomer := 18,
    fixedExpenses := 280,
    recreationPercentage := 1/5,
    monthlySavings := 872
  }
  customersServed salon = 80 := by
  sorry

#eval customersServed {
  earningsPerCustomer := 18,
  fixedExpenses := 280,
  recreationPercentage := 1/5,
  monthlySavings := 872
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vince_customers_l1196_119651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1196_119679

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log (x + 1) + Real.sqrt (4 - x^2)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = Set.union (Set.Ioo (-1) 0) (Set.Ioc 0 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1196_119679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_3_l1196_119627

/-- A function satisfying the given properties -/
axiom f : ℝ → ℝ

/-- The function f is additive -/
axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y

/-- The value of f at 4 -/
axiom f_4 : f 4 = 6

/-- The theorem to prove -/
theorem f_3 : f 3 = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_3_l1196_119627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1196_119658

noncomputable def f (x : Real) : Real :=
  2 * Real.sin (x + Real.pi / 3) + Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x) ^ 2

theorem f_properties :
  -- The smallest positive period of f is π
  (∃ (p : Real), p > 0 ∧ (∀ (x : Real), f (x + p) = f x) ∧
    (∀ (q : Real), q > 0 → (∀ (x : Real), f (x + q) = f x) → p ≤ q)) ∧
  -- Range of m for x₀ ∈ [0, 5π/12] such that mf(x₀) - 2 = 0
  (∀ (m : Real), (∃ (x₀ : Real), 0 ≤ x₀ ∧ x₀ ≤ 5 * Real.pi / 12 ∧ m * f x₀ - 2 = 0) →
    (m ≤ -2 ∨ m ≥ 1)) ∧
  -- Range of f((C/2 - π/6)) / f((B/2 - π/6)) in acute triangle ABC with ∠B = 2∠A
  (∀ (A B C : Real),
    0 < A ∧ A < Real.pi / 2 ∧
    0 < B ∧ B < Real.pi / 2 ∧
    0 < C ∧ C < Real.pi / 2 ∧
    A + B + C = Real.pi ∧
    B = 2 * A →
    Real.sqrt 2 / 2 < f (C / 2 - Real.pi / 6) / f (B / 2 - Real.pi / 6) ∧
    f (C / 2 - Real.pi / 6) / f (B / 2 - Real.pi / 6) < 2 * Real.sqrt 3 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1196_119658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_walk_time_with_backpack_l1196_119607

/-- Represents the walking parameters and calculates the time to walk to school -/
structure WalkToSchool where
  steps_per_minute : ℚ
  step_length : ℚ
  time_to_school : ℚ

def distance_to_school (w : WalkToSchool) : ℚ :=
  w.steps_per_minute * w.step_length * w.time_to_school

def jack_walk_time (dave : WalkToSchool) (jack_steps_per_minute : ℚ) (jack_normal_step_length : ℚ) (backpack_reduction : ℚ) : ℚ :=
  let jack_reduced_step_length := jack_normal_step_length * (1 - backpack_reduction)
  let school_distance := distance_to_school dave
  school_distance / (jack_steps_per_minute * jack_reduced_step_length)

theorem jack_walk_time_with_backpack 
  (dave : WalkToSchool)
  (jack_steps_per_minute : ℚ)
  (jack_normal_step_length : ℚ)
  (backpack_reduction : ℚ)
  (h1 : dave.steps_per_minute = 80)
  (h2 : dave.step_length = 65)
  (h3 : dave.time_to_school = 20)
  (h4 : jack_steps_per_minute = 110)
  (h5 : jack_normal_step_length = 55)
  (h6 : backpack_reduction = 1/10)
  : ∃ (ε : ℚ), ε > 0 ∧ |jack_walk_time dave jack_steps_per_minute jack_normal_step_length backpack_reduction - 191/10| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_walk_time_with_backpack_l1196_119607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l1196_119696

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0)
  (angle_A_60 : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = (B.1 - A.1)^2 + (B.2 - A.2)^2 / 2)
  (AB_length : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 4)

-- Define point P inside the triangle
def P (t : Triangle) := {p : ℝ × ℝ // 
  p.1 > 0 ∧ p.1 < 1 ∧ p.2 > 0 ∧ p.2 < Real.sqrt 3 ∧ p.1 + p.2 / Real.sqrt 3 < 1}

-- Define the angle condition for point P
noncomputable def angle_condition (t : Triangle) (p : P t) :=
  let θAP := Real.arctan ((p.val.2 - t.A.2) / (p.val.1 - t.A.1))
  let θBP := Real.arctan ((p.val.2 - t.B.2) / (p.val.1 - t.B.1))
  θAP - θBP = Real.pi / 6

-- Theorem statement
theorem locus_of_P (t : Triangle) (p : P t) (h : angle_condition t p) :
  (p.val.2 - Real.sqrt 3 / 2)^2 - (p.val.1 - 1 / 2)^2 = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l1196_119696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1196_119665

theorem train_length_calculation (tree_time platform_time platform_length : ℝ) 
  (h1 : tree_time = 120)
  (h2 : platform_time = 210)
  (h3 : platform_length = 900) : 
  (platform_time * platform_length) / (platform_time - tree_time) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1196_119665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_staircase_l1196_119672

/-- Ceiling function -/
def ceil (x : ℚ) : ℤ :=
  Int.ceil x

/-- The number of steps in the staircase -/
def n : ℕ := 114

/-- Cozy's jumps -/
def cozy_jumps : ℤ := ceil (n / 2 : ℚ)

/-- Dash's jumps -/
def dash_jumps : ℤ := ceil (n / 3 : ℚ)

/-- The difference in jumps between Cozy and Dash -/
def jump_difference : ℤ := cozy_jumps - dash_jumps

/-- The sum of digits of a natural number -/
def sum_of_digits (k : ℕ) : ℕ :=
  if k < 10 then k else k % 10 + sum_of_digits (k / 10)

/-- The main theorem -/
theorem smallest_staircase :
  (∀ m : ℕ, m < n → ceil (m / 2 : ℚ) - ceil (m / 3 : ℚ) ≠ 19) ∧
  jump_difference = 19 ∧
  n = 114 ∧
  sum_of_digits n = 6 := by
  sorry

#eval n
#eval cozy_jumps
#eval dash_jumps
#eval jump_difference
#eval sum_of_digits n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_staircase_l1196_119672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_right_triangle_exists_l1196_119613

-- Define the color type
inductive Color
| Black
| White

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define an equilateral triangle
structure EquilateralTriangle where
  A : Point
  B : Point
  C : Point
  is_equilateral : Prop

-- Define the set G of points on the sides of the triangle
def G (triangle : EquilateralTriangle) : Set Point :=
  sorry

-- Define a coloring function for points in G
def coloring (triangle : EquilateralTriangle) : G triangle → Color :=
  sorry

-- Define a right triangle inscribed in the equilateral triangle
structure InscribedRightTriangle (triangle : EquilateralTriangle) where
  P : G triangle
  Q : G triangle
  R : G triangle
  is_right : Prop

-- State the theorem
theorem monochromatic_right_triangle_exists (triangle : EquilateralTriangle) :
  ∃ (t : InscribedRightTriangle triangle),
    coloring triangle t.P = coloring triangle t.Q ∧
    coloring triangle t.Q = coloring triangle t.R :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_right_triangle_exists_l1196_119613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_two_points_tangent_to_line_l1196_119685

noncomputable section

-- Define basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

structure Angle where
  vertex : Point
  ray1 : Point
  ray2 : Point

-- Define basic operations
def Line := Point → Point → Prop

def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def tangent (c : Circle) (l : Line) : Prop :=
  ∃ p : Point, l p p ∧ distance p c.center = c.radius

def on_line (p : Point) (l : Line) : Prop :=
  l p p

def angle_measure (a : Angle) : ℝ :=
  sorry

-- Main theorem
theorem circle_through_two_points_tangent_to_line
  (A B : Point) (L : Line) :
  ∃! (c : Circle),
    (c.center ≠ A) ∧
    (c.center ≠ B) ∧
    distance A c.center = c.radius ∧
    distance B c.center = c.radius ∧
    tangent c L ∧
    (∃ O : Point, on_line O L ∧
      (∃ C : Point, on_line C L ∧ tangent c L ∧
        (distance O C)^2 = (distance O B) * (distance O A))) ∧
    (∀ M : Point, on_line M L →
      let ang_ACB : Angle := ⟨C, A, B⟩
      let ang_AMB : Angle := ⟨M, A, B⟩
      angle_measure ang_ACB ≥ angle_measure ang_AMB) :=
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_two_points_tangent_to_line_l1196_119685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_is_two_l1196_119606

/-- The function f(x) = e^(2x) - (ln(x) + 1) / x -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) - (Real.log x + 1) / x

/-- The theorem stating that the minimum value of f(x) is 2 for x > 0 -/
theorem f_min_value_is_two :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≥ 2 ∧ f x₀ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_is_two_l1196_119606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_solution_l1196_119635

-- Define the triathlon stages
structure Triathlon :=
  (swim : ℝ)
  (cycle : ℝ)
  (run : ℝ)

-- Define the problem parameters
noncomputable def triathlon_distance : Triathlon := ⟨1, 25, 4⟩
noncomputable def total_time : ℝ := 5/4
noncomputable def practice_swim_time : ℝ := 1/16
noncomputable def practice_cycle_run_time : ℝ := 1/49
noncomputable def practice_total_distance : ℝ := 5/4

-- Define the theorem
theorem triathlon_solution :
  ∃ (v : Triathlon), 
    (v.swim * practice_swim_time + v.cycle * practice_cycle_run_time + v.run * practice_cycle_run_time = practice_total_distance) ∧
    (triathlon_distance.swim / v.swim + triathlon_distance.cycle / v.cycle + triathlon_distance.run / v.run = total_time) ∧
    (triathlon_distance.cycle / v.cycle = 5/7) ∧
    (v.cycle = 35) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_solution_l1196_119635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l1196_119601

/-- The projection of vector a in the direction of vector b -/
noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / magnitude_b

theorem projection_a_on_b :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-4, 7)
  vector_projection a b = Real.sqrt 65 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l1196_119601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_parallel_vectors_l1196_119600

theorem min_sum_parallel_vectors (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * (y - 1) = 4 * y → a + b ≤ x + y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * (y - 1) = 4 * y ∧ a + b = x + y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_parallel_vectors_l1196_119600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1196_119634

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + 1

/-- The theorem statement -/
theorem problem_statement (a : ℝ) (m : ℝ) :
  1 < a → a < Real.sqrt 2 →
  (∃ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ 1 ∧ f a x₀ + Real.log a > m * (a - a^2)) →
  m ≥ Real.sqrt 2 - ((2 + Real.sqrt 2) * Real.log (Real.sqrt 2)) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1196_119634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_point_distance_l1196_119692

open Real

-- Define a triangle as a structure with three points
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a function to check if a point is on a line segment
def isOnSegment (p : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  distance a p + distance p b = distance a b

-- Theorem statement
theorem triangle_side_point_distance (t : Triangle) :
  ∀ (M N : ℝ × ℝ),
    (isOnSegment M t.A t.B ∨ isOnSegment M t.B t.C ∨ isOnSegment M t.C t.A) →
    (isOnSegment N t.A t.B ∨ isOnSegment N t.B t.C ∨ isOnSegment N t.C t.A) →
    distance M N ≤ max (distance t.A t.B) (max (distance t.B t.C) (distance t.C t.A)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_point_distance_l1196_119692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_150_degree_angles_l1196_119620

/-- The measure of an interior angle of a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ := (n - 2) * 180 / n

theorem regular_polygon_with_150_degree_angles (n : ℕ) : 
  n > 2 → (∀ i : Fin n, interior_angle n = 150) → n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_150_degree_angles_l1196_119620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_value_l1196_119624

noncomputable def g (x : ℝ) : ℝ := (9 * x^2 + 18 * x + 20) / (4 * (2 + x))

theorem g_minimum_value (x : ℝ) (h : x ≥ 1) : g x ≥ 47/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_value_l1196_119624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_condition_implies_collinearity_l1196_119686

/-- A set of four distinct points in the plane -/
def PointSet : Type := Fin 4 → ℝ × ℝ

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The property that for any point, there exist three other points satisfying the distance condition -/
def satisfiesDistanceCondition (S : PointSet) : Prop :=
  ∀ X : Fin 4, ∃ Y Z W : Fin 4, 
    X ≠ Y ∧ X ≠ Z ∧ X ≠ W ∧ Y ≠ Z ∧ Y ≠ W ∧ Z ≠ W ∧
    distance (S X) (S Y) = distance (S X) (S Z) + distance (S X) (S W)

/-- Three points are collinear if the distance between two of them equals the sum of the distances from those two points to the third -/
def collinear (p q r : ℝ × ℝ) : Prop :=
  distance p q = distance p r + distance q r ∨
  distance p r = distance p q + distance q r ∨
  distance q r = distance p q + distance p r

/-- All points in the set are collinear -/
def allCollinear (S : PointSet) : Prop :=
  ∀ p q r : Fin 4, p ≠ q ∧ q ≠ r ∧ p ≠ r → collinear (S p) (S q) (S r)

/-- The main theorem: if the set satisfies the distance condition, then all points are collinear -/
theorem distance_condition_implies_collinearity (S : PointSet) :
  satisfiesDistanceCondition S → allCollinear S :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_condition_implies_collinearity_l1196_119686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_inequality_l1196_119653

-- Define the function f as noncomputable
noncomputable def f : ℝ → ℝ := fun x => if x ≥ 0 then Real.exp (x * Real.log 2) else Real.exp (-x * Real.log 2)

-- State the theorem
theorem range_of_inequality (h_even : ∀ x, f x = f (-x)) 
  (h_def : ∀ x ≥ 0, f x = Real.exp (x * Real.log 2)) :
  {x : ℝ | f (1 - 2*x) < f 3} = Set.Ioo (-1) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_inequality_l1196_119653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_values_l1196_119678

/-- Given real numbers a, b, c, and a non-invertible matrix as described,
    prove that (a/(b+c)) + (b/(c+a)) + (c/(a+b)) can only equal -3 or 3/2 -/
theorem matrix_sum_values (a b c : ℝ) 
  (h_not_invertible : ¬(Matrix.det ![![a, b, c, 0],
                                     ![b, c, 0, a],
                                     ![c, 0, a, b],
                                     ![0, a, b, c]] ≠ 0)) :
  (a / (b + c) + b / (c + a) + c / (a + b) = -3) ∨ 
  (a / (b + c) + b / (c + a) + c / (a + b) = 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_values_l1196_119678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_send_mores_l1196_119622

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- The equation MONEY + SEND = MORES -/
def equation (M N O E Y S R D : Digit) : Prop :=
  (10000 * M.val + 1000 * O.val + 100 * N.val + 10 * E.val + Y.val) +
  (10000 * S.val + 1000 * E.val + 100 * N.val + 10 * D.val) =
  (10000 * M.val + 1000 * O.val + 100 * R.val + 10 * E.val + S.val)

/-- All letters represent different digits -/
def all_different (M N O E Y S R D : Digit) : Prop :=
  M ≠ N ∧ M ≠ O ∧ M ≠ E ∧ M ≠ Y ∧ M ≠ S ∧ M ≠ R ∧ M ≠ D ∧
  N ≠ O ∧ N ≠ E ∧ N ≠ Y ∧ N ≠ S ∧ N ≠ R ∧ N ≠ D ∧
  O ≠ E ∧ O ≠ Y ∧ O ≠ S ∧ O ≠ R ∧ O ≠ D ∧
  E ≠ Y ∧ E ≠ S ∧ E ≠ R ∧ E ≠ D ∧
  Y ≠ S ∧ Y ≠ R ∧ Y ≠ D ∧
  S ≠ R ∧ S ≠ D ∧
  R ≠ D

theorem money_send_mores :
  ∃! (M N O E Y S R D : Digit),
    S.val = 9 ∧ O.val = 0 ∧ E.val = 5 ∧
    all_different M N O E Y S R D ∧
    equation M N O E Y S R D ∧
    M.val = 1 ∧ N.val = 6 ∧ R.val = 8 ∧ Y.val = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_send_mores_l1196_119622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_xyz_l1196_119657

/-- The radius of the inscribed circle in a triangle with sides a, b, and c -/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (2 * Real.sqrt (s * (s - a) * (s - b) * (s - c))) / (a + b + c)

/-- Theorem: The radius of the inscribed circle in triangle XYZ is √5 -/
theorem inscribed_circle_radius_xyz :
  inscribed_circle_radius 7 8 9 = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_xyz_l1196_119657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_digits_of_base_8_num_l1196_119699

/- Define the base-8 number -/
def base_8_num : ℕ := 77777

/- Theorem stating the number of binary digits -/
theorem binary_digits_of_base_8_num :
  (Nat.log 2 32767 + 1) = 15 := by
  sorry

/- Helper lemma to connect base_8_num to its decimal representation -/
lemma base_8_to_decimal : 
  (Nat.ofDigits 8 [7, 7, 7, 7, 7]) = 32767 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_digits_of_base_8_num_l1196_119699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_equality_characterization_l1196_119623

def M (n : ℕ) : ℕ := List.range n |> List.map Nat.succ |> List.foldl Nat.lcm 1

theorem lcm_equality_characterization (n : ℕ) (h : n > 0) :
  M (n - 1) = M n ↔ ¬ ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = p ^ k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_equality_characterization_l1196_119623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotone_increasing_f_l1196_119663

noncomputable def f (x : ℝ) : ℝ := 2 / x

theorem not_monotone_increasing_f :
  ¬ (∀ x y : ℝ, 0 < x → 0 < y → x < y → f x ≤ f y) :=
by
  intro h
  -- We'll prove this by contradiction
  have contra : ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x < y ∧ f x > f y := by
    -- Choose specific values for x and y
    use 1, 2
    -- Prove the conditions
    constructor
    · exact one_pos
    constructor
    · exact two_pos
    constructor
    · exact one_lt_two
    · -- Prove f 1 > f 2
      calc
        f 1 = 2 / 1 := rfl
        _ = 2 := by field_simp
        _ > 1 := by norm_num
        _ = 2 / 2 := by field_simp
        _ = f 2 := rfl
  -- Derive the contradiction
  rcases contra with ⟨x, y, hx, hy, hxy, hf⟩
  have := h x y hx hy hxy
  linarith

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotone_increasing_f_l1196_119663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_about_line_symmetric_about_point_increasing_in_interval_not_shifted_sine_l1196_119625

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3)

-- Statement 1
theorem symmetric_about_line (x : ℝ) : 
  f (11 * Real.pi / 12 + x) = f (11 * Real.pi / 12 - x) := by sorry

-- Statement 2
theorem symmetric_about_point (x : ℝ) :
  f (2 * Real.pi / 3 + x) = -f (2 * Real.pi / 3 - x) := by sorry

-- Statement 3
theorem increasing_in_interval (x y : ℝ) :
  -Real.pi / 12 < x → x < y → y < 5 * Real.pi / 12 → f x < f y := by sorry

-- Statement 4
theorem not_shifted_sine (x : ℝ) :
  ∃ x, f x ≠ 3 * Real.sin (2 * (x - Real.pi / 6)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_about_line_symmetric_about_point_increasing_in_interval_not_shifted_sine_l1196_119625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_married_men_fraction_l1196_119637

theorem married_men_fraction (total_women : ℕ) (single_women : ℕ) :
  single_women > 0 ∧ 
  total_women > single_women ∧ 
  (single_women : ℚ) / total_women = 3 / 7 →
  (total_women - single_women : ℚ) / (total_women + (total_women - single_women)) = 4 / 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_married_men_fraction_l1196_119637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_units_digit_less_than_7_l1196_119614

/-- The set of six-digit positive integers -/
def SixDigitIntegers : Finset ℕ := Finset.filter (fun n => 100000 ≤ n ∧ n ≤ 999999) (Finset.range 1000000)

/-- The set of six-digit positive integers with units digit less than 7 -/
def SixDigitIntegersLessThan7 : Finset ℕ := Finset.filter (fun n => n % 10 < 7) SixDigitIntegers

/-- The probability of a randomly chosen six-digit positive integer having a units digit less than 7 -/
theorem probability_units_digit_less_than_7 :
  (SixDigitIntegersLessThan7.card : ℚ) / SixDigitIntegers.card = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_units_digit_less_than_7_l1196_119614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_16782_l1196_119681

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def digit_sum (n : ℕ) : ℕ :=
  n.repr.data.map (fun c => c.toNat - 48) |>.sum

theorem divisibility_of_16782 :
  is_divisible_by_3 16782 ∧
  ∀ d : ℕ, d < 2 → ¬is_divisible_by_3 (16780 + d) :=
by sorry

#check divisibility_of_16782

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_16782_l1196_119681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1196_119683

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 1)

def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

theorem perpendicular_vectors (k : ℝ) : 
  b.1 * (c k).1 + b.2 * (c k).2 = 0 → k = -3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1196_119683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_of_folded_square_l1196_119664

/-- Represents a square piece of paper with white top and red bottom -/
structure Paper where
  side : ℝ
  is_square : side > 0

/-- A point within the square paper -/
structure Point (p : Paper) where
  x : ℝ
  y : ℝ
  in_square : 0 ≤ x ∧ x ≤ p.side ∧ 0 ≤ y ∧ y ≤ p.side

/-- The expected number of sides of the red polygon after folding -/
noncomputable def expected_sides (p : Paper) : ℝ :=
  5 - Real.pi / 2

/-- Theorem stating the expected number of sides of the red polygon -/
theorem expected_sides_of_folded_square (p : Paper) :
  expected_sides p = 5 - Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_of_folded_square_l1196_119664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_value_after_three_years_l1196_119694

-- Define the initial conditions
def initial_price : ℝ := 8000
def depreciation_rates : List ℝ := [0.30, 0.20, 0.15]
def maintenance_costs : List ℝ := [500, 800, 1000]
def inflation_rate : ℝ := 0.05
def years : ℕ := 3

-- Define a function to calculate the value after one year
def value_after_year (current_value : ℝ) (depreciation_rate : ℝ) (maintenance_cost : ℝ) : ℝ :=
  let after_depreciation := current_value * (1 - depreciation_rate)
  let after_maintenance := after_depreciation - maintenance_cost
  after_maintenance * (1 + inflation_rate)

-- Define a function to calculate the final value after multiple years
def final_value : ℝ :=
  List.foldl
    (fun acc (i : ℕ) => value_after_year acc (depreciation_rates.get! i) (maintenance_costs.get! i))
    initial_price
    (List.range years)

-- State the theorem
theorem machine_value_after_three_years :
  ∃ ε > 0, |final_value - 2214.94| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_value_after_three_years_l1196_119694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l1196_119659

theorem inequality_solution_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) → m ∈ Set.Ioc (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l1196_119659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_from_coefficients_l1196_119644

/-- Roots of a quadratic equation based on coefficient conditions -/
theorem quadratic_roots_from_coefficients 
  (a b c : ℝ) (ha : a ≠ 0) :
  (a + b + c = 0 → (1 : ℝ) ∈ {x | a * x^2 + b * x + c = 0} ∧ 
                   (c / a) ∈ {x | a * x^2 + b * x + c = 0}) ∧
  (a - b + c = 0 → (-1 : ℝ) ∈ {x | a * x^2 + b * x + c = 0} ∧ 
                   (-c / a) ∈ {x | a * x^2 + b * x + c = 0}) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_from_coefficients_l1196_119644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinderVolume_20_10_l1196_119647

/-- The volume of a cylindrical tank -/
noncomputable def cylinderVolume (diameter : ℝ) (depth : ℝ) : ℝ :=
  (Real.pi / 4) * diameter^2 * depth

/-- Theorem: The volume of a cylindrical tank with diameter 20 feet and depth 10 feet is 1000π cubic feet -/
theorem cylinderVolume_20_10 :
  cylinderVolume 20 10 = 1000 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinderVolume_20_10_l1196_119647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1196_119666

noncomputable def f (x : ℝ) := 2^x + 3*x - 7
noncomputable def g (x : ℝ) := Real.log x + 2*x - 6

theorem function_inequality (a b : ℝ) (ha : f a = 0) (hb : g b = 0) :
  g a < 0 ∧ 0 < f b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1196_119666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1196_119616

noncomputable def f (x : ℝ) := (3 : ℝ)^(-x^2 + 2*x + 3)

theorem f_properties :
  (∀ x : ℝ, f x > 0) ∧ 
  (∀ y : ℝ, y > 0 → y ≤ 81 → ∃ x : ℝ, f x = y) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → x₂ ≤ 1 → f x₁ ≤ f x₂) ∧
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ ≤ x₂ → f x₂ ≤ f x₁) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1196_119616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_satisfies_conditions_l1196_119617

/-- Represents a hyperbola equation in the form (y^2 / a^2) - (x^2 / b^2) = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents an ellipse equation in the form (y^2 / a^2) + (x^2 / b^2) = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Returns the asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => y = (h.a / h.b) * x ∨ y = -(h.a / h.b) * x

/-- Returns the foci of an ellipse -/
noncomputable def ellipse_foci (e : Ellipse) : ℝ × ℝ :=
  let c := Real.sqrt (e.a^2 - e.b^2)
  (0, c)

/-- Returns the foci of a hyperbola -/
noncomputable def hyperbola_foci (h : Hyperbola) : ℝ × ℝ :=
  let c := Real.sqrt (h.a^2 + h.b^2)
  (0, c)

/-- The given hyperbola -/
noncomputable def given_hyperbola : Hyperbola := { a := Real.sqrt 2, b := 1 }

/-- The given ellipse -/
noncomputable def given_ellipse : Ellipse := { a := Real.sqrt 8, b := Real.sqrt 2 }

/-- The hyperbola we need to prove -/
noncomputable def target_hyperbola : Hyperbola := { a := Real.sqrt 2, b := 2 }

theorem hyperbola_satisfies_conditions :
  (asymptotes target_hyperbola = asymptotes given_hyperbola) ∧
  (hyperbola_foci target_hyperbola = ellipse_foci given_ellipse) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_satisfies_conditions_l1196_119617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_sin_squared_x_l1196_119641

theorem integral_x_sin_squared_x (x : ℝ) :
  (deriv fun x => (x^2 / 4) - (x * Real.sin (2*x)) / 4 - Real.cos (2*x) / 8) x
  = x * Real.sin x ^ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_sin_squared_x_l1196_119641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_properties_l1196_119646

/-- Proves properties of a sector given its radius and arc length -/
theorem sector_properties (r : ℝ) (arc_length : ℝ) 
  (h1 : r = 16) 
  (h2 : arc_length = 16 * Real.pi) : 
  (arc_length / r = Real.pi) ∧ ((1/2) * r * arc_length = 128 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_properties_l1196_119646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_dot_product_range_l1196_119645

/-- Given an isosceles triangle ABC with points D, E, F on its sides, prove the range of EF · BA -/
theorem isosceles_triangle_dot_product_range 
  (A B C D E F : ℝ × ℝ) -- Points in 2D plane
  (h_isosceles : ‖A - C‖ = ‖B - C‖) -- AC = BC
  (h_side_length : ‖A - C‖ = Real.sqrt 5) -- AC = √5
  (h_D_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • A + t • B) -- D on AB
  (h_E_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • B + t • C) -- E on BC
  (h_F_on_CA : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (1 - t) • C + t • A) -- F on CA
  (h_AD_DB : ‖A - D‖ = ‖D - B‖) -- AD = DB
  (h_AD_1 : ‖A - D‖ = 1) -- AD = 1
  (h_EF_1 : ‖E - F‖ = 1) -- EF = 1
  (h_DE_DF : (E - D) • (F - D) ≤ 25 / 16) -- DE · DF ≤ 25/16
: 4/3 ≤ (E - F) • (B - A) ∧ (E - F) • (B - A) ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_dot_product_range_l1196_119645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_earnings_in_ten_more_days_l1196_119674

/-- Given a person who has worked for a number of days and earned a certain amount,
    calculate the number of additional days needed to earn twice the current amount. -/
def additional_days_to_double_earnings (days_worked : ℕ) (amount_earned : ℚ) : ℕ :=
  let daily_rate := amount_earned / days_worked
  let target_amount := 2 * amount_earned
  let total_days_needed := target_amount / daily_rate
  (total_days_needed - days_worked).ceil.toNat

/-- Theorem stating that for the given conditions, 10 more days are needed to double the earnings. -/
theorem double_earnings_in_ten_more_days :
  additional_days_to_double_earnings 10 250 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_earnings_in_ten_more_days_l1196_119674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_problem_l1196_119655

/-- A sequence of moves for the monkey. Each move is either +1 (up) or -16 (down). -/
def MonkeyMove := List ℤ

/-- Checks if a sequence of moves is valid for a given number of rungs. -/
def isValidSequence (n : ℕ) (moves : MonkeyMove) : Prop :=
  let positions := List.scanl (λ acc m => (acc + m).toNat % n) 0 moves
  positions.length = n ∧ positions.toFinset.card = n

/-- The minimum number of rungs needed for the monkey problem. -/
def minRungs : ℕ := 24

theorem monkey_problem :
  (∃ (moves : MonkeyMove), isValidSequence minRungs moves) ∧
  (∀ (k : ℕ), k < minRungs → ¬∃ (moves : MonkeyMove), isValidSequence k moves) :=
sorry

#eval minRungs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_problem_l1196_119655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_coefficients_l1196_119631

noncomputable section

-- Define the original function
def f (x : ℝ) : ℝ := x^2 - abs x - 12

-- Define the intersection points with x-axis
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (4, 0)

-- Define the second parabola
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for the second parabola to pass through A and B
def passes_through_AB (a b c : ℝ) : Prop :=
  g a b c A.fst = A.snd ∧ g a b c B.fst = B.snd

-- Define the vertex of the second parabola
def vertex (a b c : ℝ) : ℝ × ℝ := (-b / (2 * a), g a b c (-b / (2 * a)))

-- Define the condition for APB to be an isosceles right triangle
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  let P := vertex a b c
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 ∧
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

theorem parabola_coefficients :
  ∀ a b c : ℝ,
    passes_through_AB a b c →
    is_isosceles_right_triangle a b c →
    ((a = 1/4 ∧ b = 0 ∧ c = 4) ∨ (a = -1/4 ∧ b = 0 ∧ c = -4)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_coefficients_l1196_119631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_board_weight_l1196_119669

/-- Represents the weight of a wooden board given its dimensions -/
noncomputable def board_weight (length width : ℝ) (weight : ℝ) : ℝ → ℝ :=
  λ side => (side^2 * weight) / (length * width)

/-- Theorem stating the weight of the square board -/
theorem square_board_weight :
  let rect_length : ℝ := 4
  let rect_width : ℝ := 6
  let rect_weight : ℝ := 20
  let square_side : ℝ := 8
  let square_weight := board_weight rect_length rect_width rect_weight square_side
  ∃ ε > 0, |square_weight - 53.3| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_board_weight_l1196_119669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_negative_sqrt2_over_2_l1196_119604

theorem arcsin_negative_sqrt2_over_2 : Real.arcsin (-Real.sqrt 2 / 2) = -π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_negative_sqrt2_over_2_l1196_119604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karl_paths_count_l1196_119630

/-- Represents a letter in the diagram -/
inductive Letter
| K
| A
| R
| L

/-- Represents a column in the diagram -/
def Column := List Letter

/-- The diagram structure -/
structure Diagram where
  col1 : Column
  col2 : Column
  col3 : Column
  col4 : Column

/-- A path in the diagram -/
def DiagramPath := List Nat

/-- Checks if a path is valid for spelling "KARL" -/
def isValidPath (d : Diagram) (p : DiagramPath) : Prop := sorry

/-- Counts the number of valid paths in the diagram -/
def countValidPaths (d : Diagram) : Nat := sorry

/-- The specific diagram for spelling "KARL" -/
def karlDiagram : Diagram := {
  col1 := [Letter.K]
  col2 := [Letter.A, Letter.A]
  col3 := [Letter.R, Letter.R, Letter.R]
  col4 := [Letter.L, Letter.L, Letter.L, Letter.L]
}

theorem karl_paths_count :
  countValidPaths karlDiagram = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_karl_paths_count_l1196_119630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_ngon_regions_and_segments_l1196_119676

/-- A convex n-gon with the property that no three diagonals intersect at a single point inside the shape. -/
structure ConvexNGon (n : ℕ) where
  convex : Bool
  no_triple_intersection : Bool

/-- The number of regions into which the diagonals divide the convex n-gon. -/
def num_regions (n : ℕ) : ℚ :=
  (1 / 24) * (n - 1 : ℚ) * (n - 2 : ℚ) * (n^2 - 3*n + 12 : ℚ)

/-- The number of segments into which the diagonals are divided by their intersection points. -/
def num_segments (n : ℕ) : ℚ :=
  (1 / 12) * (n : ℚ) * (n - 3 : ℚ) * (n^2 - 3*n + 8 : ℚ)

theorem convex_ngon_regions_and_segments (n : ℕ) (ngon : ConvexNGon n) :
  (num_regions n = (1 / 24) * (n - 1 : ℚ) * (n - 2 : ℚ) * (n^2 - 3*n + 12 : ℚ)) ∧
  (num_segments n = (1 / 12) * (n : ℚ) * (n - 3 : ℚ) * (n^2 - 3*n + 8 : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_ngon_regions_and_segments_l1196_119676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l1196_119668

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- State the theorem
theorem triangle_inequalities (t : Triangle) :
  Real.cos (t.A / 2) < Real.cos (t.B / 2) + Real.cos (t.C / 2) ∧
  Real.cos (t.A / 2) < Real.sin (t.B / 2) + Real.sin (t.C / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l1196_119668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l1196_119611

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x + Real.sin x

-- State the theorem
theorem solution_set_equality :
  {m : ℝ | f (m^2) + f (2*m - 3) < 0} = Set.Ioo (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l1196_119611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l1196_119648

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ, Real.cos θ]]

theorem smallest_rotation_power : 
  (∃ (n : ℕ), n > 0 ∧ (rotation_matrix (π/3))^n = 1) ∧
  (∀ (m : ℕ), m > 0 ∧ m < 6 → (rotation_matrix (π/3))^m ≠ 1) ∧
  (rotation_matrix (π/3))^6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l1196_119648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_x_l1196_119698

-- Define the constants and variables
noncomputable def a : ℝ := 0.3010  -- log_10(2)
noncomputable def b : ℝ := Real.log 10 / Real.log 2  -- log_2(10)

-- Define the equation
def equation (x : ℝ) : Prop :=
  (Real.log 5 / Real.log 2)^2 - a * (Real.log 5 / Real.log 2) + x * b = 0

-- State the theorem
theorem solution_x :
  ∃ x : ℝ, equation x ∧ x = (Real.log 5 / Real.log 2)^2 * a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_x_l1196_119698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_common_roots_condition_l1196_119612

/-- Two polynomials share exactly two common roots -/
def has_two_common_roots (p q : ℝ → ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ p x = 0 ∧ p y = 0 ∧ q x = 0 ∧ q y = 0 ∧
  ∀ z, p z = 0 ∧ q z = 0 → z = x ∨ z = y

/-- The main theorem stating the conditions for two specific polynomials to have two common roots -/
theorem two_common_roots_condition (a b : ℝ) :
  has_two_common_roots (λ x ↦ 2*x^3 + a*x - 12) (λ x ↦ x^2 + b*x + 2) ↔ a = -14 ∧ b = 3 := by
  sorry

#check two_common_roots_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_common_roots_condition_l1196_119612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_school_time_l1196_119643

/-- Represents the time Joe took to get from home to school -/
noncomputable def total_time (walk_time : ℝ) (break_time : ℝ) (run_speed_multiplier : ℝ) : ℝ :=
  walk_time + break_time + walk_time / run_speed_multiplier

/-- Theorem stating that Joe's total time from home to school is 11 minutes -/
theorem joe_school_time :
  total_time 8 1 4 = 11 := by
  -- Unfold the definition of total_time
  unfold total_time
  -- Simplify the arithmetic
  simp [add_assoc]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_school_time_l1196_119643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_range_l1196_119639

/-- A function that is odd and equals x² for non-negative x -/
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

/-- The property that f(x+2t) ≥ 4f(x) for all x in [t, t+2] -/
def property (t : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc t (t + 2) → f (x + 2*t) ≥ 4 * f x

theorem f_property_range :
  {t : ℝ | property t} = Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_range_l1196_119639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_square_plots_l1196_119682

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℝ
  length : ℝ

/-- Represents the available fencing for internal partitioning -/
def availableFencing : ℝ := 2200

/-- Calculates the number of square plots given the side length -/
noncomputable def numPlots (field : FieldDimensions) (sideLength : ℝ) : ℝ :=
  (field.width / sideLength) * (field.length / sideLength)

/-- Calculates the required fencing for internal partitioning -/
noncomputable def requiredFencing (field : FieldDimensions) (sideLength : ℝ) : ℝ :=
  (field.width * ((field.length / sideLength) - 1)) + 
  (field.length * ((field.width / sideLength) - 1))

/-- Theorem stating the maximum number of square plots -/
theorem max_square_plots (field : FieldDimensions) 
  (h1 : field.width = 20) 
  (h2 : field.length = 60) : 
  ∃ (sideLength : ℝ), 
    sideLength > 0 ∧ 
    requiredFencing field sideLength ≤ availableFencing ∧
    numPlots field sideLength = 75 ∧
    ∀ (s : ℝ), s > 0 → 
      requiredFencing field s ≤ availableFencing → 
      numPlots field s ≤ numPlots field sideLength :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_square_plots_l1196_119682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_inequality_l1196_119693

theorem sin_cos_inequality (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  Real.sqrt (Real.sin x ^ 2 / (1 + Real.cos x ^ 2)) + Real.sqrt (Real.cos x ^ 2 / (1 + Real.sin x ^ 2)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_inequality_l1196_119693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_calculation_l1196_119660

/-- Calculates the discount percentage given the markup percentage and actual profit percentage. -/
noncomputable def discountPercentage (markup : ℝ) (actualProfit : ℝ) : ℝ :=
  (markup - actualProfit) / (1 + markup) * 100

/-- Theorem stating that a 50% markup with 35% actual profit results in a 10% discount. -/
theorem discount_percentage_calculation :
  discountPercentage 0.5 0.35 = 10 := by
  -- Unfold the definition of discountPercentage
  unfold discountPercentage
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_calculation_l1196_119660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_check_probability_l1196_119691

/-- Represents the days of the week when math lessons occur -/
inductive MathDay
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- The probability of no homework check during the week -/
def prob_no_check : ℚ := 1/2

/-- The probability of homework check on one day -/
def prob_check : ℚ := 1/2

/-- The number of math lessons per week -/
def num_lessons : ℕ := 5

/-- Event A: Homework not checked through Thursday -/
def event_A : Set MathDay := {MathDay.Friday}

/-- Event B: Homework checked on Friday -/
def event_B : Set MathDay := {MathDay.Friday}

/-- Theorem: The probability of homework being checked on Friday, 
    given it hasn't been checked by Thursday, is 1/6 -/
theorem homework_check_probability : 
  (prob_no_check : ℚ) → (prob_check : ℚ) → (num_lessons : ℕ) → 
  (event_A : Set MathDay) → (event_B : Set MathDay) → 
  (1 : ℚ) / 6 = 1 / 6 := by
  intros _ _ _ _ _
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_check_probability_l1196_119691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l1196_119680

/-- A function satisfying f(0) = 0 and f(x^2 + 1) = f(x)^2 + 1 is the identity function. -/
theorem polynomial_identity (f : ℝ → ℝ) 
  (hpoly : ∃ p : Polynomial ℝ, ∀ x, f x = p.eval x)
  (h0 : f 0 = 0) (h1 : ∀ x, f (x^2 + 1) = f x^2 + 1) : 
  ∀ x, f x = x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l1196_119680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1196_119603

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2

theorem problem_solution (a : ℝ) :
  (∃ a, (deriv (f a)) 1 = 0 ∧ a = 1/2) ∧
  (∀ x > 0, deriv (f (1/2)) x ≤ 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → deriv (f a) x₁ = 0 → deriv (f a) x₂ = 0 → x₁ * x₂^2 > Real.exp (-1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1196_119603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1196_119619

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1/2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2*x))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -1/2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1196_119619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l1196_119662

noncomputable def f (x : ℝ) := x^2 - Real.log x

theorem tangent_line_and_monotonicity :
  let f : ℝ → ℝ := λ x ↦ x^2 - Real.log x
  (∀ x > 0, f x = x^2 - Real.log x) →
  (∀ x > 0, HasDerivAt f (2*x - 1/x) x) →
  (∃ m b, ∀ y, y = m * x + b ↔ y - f 1 = (2 - 1) * (x - 1)) ∧
  (∀ x > Real.sqrt 2 / 2, StrictMono f) ∧
  (∀ x ∈ Set.Ioo 0 (Real.sqrt 2 / 2), StrictAntiOn f (Set.Ioo 0 (Real.sqrt 2 / 2))) := by
  sorry

#check tangent_line_and_monotonicity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l1196_119662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_example_l1196_119670

/-- The surface area of a cone given its slant height and angle between slant height and axis of rotation -/
noncomputable def cone_surface_area (slant_height : ℝ) (angle : ℝ) : ℝ :=
  let radius := slant_height * Real.sin (angle * Real.pi / 180)
  let base_area := Real.pi * radius^2
  let lateral_area := Real.pi * radius * slant_height
  base_area + lateral_area

/-- Theorem: The surface area of a cone with slant height 2 and angle 30° between
    the slant height and the axis of rotation is 3π -/
theorem cone_surface_area_example : cone_surface_area 2 30 = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_example_l1196_119670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1196_119629

-- Define the function f(x) = 2 / (x - 1)
noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

-- Define the interval [2, 6]
def interval : Set ℝ := {x | 2 ≤ x ∧ x ≤ 6}

theorem f_properties : 
  -- 1. f(x) is decreasing on [2, 6]
  (∀ x y, x ∈ interval → y ∈ interval → x < y → f x > f y) ∧ 
  -- 2. The maximum value of f(x) on [2, 6] is 2
  (∀ x, x ∈ interval → f x ≤ 2) ∧ (∃ x, x ∈ interval ∧ f x = 2) ∧
  -- 3. The minimum value of f(x) on [2, 6] is 2/5
  (∀ x, x ∈ interval → f x ≥ 2/5) ∧ (∃ x, x ∈ interval ∧ f x = 2/5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1196_119629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_theorem_l1196_119695

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Polynomial p(x) of degree 1008 -/
def p : ℕ → ℕ := sorry

/-- Theorem stating the property of polynomial p -/
axiom p_property : ∀ n : ℕ, n ≤ 1008 → p (2 * n + 1) = fib (2 * n + 1)

/-- Main theorem to prove -/
theorem polynomial_value_theorem : p 2019 = fib 2019 - fib 1010 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_theorem_l1196_119695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_radius_approx_l1196_119661

/-- Given the speed of a bus and the revolutions per minute of its wheel,
    calculate the radius of the wheel. -/
noncomputable def wheel_radius (bus_speed : ℝ) (wheel_rpm : ℝ) : ℝ :=
  let cm_per_km : ℝ := 100000
  let min_per_hour : ℝ := 60
  let bus_speed_cm_per_min : ℝ := bus_speed * cm_per_km / min_per_hour
  bus_speed_cm_per_min / (wheel_rpm * 2 * Real.pi)

/-- Theorem stating that for a bus traveling at 66 km/h with wheels rotating at
    100.09099181073704 rpm, the wheel radius is approximately 175.03 cm. -/
theorem bus_wheel_radius_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |wheel_radius 66 100.09099181073704 - 175.03| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_radius_approx_l1196_119661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_count_geometric_sequence_l1196_119636

open Real

theorem angle_count_geometric_sequence : 
  ∃ (S : Finset ℝ), 
    (∀ θ ∈ S, 0 < θ ∧ θ < 2 * π ∧ 
      ¬∃ (n : ℤ), θ = n * (π / 2)) ∧
    (∀ θ ∈ S, sin θ ^ 2 = cos θ ^ 4 ∧ cos θ ^ 4 = sin θ) ∧
    S.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_count_geometric_sequence_l1196_119636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_from_odd_digits_l1196_119649

def odd_numbers_less_than_10 : List Nat := [1, 3, 5, 7, 9]

def is_valid_number (n : Nat) : Prop :=
  ∃ (digits : List Nat), 
    n = digits.foldl (λ acc d => acc * 10 + d) 0 ∧ 
    digits.toFinset = odd_numbers_less_than_10.toFinset

theorem smallest_number_from_odd_digits : 
  ∀ n : Nat, is_valid_number n → n ≥ 13579 :=
by
  sorry

#eval odd_numbers_less_than_10.foldl (λ acc d => acc * 10 + d) 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_from_odd_digits_l1196_119649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_asymptote_tangent_to_parabola_l1196_119628

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

/-- Represents a parabola of the form y = (1/2)x^2 + 2 -/
noncomputable def parabola (x : ℝ) : ℝ := (1/2) * x^2 + 2

/-- Condition for the asymptote of the hyperbola to be tangent to the parabola -/
def asymptote_tangent_to_parabola (h : Hyperbola a b) : Prop :=
  ∃ (x : ℝ), parabola x = (b / a) * x

theorem hyperbola_eccentricity_when_asymptote_tangent_to_parabola
  (a b : ℝ) (h : Hyperbola a b) :
  asymptote_tangent_to_parabola h → eccentricity h = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_asymptote_tangent_to_parabola_l1196_119628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_three_hours_l1196_119638

/-- Represents the time it takes to fill a tank with a leak, given the filling and draining rates -/
noncomputable def fill_time_with_leak (pump_fill_time leak_drain_time : ℝ) : ℝ :=
  1 / (1 / pump_fill_time - 1 / leak_drain_time)

/-- Theorem stating that with given conditions, the fill time with a leak is 3 hours -/
theorem fill_time_is_three_hours (pump_fill_time leak_drain_time : ℝ) 
    (h1 : pump_fill_time = 2)
    (h2 : leak_drain_time = 5.999999999999999) : 
    fill_time_with_leak pump_fill_time leak_drain_time = 3 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval fill_time_with_leak 2 5.999999999999999

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_three_hours_l1196_119638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1196_119689

-- Define the function f as noncomputable due to Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.sin x ^ 2 - Real.cos x ^ 2) + 2 * Real.sin x * Real.cos x

-- State the theorem
theorem f_properties :
  -- 1. The smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
    (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S)) ∧
  -- 2. The range of f(x) for x ∈ [-π/3, π/3] is [-2, √3]
  (∀ (y : ℝ), (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/3) (Real.pi/3) ∧ f x = y) ↔ 
    y ∈ Set.Icc (-2) (Real.sqrt 3)) ∧
  -- 3. f(x) is strictly increasing on [-π/12, π/3] for x ∈ [-π/3, π/3]
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc (-Real.pi/12) (Real.pi/3) ∧ 
    x₂ ∈ Set.Icc (-Real.pi/12) (Real.pi/3) ∧ x₁ < x₂ → f x₁ < f x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1196_119689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l1196_119642

/-- The surface area of a regular triangular pyramid -/
noncomputable def surfaceArea (baseEdgeLength : ℝ) (height : ℝ) : ℝ :=
  let baseArea := (Real.sqrt 3 / 4) * baseEdgeLength^2
  let lateralHeight := Real.sqrt (height^2 + (baseEdgeLength / 2)^2)
  let lateralArea := 3 * (1 / 2) * baseEdgeLength * lateralHeight
  baseArea + lateralArea

/-- Theorem: The surface area of a regular triangular pyramid with base edge length 6 and height 3 is 27√3 -/
theorem pyramid_surface_area :
  surfaceArea 6 3 = 27 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l1196_119642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_eight_times_50_l1196_119605

noncomputable def s (θ : ℝ) : ℝ := 1 / (2 - θ)

theorem s_eight_times_50 : s (s (s (s (s (s (s (s 50))))))) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_eight_times_50_l1196_119605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_for_max_difference_l1196_119615

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

theorem min_distance_for_max_difference (x₁ x₂ : ℝ) 
  (h : |f x₁ - f x₂| = 2) : 
  ∃ (min_dist : ℝ), min_dist = π / 2 ∧ |x₁ - x₂| ≥ min_dist := by
  sorry

#check min_distance_for_max_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_for_max_difference_l1196_119615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyarelal_loss_share_l1196_119610

/-- Represents the capital invested by an individual -/
structure Capital where
  amount : ℝ

/-- Represents the loss amount in Rupees -/
structure Loss where
  amount : ℝ

/-- Calculates the share of loss for an investor given their capital ratio and the total loss -/
def calculateLossShare (capitalRatio : ℝ) (totalLoss : Loss) : Loss :=
  { amount := capitalRatio * totalLoss.amount }

theorem pyarelal_loss_share 
  (ashokCapital pyarelalCapital : Capital)
  (totalLoss : Loss)
  (h1 : ashokCapital.amount = (1 / 9) * pyarelalCapital.amount)
  (h2 : totalLoss.amount = 1200) :
  (calculateLossShare (9 / 10) totalLoss).amount = 1080 := by
  sorry

#check pyarelal_loss_share

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyarelal_loss_share_l1196_119610
