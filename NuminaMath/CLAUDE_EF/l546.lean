import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juniors_to_freshmen_ratio_l546_54695

/-- Represents the number of students in a class -/
structure ClassSize where
  count : ℕ

/-- Represents the participation rate of a class in AMC-10 -/
structure ParticipationRate where
  rate : ℚ

/-- Represents a class (freshmen, sophomores, or juniors) -/
structure SchoolClass where
  size : ClassSize
  participationRate : ParticipationRate

variable (freshmen sophomores juniors : SchoolClass)

/-- The number of participants from each class is equal -/
axiom equal_participants :
  freshmen.size.count * freshmen.participationRate.rate =
  sophomores.size.count * sophomores.participationRate.rate ∧
  freshmen.size.count * freshmen.participationRate.rate =
  juniors.size.count * juniors.participationRate.rate

/-- Participation rates for each class -/
axiom participation_rates :
  freshmen.participationRate.rate = 3/7 ∧
  sophomores.participationRate.rate = 5/7 ∧
  juniors.participationRate.rate = 1/2

/-- The ratio of juniors to freshmen is 7/6 -/
theorem juniors_to_freshmen_ratio :
  (juniors.size.count : ℚ) / freshmen.size.count = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juniors_to_freshmen_ratio_l546_54695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_angle_ratio_is_right_triangle_l546_54649

theorem triangle_with_angle_ratio_is_right_triangle 
  (A B C : Real) 
  (triangle : A + B + C = 180) 
  (ratio : ∃ k : Real, A = k ∧ B = 2*k ∧ C = 3*k) : 
  C = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_angle_ratio_is_right_triangle_l546_54649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l546_54608

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 - 2 / (3^x + 1)

-- Theorem statement
theorem f_properties :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ z : ℝ, z ∈ Set.Icc (-1) 2 → f z ∈ Set.Icc (-1/2) (4/5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l546_54608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_line_l546_54622

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 5

-- Define point M
def point_M : ℝ × ℝ := (1, 0)

-- Define the line
def my_line (x y : ℝ) : Prop := x + y = 1

-- Theorem statement
theorem shortest_chord_line :
  ∃ (l : ℝ × ℝ → Prop),
    (∀ (x y : ℝ), l (x, y) ↔ my_line x y) ∧
    (∀ (c : ℝ × ℝ → Prop),
      (∀ (x y : ℝ), c (x, y) ↔ my_circle x y) →
      ∀ (p q : ℝ × ℝ),
        c p ∧ c q ∧ l p ∧ l q ∧ 
        point_M.1 = (p.1 + q.1) / 2 ∧ point_M.2 = (p.2 + q.2) / 2 →
        ∀ (r s : ℝ × ℝ),
          c r ∧ c s ∧ r ≠ s ∧
          point_M.1 = (r.1 + s.1) / 2 ∧ point_M.2 = (r.2 + s.2) / 2 →
          (p.1 - q.1)^2 + (p.2 - q.2)^2 ≤ (r.1 - s.1)^2 + (r.2 - s.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_line_l546_54622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_rectangles_area_prove_overlapping_rectangles_area_l546_54619

theorem overlapping_rectangles_area 
  (total_length : ℝ) 
  (left_length right_length : ℝ) 
  (left_only_area right_only_area : ℝ) 
  (overlapping_area : ℝ) : Prop :=
  total_length = 16 ∧
  left_length = 9 ∧
  right_length = 7 ∧
  left_only_area = 27 ∧
  right_only_area = 18 ∧
  left_length + right_length = total_length ∧
  ∃ w : ℝ, 
    left_only_area = left_length * w - overlapping_area ∧ 
    right_only_area = right_length * w - overlapping_area ∧
    left_length * w / (right_length * w) = (left_only_area + overlapping_area) / (right_only_area + overlapping_area) ∧
  overlapping_area = 13.5

theorem prove_overlapping_rectangles_area : 
  ∃ total_length left_length right_length left_only_area right_only_area overlapping_area : ℝ,
  overlapping_rectangles_area total_length left_length right_length left_only_area right_only_area overlapping_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_rectangles_area_prove_overlapping_rectangles_area_l546_54619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_area_inscribed_circle_area_l546_54680

-- Define the side length of the equilateral triangle
def side_length : ℝ := 12

-- Define the area of a circle given its radius
noncomputable def circle_area (radius : ℝ) : ℝ := Real.pi * radius^2

-- Define the radius of the circumscribed circle for an equilateral triangle
noncomputable def circumscribed_radius (s : ℝ) : ℝ := s / Real.sqrt 3

-- Define the radius of the inscribed circle for an equilateral triangle
noncomputable def inscribed_radius (s : ℝ) : ℝ := (Real.sqrt 3 / 6) * s

-- Theorem for the area of the circumscribed circle
theorem circumscribed_circle_area :
  circle_area (circumscribed_radius side_length) = 48 * Real.pi := by sorry

-- Theorem for the area of the inscribed circle
theorem inscribed_circle_area :
  circle_area (inscribed_radius side_length) = 12 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_area_inscribed_circle_area_l546_54680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_range_l546_54609

-- Define the function f(x) = √(1-2x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - 2 * x)

-- Define the inverse function of f
noncomputable def f_inverse : ℝ → ℝ := f⁻¹

-- Theorem statement
theorem inverse_function_range :
  Set.range f_inverse = Set.Iic (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_range_l546_54609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_run_time_l546_54685

/-- The number of hours per day a TV is run, given its power consumption, electricity cost, and weekly running cost. -/
noncomputable def tv_hours_per_day (power_watts : ℝ) (electricity_cost_cents_per_kwh : ℝ) (weekly_cost_cents : ℝ) : ℝ :=
  let power_kw := power_watts / 1000
  let weekly_kwh := weekly_cost_cents / electricity_cost_cents_per_kwh
  let weekly_hours := weekly_kwh / power_kw
  weekly_hours / 7

/-- Theorem stating that under the given conditions, the TV is run for 4 hours per day. -/
theorem tv_run_time (power_watts : ℝ) (electricity_cost_cents_per_kwh : ℝ) (weekly_cost_cents : ℝ)
    (h1 : power_watts = 125)
    (h2 : electricity_cost_cents_per_kwh = 14)
    (h3 : weekly_cost_cents = 49) :
    tv_hours_per_day power_watts electricity_cost_cents_per_kwh weekly_cost_cents = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_run_time_l546_54685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l546_54638

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2

-- Define the point of tangency
def point : ℝ × ℝ := (1, 2)

-- Define the proposed tangent line equation
def tangent_line (x y : ℝ) : Prop := 4 * x - y - 2 = 0

-- Theorem statement
theorem tangent_line_is_correct : 
  let (x₀, y₀) := point
  ∀ x y : ℝ, 
    tangent_line x y ↔ 
    (y - y₀ = (deriv curve x₀) * (x - x₀) ∧ 
     y₀ = curve x₀) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l546_54638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_vs_running_time_difference_l546_54642

/-- Calculates the difference in minutes per mile between two speeds --/
noncomputable def timeDifference (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ) : ℝ :=
  (time2 * 60 / distance2) - (time1 * 60 / distance1)

theorem walking_vs_running_time_difference :
  timeDifference 20 4 12 5 = 13 := by
  -- Unfold the definition of timeDifference
  unfold timeDifference
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_vs_running_time_difference_l546_54642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_epidemic_ends_without_initial_immunity_epidemic_can_continue_with_initial_immunity_l546_54698

-- Define the state of an elf
inductive ElfState
  | Infected
  | Sick
  | ImmuneRecovered
  | NonImmuneRecovered

-- Define the population as a function from elf identifiers to their states
def Population := ℕ → ElfState

-- Define the social network as a relation between elves
def SocialNetwork := ℕ → ℕ → Prop

-- Function to update the population state for one day
def updatePopulation (pop : Population) (network : SocialNetwork) : Population :=
  sorry

-- Predicate to check if the epidemic has ended (no sick elves)
def epidemicEnded (pop : Population) : Prop :=
  ∀ n, pop n ≠ ElfState.Sick

-- Theorem 1: The epidemic will eventually end if no one is initially immune
theorem epidemic_ends_without_initial_immunity 
  (initial_pop : Population) 
  (network : SocialNetwork)
  (h_no_initial_immunity : ∀ n, initial_pop n ≠ ElfState.ImmuneRecovered) :
  ∃ k, epidemicEnded (Nat.iterate (updatePopulation · network) k initial_pop) :=
by sorry

-- Theorem 2: With initial immunity, the epidemic can continue indefinitely
theorem epidemic_can_continue_with_initial_immunity :
  ∃ initial_pop : Population, ∃ network : SocialNetwork,
    (∃ n, initial_pop n = ElfState.ImmuneRecovered) ∧
    (∀ k, ∃ n, (Nat.iterate (updatePopulation · network) k initial_pop) n = ElfState.Sick) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_epidemic_ends_without_initial_immunity_epidemic_can_continue_with_initial_immunity_l546_54698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l546_54673

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  Real.sin t.A ^ 2 = Real.sin t.B * Real.sin t.C

theorem triangle_properties (t : Triangle) (h : satisfiesCondition t) :
  (t.A = Real.pi / 3 → t.B = Real.pi / 3) ∧
  (t.b * t.c = 1 → ∃ (S : Real), S ≤ Real.sqrt 3 / 4 ∧ 
    ∀ (S' : Real), S' = 1/2 * t.b * t.c * Real.sin t.A → S' ≤ S) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l546_54673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l546_54681

theorem sufficient_not_necessary : 
  (∀ x : ℝ, (abs x < 1 → x^2 - 2*x - 3 < 0)) ∧ 
  (∃ x : ℝ, (x^2 - 2*x - 3 < 0 ∧ ¬(abs x < 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l546_54681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_point_l546_54624

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem max_distance_ellipse_to_point :
  ∀ x y : ℝ, ellipse x y →
  distance x y 0 3 ≤ 4 ∧
  ∃ x' y' : ℝ, ellipse x' y' ∧ distance x' y' 0 3 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_point_l546_54624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l546_54694

-- Define the triangle and its properties
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧
  0 < B ∧ B < Real.pi/2 ∧
  0 < C ∧ C < Real.pi/2 ∧
  A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C)

-- State the theorem
theorem triangle_side_value
  (A B C : ℝ) (a b c : ℝ)
  (h_triangle : Triangle A B C a b c)
  (h_sin_A : Real.sin A = 2 * Real.sqrt 2 / 3)
  (h_a : a = 2)
  (h_relation : c * Real.cos B + b * Real.cos C = 2 * a * Real.cos B) :
  b = 3 * Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l546_54694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a1_a5_l546_54669

/-- Given a sequence a, where S_n is the sum of its first n terms -/
noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := n^2 + a 1 / 2

/-- The main theorem -/
theorem sum_a1_a5 (a : ℕ → ℝ) (h : ∀ n, S a n = n^2 + a 1 / 2) : 
  a 1 + a 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a1_a5_l546_54669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l546_54674

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  C = π / 3 →
  Real.sin B = 2 * Real.sin A →
  (1 / 2) * a * b * Real.sin C = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l546_54674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_one_expressions_l546_54633

theorem max_equal_one_expressions 
  (a b c x y z : ℝ) 
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ x ∧ a ≠ y ∧ a ≠ z ∧
            b ≠ c ∧ b ≠ x ∧ b ≠ y ∧ b ≠ z ∧
            c ≠ x ∧ c ≠ y ∧ c ≠ z ∧
            x ≠ y ∧ x ≠ z ∧ y ≠ z) : 
  ∃ (S : Finset (ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ)), 
    S.card = 6 ∧ 
    (∀ f ∈ S, f ∈ [
      (fun a b c x y z => a*x + b*y + c*z),
      (fun a b c x y z => a*x + b*z + c*y),
      (fun a b c x y z => a*y + b*x + c*z),
      (fun a b c x y z => a*y + b*z + c*x),
      (fun a b c x y z => a*z + b*x + c*y),
      (fun a b c x y z => a*z + b*y + c*x)
    ]) ∧
    (∀ T : Finset (ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ), T ⊆ S →
      (∀ f ∈ T, f a b c x y z = 1) → T.card ≤ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_one_expressions_l546_54633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l546_54677

noncomputable def f (x : ℝ) : ℝ := (1/4)^x - (1/2)^(x-1) + 2

theorem f_range :
  ∀ x ∈ Set.Icc (-2 : ℝ) 1,
    1 ≤ f x ∧ f x ≤ 10 ∧
    (∃ x₁ ∈ Set.Icc (-2 : ℝ) 1, f x₁ = 1) ∧
    (∃ x₂ ∈ Set.Icc (-2 : ℝ) 1, f x₂ = 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l546_54677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_prime_probability_l546_54616

def is_prime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else (List.range (n - 1)).all (fun m => m + 2 = n ∨ n % (m + 2) ≠ 0)

def count_primes (n : ℕ) : ℕ := (List.range n).filter is_prime |>.length

theorem dice_prime_probability :
  let n_dice : ℕ := 7
  let n_sides : ℕ := 12
  let n_primes : ℕ := count_primes n_sides
  let p_prime : ℚ := n_primes / n_sides
  let p_not_prime : ℚ := 1 - p_prime
  let k : ℕ := 3
  (Nat.choose n_dice k : ℚ) * p_prime ^ k * p_not_prime ^ (n_dice - k) = 1045875 / 35831808 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_prime_probability_l546_54616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l546_54636

-- Define the propositions
def p : Prop := ∀ α β : ℝ, (α = β ↔ Real.tan α = Real.tan β)
def q : Prop := ∀ (α : Type) (A : Set α), ∅ ⊆ A

-- Theorem statement
theorem problem_solution :
  (¬p) ∧ q ∧ (p ∨ q) ∧ (¬p) := by
  constructor
  · sorry -- Proof that ¬p is true
  constructor
  · sorry -- Proof that q is true
  constructor
  · sorry -- Proof that p ∨ q is true
  · sorry -- Proof that ¬p is true

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l546_54636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_square_pyramid_volume_l546_54684

/-- The volume of a regular square pyramid with all edges of length 2 -/
noncomputable def regularSquarePyramidVolume : ℝ :=
  4 * Real.sqrt 2 / 3

/-- Theorem: The volume of a regular square pyramid with all edges of length 2 is 4√2/3 -/
theorem regular_square_pyramid_volume :
  let edge_length : ℝ := 2
  regularSquarePyramidVolume = 4 * Real.sqrt 2 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_square_pyramid_volume_l546_54684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_preference_percentage_l546_54699

-- Define the set of colors
inductive Color
| Red
| Blue
| Green
| Yellow
| Purple

-- Define the frequency function
def frequency : Color → ℕ
| Color.Red => 50
| Color.Blue => 60
| Color.Green => 40
| Color.Yellow => 60
| Color.Purple => 40

-- Define the total number of responses
def total_responses : ℕ := 
  frequency Color.Red + frequency Color.Blue + frequency Color.Green + 
  frequency Color.Yellow + frequency Color.Purple

-- Define the percentage calculation
def percentage (c : Color) : ℚ :=
  (frequency c : ℚ) / (total_responses : ℚ) * 100

-- Theorem statement
theorem blue_preference_percentage :
  percentage Color.Blue = 24 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_preference_percentage_l546_54699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annualRepayment_correct_loan_fully_repaid_l546_54668

/-- Calculate the annual repayment amount for a loan -/
noncomputable def annualRepayment (a : ℝ) (r : ℝ) : ℝ :=
  (a * r * (1 + r)^5) / ((1 + r)^5 - 1)

/-- Theorem stating the correctness of the annual repayment calculation -/
theorem annualRepayment_correct (a r : ℝ) (hr : r > 0) (ha : a > 0) :
  let x := annualRepayment a r
  a * (1 + r)^5 = x * ((1 + r)^5 - 1) / r :=
by sorry

/-- The loan is fully repaid after 5 years -/
theorem loan_fully_repaid (a r : ℝ) (hr : r > 0) (ha : a > 0) :
  let x := annualRepayment a r
  a * (1 + r)^5 = x * ((1 + r)^4 + (1 + r)^3 + (1 + r)^2 + (1 + r) + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annualRepayment_correct_loan_fully_repaid_l546_54668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_calculation_l546_54644

/-- Calculates the required monthly rent given house price, annual return rate, annual taxes, and repair percentage. -/
noncomputable def required_monthly_rent (house_price : ℝ) (annual_return_rate : ℝ) (annual_taxes : ℝ) (repair_percentage : ℝ) : ℝ :=
  let annual_return := house_price * annual_return_rate
  let total_annual_earnings := annual_return + annual_taxes
  let monthly_earnings := total_annual_earnings / 12
  monthly_earnings / (1 - repair_percentage)

/-- Theorem stating that the required monthly rent for the given conditions is approximately $181.38 -/
theorem house_rent_calculation :
  let house_price : ℝ := 20000
  let annual_return_rate : ℝ := 0.06
  let annual_taxes : ℝ := 650
  let repair_percentage : ℝ := 0.15
  ∃ ε > 0, abs (required_monthly_rent house_price annual_return_rate annual_taxes repair_percentage - 181.38) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_calculation_l546_54644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_speed_l546_54630

/-- Given a man's upstream and downstream speeds, calculates his speed in still water -/
noncomputable def speed_in_still_water (upstream_speed downstream_speed : ℝ) : ℝ :=
  (upstream_speed + downstream_speed) / 2

/-- Theorem: A man rowing upstream at 32 kmph and downstream at 48 kmph has a speed of 40 kmph in still water -/
theorem man_rowing_speed :
  let upstream_speed := (32 : ℝ)
  let downstream_speed := (48 : ℝ)
  speed_in_still_water upstream_speed downstream_speed = 40 := by
  unfold speed_in_still_water
  norm_num
  
-- The following line is commented out because it's not computable
-- #eval speed_in_still_water 32 48

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_speed_l546_54630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_vertex_angle_cone_vertex_angle_arccos_l546_54657

-- Define the cone structure
structure Cone where
  height : ℝ
  base_radius : ℝ

-- Define the condition of equal areas
noncomputable def equal_cross_sections (c : Cone) : Prop :=
  let cross_section_area := (2 * c.height * Real.sqrt (3 * c.base_radius^2 - c.height^2)) / 3
  let axial_section_area := c.base_radius * c.height
  cross_section_area = axial_section_area

-- Define the angle at the vertex of the axial section
noncomputable def vertex_angle (c : Cone) : ℝ :=
  2 * Real.arctan (2 / Real.sqrt 3)

-- Theorem statement
theorem cone_vertex_angle (c : Cone) (h : equal_cross_sections c) :
  vertex_angle c = 2 * Real.arctan (2 / Real.sqrt 3) := by
  sorry

-- Alternative formulation using arccos
theorem cone_vertex_angle_arccos (c : Cone) (h : equal_cross_sections c) :
  vertex_angle c = Real.arccos (-1 / 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_vertex_angle_cone_vertex_angle_arccos_l546_54657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_ten_l546_54617

noncomputable def a : ℝ × ℝ := (3, 2)
noncomputable def b : ℝ × ℝ := (-1, 6)

noncomputable def triangle_area (v₁ v₂ : ℝ × ℝ) : ℝ :=
  (1/2) * abs (v₁.1 * v₂.2 - v₁.2 * v₂.1)

theorem triangle_area_is_ten : triangle_area a b = 10 := by
  -- Unfold definitions
  unfold triangle_area
  unfold a
  unfold b
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_ten_l546_54617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_implies_product_l546_54691

theorem power_sum_implies_product (x : ℝ) :
  (2 : ℝ)^x + (2 : ℝ)^x + (2 : ℝ)^x + (2 : ℝ)^x = 128 → (x + 1) * (x - 1) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_implies_product_l546_54691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_volume_cube_minus_cylinder_l546_54626

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_volume_cube_minus_cylinder (π : ℝ) : 
  let cube_side : ℝ := 6
  let cylinder_radius : ℝ := 3
  let cylinder_height : ℝ := 6
  let cube_volume : ℝ := cube_side ^ 3
  let cylinder_volume : ℝ := π * cylinder_radius ^ 2 * cylinder_height
  cube_volume - cylinder_volume = 216 - 54 * π := by
  -- Proof steps would go here
  sorry

#check remaining_volume_cube_minus_cylinder

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_volume_cube_minus_cylinder_l546_54626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_four_plus_two_alpha_l546_54647

theorem cos_three_pi_four_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π/8 - α) = 1/6) : 
  Real.cos (3*π/4 + 2*α) = 17/18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_four_plus_two_alpha_l546_54647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relationship_l546_54675

/-- The distance from a point to a line --/
noncomputable def distancePointToLine (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

/-- The line equation --/
def lineEquation (x y : ℝ) : Prop :=
  3 * x - 4 * y - 9 = 0

/-- The circle equation --/
def circleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

theorem line_circle_relationship :
  ∃ (x y : ℝ), lineEquation x y ∧ circleEquation x y ∧
  (distancePointToLine 0 0 3 (-4) (-9) < 2) ∧
  ¬ lineEquation 0 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relationship_l546_54675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_path_area_ratio_l546_54651

/-- Represents a particle moving along the edges of an equilateral triangle -/
structure Particle where
  start : ℝ × ℝ
  speed : ℝ

/-- The equilateral triangle ABC -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The path traced by the midpoint of the line segment joining two particles -/
noncomputable def midpointPath (p1 p2 : Particle) (triangle : EquilateralTriangle) : Set (ℝ × ℝ) :=
  sorry

/-- The area enclosed by the midpoint path -/
noncomputable def enclosedArea (path : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The area of the equilateral triangle -/
noncomputable def triangleArea (triangle : EquilateralTriangle) : ℝ :=
  sorry

/-- The centroid of the equilateral triangle -/
noncomputable def centroid (triangle : EquilateralTriangle) : ℝ × ℝ :=
  sorry

theorem midpoint_path_area_ratio 
  (triangle : EquilateralTriangle)
  (p1 : Particle)
  (p2 : Particle)
  (h1 : p1.start = triangle.A)
  (h2 : p2.start = centroid triangle)
  (h3 : p2.speed = 2 * p1.speed)
  (h4 : p1.speed > 0) :
  enclosedArea (midpointPath p1 p2 triangle) / triangleArea triangle = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_path_area_ratio_l546_54651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tickets_to_guarantee_win_l546_54643

/-- Represents a lottery ticket as a permutation of numbers from 1 to 50 -/
def Ticket := Fin 50 → Fin 50

/-- The set of all possible tickets -/
def AllTickets : Set Ticket := {t : Ticket | Function.Injective t}

/-- A ticket wins if it matches the base ticket in at least one position -/
def IsWinningTicket (base : Ticket) (t : Ticket) : Prop :=
  ∃ i : Fin 50, t i = base i

/-- A set of tickets guarantees a win if for any base ticket, 
    at least one ticket in the set is a winning ticket -/
def GuaranteesWin (tickets : Set Ticket) : Prop :=
  ∀ base : Ticket, base ∈ AllTickets → 
    ∃ t ∈ tickets, IsWinningTicket base t

theorem min_tickets_to_guarantee_win :
  ∃ (tickets : Finset Ticket), GuaranteesWin tickets.toSet ∧ tickets.card = 26 ∧
    ∀ (other_tickets : Finset Ticket), GuaranteesWin other_tickets.toSet →
      other_tickets.card ≥ 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tickets_to_guarantee_win_l546_54643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_jelly_beans_after_replacement_l546_54628

/-- Represents the distribution of jelly beans in a jar -/
structure JellyBeanJar where
  red : ℕ
  orange : ℕ
  purple : ℕ
  yellow : ℕ
  green : ℕ
  blue : ℕ

/-- Calculates the total number of jelly beans in the jar -/
def total (jar : JellyBeanJar) : ℕ :=
  jar.red + jar.orange + jar.purple + jar.yellow + jar.green + jar.blue

/-- Theorem: After replacing one-third of red jelly beans with purple, 
    the number of purple jelly beans will be 100 -/
theorem purple_jelly_beans_after_replacement (jar : JellyBeanJar) : 
  jar.red = (25 * total jar) / 100 →
  jar.orange = (20 * total jar) / 100 →
  jar.purple = (25 * total jar) / 100 →
  jar.yellow = (15 * total jar) / 100 →
  jar.green = (10 * total jar) / 100 →
  jar.blue = 15 →
  jar.purple + jar.red / 3 = 100 := by
  sorry

-- Remove the #eval line as it's not necessary and might cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_jelly_beans_after_replacement_l546_54628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equations_correct_l546_54672

/-- Represents a triangle with side lengths and angles -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Represents trilinear coordinates -/
structure TrilinearCoord where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The equation of the circumcircle in trilinear coordinates -/
def circumcircle_equation (t : Triangle) (coord : TrilinearCoord) : Prop :=
  t.a * coord.y * coord.z + t.b * coord.x * coord.z + t.c * coord.x * coord.y = 0

/-- The equation of the incircle in trilinear coordinates -/
noncomputable def incircle_equation (t : Triangle) (coord : TrilinearCoord) : Prop :=
  Real.cos (t.α / 2) * Real.sqrt coord.x + Real.cos (t.β / 2) * Real.sqrt coord.y + Real.cos (t.γ / 2) * Real.sqrt coord.z = 0

/-- The equation of the excircle (tangent to BC) in trilinear coordinates -/
noncomputable def excircle_equation (t : Triangle) (coord : TrilinearCoord) : Prop :=
  Real.cos (t.α / 2) * Real.sqrt (-coord.x) + Real.cos (t.β / 2) * Real.sqrt coord.y + Real.cos (t.γ / 2) * Real.sqrt coord.z = 0

/-- Predicate to check if a point is on the circumcircle -/
def is_on_circumcircle (t : Triangle) (coord : TrilinearCoord) : Prop :=
  sorry

/-- Predicate to check if a point is on the incircle -/
def is_on_incircle (t : Triangle) (coord : TrilinearCoord) : Prop :=
  sorry

/-- Predicate to check if a point is on the excircle -/
def is_on_excircle (t : Triangle) (coord : TrilinearCoord) : Prop :=
  sorry

/-- Theorem stating that the given equations represent the circumcircle, incircle, and excircle -/
theorem circle_equations_correct (t : Triangle) (coord : TrilinearCoord) :
  (circumcircle_equation t coord ↔ is_on_circumcircle t coord) ∧
  (incircle_equation t coord ↔ is_on_incircle t coord) ∧
  (excircle_equation t coord ↔ is_on_excircle t coord) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equations_correct_l546_54672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_when_p_true_m_set_characterization_l546_54613

/-- Proposition p -/
def prop_p (m : ℝ) : Prop :=
  ∀ x > -2, x + 49 / (x + 2) ≥ 6 * Real.sqrt 2 * m

/-- Proposition q -/
def prop_q (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - m*x + 1 = 0

/-- If proposition p is true, then m ≤ √2 -/
theorem max_m_when_p_true (m : ℝ) (h : prop_p m) : m ≤ Real.sqrt 2 := by
  sorry

/-- The set of m values where exactly one proposition is true -/
def m_set : Set ℝ := {m | (prop_p m ∧ ¬prop_q m) ∨ (¬prop_p m ∧ prop_q m)}

/-- The set of m values where exactly one proposition is true
    is equal to (-2,√2] ∪ [2,+∞) -/
theorem m_set_characterization :
  m_set = Set.Ioo (-2) (Real.sqrt 2) ∪ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_when_p_true_m_set_characterization_l546_54613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_set_l546_54600

/-- A function that checks if a natural number is of the form m^k where k ≥ 2 -/
def is_power_form (n : ℕ) : Prop :=
  ∃ (m k : ℕ), k ≥ 2 ∧ n = m^k

/-- A function that checks if all elements of a set and their sums are in power form -/
def all_sums_power_form (M : Finset ℕ) : Prop :=
  ∀ (S : Finset ℕ), S ⊆ M → is_power_form (S.sum id)

/-- The main theorem stating the existence of a set with the required properties -/
theorem exists_special_set :
  ∃ (M : Finset ℕ), M.card = 1992 ∧ all_sums_power_form M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_set_l546_54600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_theorem_l546_54625

theorem polynomial_sum_theorem (a₁ a₂ a₃ a₄ a₅ : ℤ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ 
                a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ 
                a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ 
                a₄ ≠ a₅) :
  let f := λ (x : ℤ) ↦ (x - a₁) * (x - a₂) * (x - a₃) * (x - a₄) * (x - a₅)
  f 104 = 2012 → a₁ + a₂ + a₃ + a₄ + a₅ = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_theorem_l546_54625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_proof_l546_54602

/-- The price of an apple, given the conditions of Alexander's shopping trip -/
def apple_price : ℚ := 1

theorem apple_price_proof (num_apples : ℕ) (num_oranges : ℕ) (orange_price : ℚ) (total_spent : ℚ) 
  (h1 : num_apples = 5)
  (h2 : num_oranges = 2)
  (h3 : orange_price = 2)
  (h4 : total_spent = 9)
  (h5 : num_apples * apple_price + num_oranges * orange_price = total_spent) :
  apple_price = 1 := by
  -- Proof steps would go here
  sorry

#check apple_price_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_proof_l546_54602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l546_54666

theorem expansion_properties (n : ℕ) :
  (3^n + 2^n = 275) →
  (n = 5 ∧ 
   Finset.sum (Finset.range 6) (λ r ↦ (Nat.choose 5 r) * 2^r * if 10 - 3*r = 4 then 1 else 0) = 40) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l546_54666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increasing_condition_l546_54662

/-- The inverse proportion function -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^(3 - m^2)

/-- The function is an inverse proportion -/
def is_inverse_proportion (m : ℝ) : Prop :=
  3 - m^2 = -1

/-- The graph increases with x -/
def is_increasing (m : ℝ) : Prop :=
  m + 1 < 0

theorem inverse_proportion_increasing_condition (m : ℝ) :
  is_inverse_proportion m ∧ is_increasing m ↔ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increasing_condition_l546_54662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_has_triangle_face_l546_54645

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  is_convex : Prop

/-- A face of a polyhedron. -/
structure Face where
  is_triangle : Prop

/-- A vertex of a polyhedron. -/
structure Vertex where

/-- The number of edges meeting at a vertex. -/
def edges_at_vertex (v : Vertex) : ℕ := sorry

/-- The faces of a polyhedron. -/
def faces (p : ConvexPolyhedron) : Set Face := sorry

/-- The vertices of a polyhedron. -/
def vertices (p : ConvexPolyhedron) : Set Vertex := sorry

/-- Main theorem: If at least four edges meet at each vertex of a convex polyhedron,
    then at least one of its faces is a triangle. -/
theorem convex_polyhedron_has_triangle_face (p : ConvexPolyhedron) :
  (∀ v ∈ vertices p, edges_at_vertex v ≥ 4) →
  ∃ f ∈ faces p, f.is_triangle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_has_triangle_face_l546_54645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_definiteness_l546_54606

/-- Represents the properties of an algorithm --/
inductive AlgorithmProperty
  | Orderliness
  | Definiteness
  | Feasibility
  | Uncertainty

/-- Defines the characteristics of an algorithm's rules and steps --/
structure AlgorithmCharacteristics where
  uniquelyDetermined : Bool
  notAmbiguous : Bool
  noMultiplePossibilities : Bool

/-- Theorem stating that an algorithm with uniquely determined rules and steps,
    without ambiguity or multiple possibilities, has the property of definiteness --/
theorem algorithm_definiteness 
  (characteristics : AlgorithmCharacteristics)
  (h1 : characteristics.uniquelyDetermined = true)
  (h2 : characteristics.notAmbiguous = true)
  (h3 : characteristics.noMultiplePossibilities = true) :
  AlgorithmProperty.Definiteness = 
    (fun (c : AlgorithmCharacteristics) ↦ 
      if c.uniquelyDetermined ∧ c.notAmbiguous ∧ c.noMultiplePossibilities 
      then AlgorithmProperty.Definiteness 
      else AlgorithmProperty.Uncertainty) characteristics := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_definiteness_l546_54606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l546_54603

/-- The equation of a line parameterized by (x, y) = (3t + 2, 5t - 7) is y = (5/3)x - 31/3 -/
theorem line_equation (t : ℝ) : 
  (5 * t - 7) = (5 / 3) * (3 * t + 2) - 31 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l546_54603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_difference_96_l546_54646

theorem existence_of_difference_96 (S : Finset ℕ) (h1 : S ⊆ Finset.range 3840) (h2 : S.card = 1996) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ (a - b = 96 ∨ b - a = 96) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_difference_96_l546_54646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_theorem_l546_54634

/-- Represents the outcome of a single dice toss -/
inductive DiceOutcome
  | one | two | three | four | five | six
  deriving Fintype, DecidableEq

/-- Represents the outcome of three dice tosses -/
structure ThreeTosses where
  first : DiceOutcome
  second : DiceOutcome
  third : DiceOutcome
deriving Fintype, DecidableEq

def is_valid_toss (t : ThreeTosses) : Bool :=
  match t.first, t.second, t.third with
  | DiceOutcome.one,   DiceOutcome.one,   DiceOutcome.two   => true
  | DiceOutcome.one,   DiceOutcome.two,   DiceOutcome.three => true
  | DiceOutcome.two,   DiceOutcome.one,   DiceOutcome.three => true
  | DiceOutcome.one,   DiceOutcome.three, DiceOutcome.four  => true
  | DiceOutcome.two,   DiceOutcome.two,   DiceOutcome.four  => true
  | DiceOutcome.three, DiceOutcome.one,   DiceOutcome.four  => true
  | DiceOutcome.one,   DiceOutcome.four,  DiceOutcome.five  => true
  | DiceOutcome.two,   DiceOutcome.three, DiceOutcome.five  => true
  | DiceOutcome.three, DiceOutcome.two,   DiceOutcome.five  => true
  | DiceOutcome.four,  DiceOutcome.one,   DiceOutcome.five  => true
  | DiceOutcome.one,   DiceOutcome.five,  DiceOutcome.six   => true
  | DiceOutcome.two,   DiceOutcome.four,  DiceOutcome.six   => true
  | DiceOutcome.three, DiceOutcome.three, DiceOutcome.six   => true
  | DiceOutcome.four,  DiceOutcome.two,   DiceOutcome.six   => true
  | DiceOutcome.five,  DiceOutcome.one,   DiceOutcome.six   => true
  | _,                 _,                 _                 => false

def has_two (t : ThreeTosses) : Bool :=
  t.first = DiceOutcome.two || t.second = DiceOutcome.two || t.third = DiceOutcome.two

theorem dice_probability_theorem :
  (Finset.filter (λ t : ThreeTosses => is_valid_toss t ∧ has_two t) Finset.univ).card /
  (Finset.filter (λ t : ThreeTosses => is_valid_toss t) Finset.univ).card = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_theorem_l546_54634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_l546_54604

theorem closest_integer_to_cube_root : 
  ∃ (m : ℤ), |m - (11^3 + 2^3 : ℝ)^(1/3)| ≤ |m' - (11^3 + 2^3 : ℝ)^(1/3)| ∧ 
  m = 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_l546_54604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_function_property_l546_54641

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x

theorem tangent_line_and_function_property (a t : ℝ) (h1 : t > 0) :
  (∃ (b : ℝ), ∀ x, 3 * x + 1 = (deriv (f a)) t * (x - t) + f a t) →
  (a = 2 ∧
   ∀ (k x : ℝ), x > 1 → k ≤ 2 →
     (f a x > k * (1 - 3 / x) + 2 * x - 1 → -1/2 ≤ k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_function_property_l546_54641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_cost_l546_54686

def total_spent : ℝ := 12.30
def marbles_cost : ℝ := 6.59

theorem football_cost : 
  total_spent - marbles_cost = 5.71 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_cost_l546_54686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_reach_height_l546_54652

/-- Represents a Ferris wheel -/
structure FerrisWheel where
  radius : ℝ
  revolutionTime : ℝ
  initialDelay : ℝ

/-- Calculates the height of a rider on the Ferris wheel at a given time -/
noncomputable def riderHeight (wheel : FerrisWheel) (t : ℝ) : ℝ :=
  wheel.radius * (Real.cos ((2 * Real.pi / wheel.revolutionTime) * t) + 1)

/-- Theorem: Time to reach 15 feet above lowest point is 20 seconds -/
theorem time_to_reach_height (wheel : FerrisWheel) 
  (h1 : wheel.radius = 30)
  (h2 : wheel.revolutionTime = 90)
  (h3 : wheel.initialDelay = 5) :
  ∃ t : ℝ, t = 20 ∧ riderHeight wheel (t - wheel.initialDelay) = 45 := by
  sorry

#check time_to_reach_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_reach_height_l546_54652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l546_54640

noncomputable def a (x : ℝ) : Fin 2 → ℝ
  | 0 => Real.cos x
  | 1 => 1/2

noncomputable def b (x : ℝ) : Fin 2 → ℝ
  | 0 => Real.sqrt 3 * Real.sin x
  | 1 => Real.cos (2*x)

noncomputable def f (x : ℝ) : ℝ := 
  (a x 0 * b x 0) + (a x 1 * b x 1)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ 
   ∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  (∃ x₀ ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f x ≤ f x₀ ∧ 
   x₀ = π/3 ∧ f x₀ = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l546_54640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_is_two_ninths_l546_54605

/-- Represents a spinner with a list of numbers -/
def Spinner := List Nat

/-- The three spinners -/
def S : Spinner := [1, 2, 4]
def T : Spinner := [1, 3, 5]
def U : Spinner := [2, 3, 6]

/-- Predicate to check if a number is even -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- Calculate the probability of an event given the number of favorable outcomes and total outcomes -/
def probability (favorableOutcomes totalOutcomes : Nat) : ℚ :=
  ↑favorableOutcomes / ↑totalOutcomes

/-- Calculate the number of possible outcomes when spinning three spinners -/
def totalOutcomes (s t u : Spinner) : Nat :=
  s.length * t.length * u.length

/-- Count the number of even sums when spinning three spinners -/
def countEvenSums (s t u : Spinner) : Nat :=
  (List.filter (fun (x : Nat × Nat × Nat) => isEven (x.1 + x.2.1 + x.2.2)) 
    (List.product s (List.product t u))).length

/-- The main theorem: probability of even sum is 2/9 -/
theorem prob_even_sum_is_two_ninths :
  probability (countEvenSums S T U) (totalOutcomes S T U) = 2 / 9 := by
  sorry

#eval probability (countEvenSums S T U) (totalOutcomes S T U)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_is_two_ninths_l546_54605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_B_is_150_boxes_A_eq_boxes_B_l546_54623

/-- The unit price per box of type B masks -/
noncomputable def price_B : ℝ := 150

/-- The unit price per box of type A masks -/
noncomputable def price_A : ℝ := price_B + 50

/-- The number of boxes of type A masks that can be purchased with 2000 yuan -/
noncomputable def boxes_A : ℝ := 2000 / price_A

/-- The number of boxes of type B masks that can be purchased with 1500 yuan -/
noncomputable def boxes_B : ℝ := 1500 / price_B

/-- Theorem stating that the unit price per box of type B masks is 150 yuan -/
theorem price_B_is_150 : price_B = 150 :=
  by rfl

/-- Theorem stating that the number of boxes of type A that can be purchased with 2000 yuan
    is equal to the number of boxes of type B that can be purchased with 1500 yuan -/
theorem boxes_A_eq_boxes_B : boxes_A = boxes_B :=
  by
    simp [boxes_A, boxes_B, price_A, price_B]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_B_is_150_boxes_A_eq_boxes_B_l546_54623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l546_54627

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

def intersecting_line (x y m : ℝ) : Prop := x - y + m = 0

def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem hyperbola_intersection_theorem (m : ℝ) 
  (x₁ y₁ x₂ y₂ : ℝ) :
  hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧
  intersecting_line x₁ y₁ m ∧ intersecting_line x₂ y₂ m ∧
  my_circle ((x₁ + x₂)/2) ((y₁ + y₂)/2) →
  (m = 1 ∨ m = -1) ∧ 
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l546_54627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_descending_order_proof_l546_54615

theorem descending_order_proof : 
  let a : ℝ := (3/5 : ℝ)^(-(1/3 : ℝ))
  let b : ℝ := (4/3 : ℝ)^(-(1/2 : ℝ))
  let c : ℝ := Real.log (3/5)
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_descending_order_proof_l546_54615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l546_54635

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4.5) / (x^2 - 1)

theorem inequality_solution_set :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 →
    (f x ≤ 0 ↔ x ∈ Set.Ioo (-3 * Real.sqrt 2 / 2) (-1) ∪ 
                Set.Ioo (-1) 1 ∪ 
                Set.Ioo 1 (3 * Real.sqrt 2 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l546_54635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_l546_54664

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - Real.log x / Real.log 2

theorem solution_set_of_f (a : ℝ) :
  f a 1 = 1 →
  {x : ℝ | f a x > 1} = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_l546_54664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_divides_difference_l546_54688

theorem exists_prime_divides_difference : ∃ p : Nat, 
  Nat.Prime p ∧ p > 2019 ∧ 
  ∀ α : Nat, α > 0 → 
    ∃ n : Nat, n > 0 ∧ 
      (p^α ∣ (2019^n - n^2017)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_divides_difference_l546_54688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_in_trapezoid_l546_54621

/-- Represents a trapezoid with given side lengths and bases -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  base1 : ℝ
  base2 : ℝ

/-- Represents a triangle with given side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- The triangle formed by drawing a line from the vertex of the smaller base
    parallel to the larger side in the given trapezoid -/
noncomputable def triangleInTrapezoid (t : Trapezoid) : Triangle :=
  { side1 := min t.side1 t.side2
  , side2 := t.base2 - t.base1
  , side3 := max t.side1 t.side2 }

theorem triangle_sides_in_trapezoid :
  let t : Trapezoid := { side1 := 7, side2 := 11, base1 := 5, base2 := 15 }
  let triangle := triangleInTrapezoid t
  triangle.side1 = 7 ∧ triangle.side2 = 10 ∧ triangle.side3 = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_in_trapezoid_l546_54621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silenos_drinking_time_l546_54682

noncomputable section

/-- Time for Silenos to finish the barrel alone -/
def silenos_time : ℝ := 6

/-- Time for Bacchus to finish the barrel alone -/
def bacchus_time : ℝ := 9

/-- Time for Silenos and Bacchus to finish the barrel together -/
def combined_time : ℝ := 4

/-- Bacchus's drinking rate relative to Silenos when drinking together -/
def bacchus_relative_rate : ℝ := 1/2

theorem silenos_drinking_time :
  (silenos_time = 6) ∧
  (combined_time = silenos_time - 2) ∧
  (1 / combined_time = 1 / silenos_time + bacchus_relative_rate / silenos_time) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_silenos_drinking_time_l546_54682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_eq_neg_one_l546_54618

/-- A function of the form a*sin(πx + α) + b*cos(πx + β) -/
noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

/-- Theorem stating that if f(2015) = -1, then f(2016) = -1 -/
theorem f_2016_eq_neg_one (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0)
    (h : f a b α β 2015 = -1) : f a b α β 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_eq_neg_one_l546_54618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_example_l546_54629

/-- Represents a cone with given slant height and lateral area -/
structure Cone where
  slant_height : ℝ
  lateral_area : ℝ

/-- Calculates the height of a cone given its slant height and lateral area -/
noncomputable def cone_height (c : Cone) : ℝ :=
  Real.sqrt (c.slant_height ^ 2 - (c.lateral_area / (Real.pi * c.slant_height)) ^ 2)

/-- Theorem: A cone with slant height 13 and lateral area 65π has height 12 -/
theorem cone_height_example : 
  let c : Cone := { slant_height := 13, lateral_area := 65 * Real.pi }
  cone_height c = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_example_l546_54629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_vector_equation_l546_54658

/-- Given a triangle ABC and a point M, if M satisfies the centroid condition
and there exists a real number m satisfying the vector equation,
then m must equal 3. -/
theorem triangle_centroid_vector_equation (A B C M : ℝ × ℝ) (m : ℝ) :
  (M - A) + (M - B) + (M - C) = (0, 0) →
  (B - A) + (C - A) = m • (M - A) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_vector_equation_l546_54658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_div_g_l546_54659

open Set Function Real

-- Define odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the theorem
theorem solution_set_f_div_g (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g)
  (hg_nonzero : ∀ x, g x ≠ 0)
  (h_deriv : ∀ x < 0, (deriv f x) * (g x) < (f x) * (deriv g x))
  (hf_zero : f (-3) = 0) :
  {x : ℝ | f x / g x < 0} = Ioo (-3) 0 ∪ Ioi 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_div_g_l546_54659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt_2_l546_54611

/-- The radius of the circle formed by points with spherical coordinates (2, θ, π/4) -/
noncomputable def circle_radius (θ : Real) : Real :=
  Real.sqrt ((2 * Real.sin (Real.pi / 4) * Real.cos θ) ^ 2 + (2 * Real.sin (Real.pi / 4) * Real.sin θ) ^ 2)

/-- Theorem: The radius of the circle is √2 -/
theorem circle_radius_is_sqrt_2 (θ : Real) : circle_radius θ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt_2_l546_54611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l546_54665

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + 1 / (3 - x)

theorem domain_of_f : 
  {x : ℝ | x ≥ -1 ∧ x ≠ 3} = {x : ℝ | ∃ y, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l546_54665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_equals_20_l546_54660

-- Define the vertices of the original triangle
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (5, -2)
def C : ℝ × ℝ := (7, 2)

-- Define the reflection line
def reflection_line : ℝ := 5

-- Function to reflect a point about x = reflection_line
noncomputable def reflect (p : ℝ × ℝ) : ℝ × ℝ :=
  (2 * reflection_line - p.1, p.2)

-- Define the vertices of the reflected triangle
noncomputable def A' : ℝ × ℝ := reflect A
noncomputable def B' : ℝ × ℝ := reflect B
noncomputable def C' : ℝ × ℝ := reflect C

-- Function to calculate the area of a triangle given its vertices
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

-- Theorem statement
theorem area_of_union_equals_20 :
  triangle_area A B C + triangle_area A' B' C' = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_equals_20_l546_54660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_values_count_l546_54654

theorem xy_values_count (x y : ℕ+) 
  (h : (1 : ℝ) / Real.sqrt (x : ℝ) + (1 : ℝ) / Real.sqrt (y : ℝ) = (1 : ℝ) / Real.sqrt 20) : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, ∃ (a b : ℕ+), (a : ℕ) * (b : ℕ) = n ∧ 
    ((1 : ℝ) / Real.sqrt (a : ℝ) + (1 : ℝ) / Real.sqrt (b : ℝ) = (1 : ℝ) / Real.sqrt 20)) ∧ 
  S.card = 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_values_count_l546_54654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_symmetric_point_l546_54696

/-- The eccentricity of a hyperbola with a symmetric point condition -/
theorem hyperbola_eccentricity_with_symmetric_point 
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  let hyperbola := λ (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let symmetry_line := λ (x y : ℝ) ↦ y = (b / a) * x
  let right_focus : ℝ × ℝ := (c, 0)
  ∃ (P : ℝ × ℝ), 
    hyperbola P.1 P.2 ∧ 
    P.1 < 0 ∧
    ∃ (S : ℝ × ℝ), 
      symmetry_line S.1 S.2 ∧
      S = ((P.1 + right_focus.1) / 2, (P.2 + right_focus.2) / 2) →
  c / a = Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_symmetric_point_l546_54696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l546_54683

-- Define the function f
noncomputable def f (θ : ℝ) : ℝ :=
  let r : ℝ := 1  -- We can assume r = 1 without loss of generality
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x - y) / r

-- State the theorem
theorem f_properties :
  (∀ θ, f θ ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2)) ∧
  (∀ θ, f (3 * Real.pi / 2 - θ) = f θ) ∧
  (∀ θ, f (θ + 2 * Real.pi) = f θ) ∧
  (∀ p, p > 0 → (∀ θ, f (θ + p) = f θ) → p ≥ 2 * Real.pi) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l546_54683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integral_value_l546_54632

/-- The function to be integrated -/
noncomputable def f (a x : ℝ) : ℝ := |1 / (1 + Real.exp x) - 1 / (1 + Real.exp a)|

/-- The integral of f from 0 to 2 -/
noncomputable def integral (a : ℝ) : ℝ := ∫ x in Set.Icc 0 2, f a x

/-- The theorem stating the minimum value of the integral -/
theorem min_integral_value :
  ∃ (a : ℝ), 0 ≤ a ∧ a ≤ 2 ∧
  ∀ (b : ℝ), 0 ≤ b ∧ b ≤ 2 → integral a ≤ integral b ∧
  integral a = Real.log ((2 + 2 * Real.exp 2) / (1 + 2 * Real.exp 1 + Real.exp 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integral_value_l546_54632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_avg_distance_l546_54656

noncomputable section

/-- The side length of the square -/
def side_length : ℝ := 8

/-- The distance the lemming moves along the diagonal -/
def diagonal_move : ℝ := 4.8

/-- The distance the lemming moves after turning -/
def turn_move : ℝ := 2.5

/-- The final x-coordinate of the lemming -/
noncomputable def final_x : ℝ := (diagonal_move / (side_length * Real.sqrt 2)) * side_length

/-- The final y-coordinate of the lemming -/
noncomputable def final_y : ℝ := (diagonal_move / (side_length * Real.sqrt 2)) * side_length + turn_move

/-- The average distance from the lemming to the sides of the square -/
noncomputable def avg_distance : ℝ := (final_x + (side_length - final_x) + final_y + (side_length - final_y)) / 4

theorem lemming_avg_distance : avg_distance = 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_avg_distance_l546_54656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ellipse_perimeter_l546_54667

/-- Given a rectangle EFGH with area 2530 and an ellipse with area 2530π
    passing through E and G with foci at F and H, 
    the perimeter of the rectangle is 8√1265. -/
theorem rectangle_ellipse_perimeter : 
  ∀ (E F G H : ℝ × ℝ) (a b : ℝ),
    let rectangle_area := 2530
    let ellipse_area := 2530 * Real.pi
    let rectangle_perimeter := 2 * (dist E F + dist F G)
    let ellipse_major_axis := 2 * a
    let ellipse_minor_axis := 2 * b
    (dist E F) * (dist F G) = rectangle_area →
    ellipse_area = Real.pi * a * b →
    dist E G = ellipse_major_axis →
    dist F H = 2 * Real.sqrt (a^2 - b^2) →
    rectangle_perimeter = 8 * Real.sqrt 1265 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ellipse_perimeter_l546_54667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_problem_l546_54612

/-- Predicate to check if a line from P to A is tangent to a circle with center C and radius r -/
def is_tangent (P A C : ℝ × ℝ) (r : ℝ) : Prop :=
  let d := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let h := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  d^2 = r^2 + h^2

/-- Function to calculate the minimum area of the quadrilateral PACB -/
noncomputable def min_area_quadrilateral (P A C B : ℝ × ℝ) : ℝ :=
  sorry  -- The actual implementation would involve complex geometric calculations

/-- The value of k for which the minimum area of the quadrilateral is 2 -/
theorem tangent_line_circle_problem (k : ℝ) : 
  k > 0 →
  (∃ (P : ℝ × ℝ), 
    (k * P.1 + P.2 + 4 = 0) ∧ 
    (∃ (A B : ℝ × ℝ),
      (A.1^2 + A.2^2 - 2*A.2 = 0) ∧
      (B.1^2 + B.2^2 - 2*B.2 = 0) ∧
      (is_tangent P A (0, 1) 1) ∧
      (is_tangent P B (0, 1) 1) ∧
      (min_area_quadrilateral P A (0, 1) B = 2))) →
  k = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_problem_l546_54612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johnny_hourly_wage_l546_54671

/-- Johnny's work scenario -/
structure WorkScenario where
  hours_worked : ℚ
  total_earnings : ℚ

/-- Calculate hourly wage given a work scenario -/
def hourly_wage (scenario : WorkScenario) : ℚ :=
  scenario.total_earnings / scenario.hours_worked

/-- Johnny's specific work scenario -/
def johnny_scenario : WorkScenario :=
  { hours_worked := 6
    total_earnings := 285/10 }

/-- Theorem: Johnny's hourly wage is $4.75 -/
theorem johnny_hourly_wage :
  hourly_wage johnny_scenario = 475/100 := by
  unfold hourly_wage johnny_scenario
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johnny_hourly_wage_l546_54671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_in_cube_five_l546_54648

theorem length_in_cube_five (cube_edges : List ℝ) (P Q : Fin 3 → ℝ) : 
  cube_edges = [2, 3, 5, 7] →
  P = ![0, 0, 0] →
  Q = ![7, 7, 17] →
  let entry_point := ![0, 0, 5]
  let exit_point := ![5, 5, 10]
  let length_in_cube := Real.sqrt ((exit_point 0 - entry_point 0)^2 + 
                                   (exit_point 1 - entry_point 1)^2 + 
                                   (exit_point 2 - entry_point 2)^2)
  length_in_cube = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_in_cube_five_l546_54648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_box_height_is_70cm_l546_54639

/-- Represents the problem of finding the minimum box height for Alice to reach the light bulb -/
def min_box_height_for_light_bulb (ceiling_height : ℝ) (light_bulb_below_ceiling : ℝ) 
  (alice_height : ℝ) (alice_reach_above_head : ℝ) : ℝ :=
  let floor_to_light_bulb := ceiling_height - light_bulb_below_ceiling
  let alice_total_reach := alice_height + alice_reach_above_head
  floor_to_light_bulb - alice_total_reach

/-- The minimum box height for Alice to reach the light bulb is 70 cm -/
theorem min_box_height_is_70cm : 
  min_box_height_for_light_bulb 300 20 160 50 = 70 := by
  unfold min_box_height_for_light_bulb
  simp
  -- The proof is straightforward arithmetic, but we'll use sorry for now
  sorry

-- We can use #eval to check the result
#eval min_box_height_for_light_bulb 300 20 160 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_box_height_is_70cm_l546_54639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_cos_squared_45_to_135_equals_273_div_4_main_proof_l546_54697

open Real BigOperators

/-- The sum of cos^2 from 0° to 90° -/
noncomputable def sum_cos_squared_0_to_90 : ℝ := 91 / 2

/-- The sum of cos^2 from 45° to 135° -/
noncomputable def sum_cos_squared_45_to_135 : ℝ := ∑ k in Finset.range 91, (cos (((k + 45) * π) / 180))^2

/-- Theorem stating that the sum of cos^2 from 45° to 135° equals 273/4 -/
theorem sum_cos_squared_45_to_135_equals_273_div_4 :
  sum_cos_squared_45_to_135 = 273 / 4 :=
by sorry

/-- Main theorem proving the original problem -/
theorem main_proof :
  sum_cos_squared_45_to_135 = 68.25 :=
by
  rw [sum_cos_squared_45_to_135_equals_273_div_4]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_cos_squared_45_to_135_equals_273_div_4_main_proof_l546_54697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_MPN_l546_54653

noncomputable section

open Real

-- Define the curves C₁ and C₂ in polar coordinates
def C₁ (θ : ℝ) : ℝ := 4 * cos θ
def C₂ (θ : ℝ) : ℝ := 2 * sin θ

-- Define the points P, M, and N
def P (α : ℝ) : ℝ × ℝ := (C₁ α * cos α, C₁ α * sin α)
def M (α : ℝ) : ℝ × ℝ := (C₁ (α + Real.pi/2) * cos (α + Real.pi/2), C₁ (α + Real.pi/2) * sin (α + Real.pi/2))
def N (α : ℝ) : ℝ × ℝ := (C₂ (α + Real.pi/2) * cos (α + Real.pi/2), C₂ (α + Real.pi/2) * sin (α + Real.pi/2))

-- Define the area of triangle MPN
def areaMPN (α : ℝ) : ℝ :=
  let op := sqrt ((P α).1^2 + (P α).2^2)
  let nm := sqrt ((M α).1 - (N α).1)^2 + ((M α).2 - (N α).2)^2
  1/2 * op * nm

-- State the theorem
theorem max_area_MPN :
  ∀ α, 0 < α → α < Real.pi/2 →
  areaMPN α ≤ 2 * sqrt 5 + 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_MPN_l546_54653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_circle_l546_54687

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 6 = (y - 1)^2

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 4

-- Theorem statement
theorem intersection_points_on_circle :
  ∀ x y : ℝ, parabola1 x y ∧ parabola2 x y → circle_equation x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_circle_l546_54687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_participants_l546_54655

-- Define indoor_participants and outdoor_participants as functions from ℕ to Prop
def indoor_participants : ℕ → Prop := λ x => True
def outdoor_participants : ℕ → Prop := λ y => True

-- Define axioms
axiom initial_difference : ∀ x y, indoor_participants x → outdoor_participants y → y = x + 480

axiom after_move : ∀ x y, indoor_participants x → outdoor_participants y → 
  y + 50 = 5 * (x - 50)

-- State the theorem
theorem total_participants : ∃ x y, indoor_participants x ∧ outdoor_participants y ∧ x + y = 870 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_participants_l546_54655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_longest_chord_l546_54689

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point on a circle --/
def PointOnCircle (c : Circle) : Type := { p : ℝ × ℝ // (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 }

/-- A chord of a circle --/
def Chord (c : Circle) : Type := { pair : PointOnCircle c × PointOnCircle c // pair.1 ≠ pair.2 }

/-- The length of a chord --/
noncomputable def chordLength (c : Circle) (chord : Chord c) : ℝ :=
  Real.sqrt ((chord.val.1.val.1 - chord.val.2.val.1)^2 + (chord.val.1.val.2 - chord.val.2.val.2)^2)

/-- A chord is a diameter if it passes through the center of the circle --/
def isDiameter (c : Circle) (chord : Chord c) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    c.center = (chord.val.1.val.1 + t * (chord.val.2.val.1 - chord.val.1.val.1),
                chord.val.1.val.2 + t * (chord.val.2.val.2 - chord.val.1.val.2))

theorem unique_longest_chord (c : Circle) (p : PointOnCircle c) :
  ∃! (chord : Chord c), (chord.val.1 = p ∨ chord.val.2 = p) ∧
    ∀ (other : Chord c), (other.val.1 = p ∨ other.val.2 = p) →
      chordLength c chord ≥ chordLength c other :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_longest_chord_l546_54689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_between_parallel_planes_l546_54614

-- Define a structure for a line segment in 3D space
structure LineSegment3D where
  start : ℝ × ℝ × ℝ
  endPoint : ℝ × ℝ × ℝ

-- Define a structure for a plane in 3D space
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Define what it means for two planes to be parallel
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Define what it means for a line segment to be enclosed between two planes
def enclosed_between (seg : LineSegment3D) (p1 p2 : Plane) : Prop := sorry

-- Define what it means for two line segments to be parallel
def parallel_segments (seg1 seg2 : LineSegment3D) : Prop := sorry

-- Define equality for line segments
def segment_equal (seg1 seg2 : LineSegment3D) : Prop := sorry

-- Theorem statement
theorem parallel_segments_between_parallel_planes 
  (p1 p2 : Plane) (seg1 seg2 : LineSegment3D) :
  parallel_planes p1 p2 →
  enclosed_between seg1 p1 p2 →
  enclosed_between seg2 p1 p2 →
  parallel_segments seg1 seg2 →
  segment_equal seg1 seg2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_between_parallel_planes_l546_54614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisors_l546_54661

/-- Sum of positive divisors of a natural number -/
def sigma (n : ℕ) : ℕ := sorry

/-- Sequence of natural numbers satisfying the divisibility property -/
def a : ℕ → ℕ := sorry

theorem infinitely_many_divisors :
  ∀ n : ℕ, (a n) ∣ (2^(sigma (a n)) - 1) ∧
           a n < a (n + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisors_l546_54661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l546_54676

theorem polynomial_factorization (k : ℤ) :
  let N : ℤ := 4 * k^4 - 8 * k^2 + 2
  ∀ (R : Type) [CommRing R] [Algebra ℤ R] (x : R),
    x^8 + N * x^4 + 1 = (x^4 - 2*k*x^3 + 2*k^2*x^2 - 2*k*x + 1) * (x^4 + 2*k*x^3 + 2*k^2*x^2 + 2*k*x + 1) := by
  intro R _ _ x
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l546_54676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_x_intercept_l546_54692

/-- A line passing through (0, 2) and parallel to 2x - y = 0 has x-intercept -1 -/
theorem parallel_line_x_intercept :
  ∀ (l : Set (ℝ × ℝ)),
    (∃ (P : ℝ × ℝ), P.1 = 0 ∧ P.2 = 2 ∧ P ∈ l) →
    (∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ c, 2 * x - y + c = 0) →
    (∃ (x : ℝ), (x, 0) ∈ l ∧ x = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_x_intercept_l546_54692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l546_54631

/-- The function f(x) = 2sin(x) - 2cos(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x - 2 * Real.cos x

/-- The period of f is 2π -/
theorem f_period : ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), 0 < q ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l546_54631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sin_to_cos_l546_54607

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)
noncomputable def g (x : ℝ) := Real.cos (2 * x)

theorem shift_sin_to_cos (x : ℝ) : 
  f (x + Real.pi / 12) = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sin_to_cos_l546_54607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1008th_term_l546_54693

/-- The nth term of the sequence -/
noncomputable def sequenceTerm (a : ℝ) (n : ℕ) : ℝ := a^(2*n) / (2*n - 1)

/-- The 1008th term of the sequence -/
noncomputable def term1008 (a : ℝ) : ℝ := a^2016 / 2015

theorem sequence_1008th_term (a : ℝ) : sequenceTerm a 1008 = term1008 a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1008th_term_l546_54693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_72_l546_54663

/-- The total profit function for a bicycle company investing in two cities. -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt (2 * x) - (1/4) * x + 26

/-- The theorem stating that the total profit function reaches its maximum when x = 72. -/
theorem max_profit_at_72 :
  ∀ x ∈ Set.Icc 40 80, f 72 ≥ f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_72_l546_54663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_line_l546_54610

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ a = Real.sqrt 3 * b ∧ ellipse a b (Real.sqrt 2) ((2 * Real.sqrt 3) / 3)

-- Define the line l
def line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 2)

-- Define the area of triangle OAB
noncomputable def triangle_area (k : ℝ) : ℝ :=
  2 * Real.sqrt 6 * Real.sqrt ((k^4 + k^2) / (1 + 6 * k^2 + 9 * k^4))

-- Theorem statement
theorem ellipse_and_max_area_line 
  (a b : ℝ) (h : conditions a b) :
  (∀ x y, ellipse a b x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
  (∀ k, triangle_area k ≤ triangle_area 1 ∧ triangle_area k ≤ triangle_area (-1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_line_l546_54610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_dot_product_l546_54679

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = x -/
def isOnParabola (p : Point) : Prop :=
  p.y^2 = p.x

/-- Represents the focus of the parabola y² = x -/
noncomputable def focus : Point :=
  { x := 1/4, y := 0 }

/-- Checks if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Calculates the dot product of two vectors from the origin -/
def dotProduct (p1 p2 : Point) : ℝ :=
  p1.x * p2.x + p1.y * p2.y

/-- Main theorem -/
theorem parabola_line_intersection_dot_product
  (A B : Point)
  (h1 : isOnParabola A)
  (h2 : isOnParabola B)
  (h3 : areCollinear A B focus)
  (h4 : A ≠ B) :
  dotProduct A B = -3/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_dot_product_l546_54679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_equals_102_l546_54678

/-- Represents a trapezoid with parallel sides AB and CD -/
structure Trapezoid where
  ab : ℝ  -- Length of side AB
  cd : ℝ  -- Length of side CD
  h : ℝ   -- Altitude of the trapezoid

/-- Calculates the area of a quadrilateral EFCD within the trapezoid -/
noncomputable def area_EFCD (t : Trapezoid) : ℝ :=
  let ef := (t.ab + t.cd) / 2  -- Length of EF (midline of trapezoid)
  let h_ef := t.h / 2          -- Altitude of EFCD
  h_ef * (ef + t.cd) / 2       -- Area formula for trapezoid EFCD

/-- Theorem statement -/
theorem area_EFCD_equals_102 (t : Trapezoid) 
    (h_ab : t.ab = 8) 
    (h_cd : t.cd = 20) 
    (h_h : t.h = 12) : 
  area_EFCD t = 102 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_equals_102_l546_54678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l546_54650

theorem interest_rate_calculation (total_amount : ℝ) (second_rate : ℝ) (total_interest_rate : ℝ) (second_amount : ℝ) :
  total_amount = 100000 →
  second_rate = 11 →
  total_interest_rate = 9.25 →
  second_amount = 12500 →
  let first_amount := total_amount - second_amount
  let total_interest := total_interest_rate / 100 * total_amount
  let second_interest := second_rate / 100 * second_amount
  let first_interest := total_interest - second_interest
  first_interest / first_amount * 100 = 9 := by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l546_54650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_program_output_l546_54637

def i_sequence (n : ℕ) : ℕ := 2 * n + 1

def s (n : ℕ) : ℕ := 2 * (i_sequence n) + 3

theorem program_output :
  (∃ (k : ℕ), i_sequence k < 8 ∧ i_sequence (k + 1) ≥ 8) →
  (∀ (n : ℕ), i_sequence n < 8 → s n ≤ 17) ∧
  (∃ (m : ℕ), i_sequence m < 8 ∧ s m = 17) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_program_output_l546_54637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_term_between_squares_l546_54620

theorem new_term_between_squares (n : ℕ) : 
  let y := (n : ℝ)^2
  let z := ((n : ℝ) + 1)^2
  (y + z) / 2 + Real.sqrt y = (n : ℝ)^2 + 2*(n : ℝ) + 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_term_between_squares_l546_54620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l546_54601

-- Define the constants
noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log (1/3) / Real.log 2
noncomputable def c : ℝ := Real.sqrt 2

-- State the theorem
theorem abc_inequality : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l546_54601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l546_54690

-- Define the hyperbola
noncomputable def is_hyperbola (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the asymptote condition
noncomputable def asymptote_condition (a b : ℝ) : Prop :=
  3 / a = Real.sqrt 3 / b

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) :
  is_hyperbola a b 3 (Real.sqrt 3) →
  asymptote_condition a b →
  eccentricity a b = 2 * Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l546_54690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_inequality_condition_l546_54670

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (sin x) / (cos x)^3

-- Part 1
theorem f_monotonicity :
  ∀ x ∈ Set.Ioo 0 (π/2),
    (x < π/4 → (deriv (f 8)) x > 0) ∧
    (x > π/4 → (deriv (f 8)) x < 0) := by sorry

-- Part 2
theorem f_inequality_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (π/2), f a x < sin (2*x)) ↔ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_inequality_condition_l546_54670
