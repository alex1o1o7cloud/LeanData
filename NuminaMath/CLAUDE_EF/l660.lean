import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_floor_l660_66079

open Real

theorem smallest_positive_root_floor :
  ∃ r : ℝ, r > 0 ∧
    (sin r + 2 * cos r + 2 * tan r = 0) ∧
    (∀ x, 0 < x ∧ x < r → sin x + 2 * cos x + 2 * tan x ≠ 0) ∧
    3 ≤ r ∧ r < 4 :=
by
  -- The proof goes here
  sorry

#check smallest_positive_root_floor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_floor_l660_66079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_seat_number_l660_66045

def systematic_sample (total : ℕ) (sample_size : ℕ) (known_seats : List ℕ) : Prop :=
  let interval := total / sample_size
  ∀ i j, i < j → j < known_seats.length →
    known_seats.get! j - known_seats.get! i = (j - i) * interval

theorem fourth_seat_number
  (total_students : ℕ)
  (sample_size : ℕ)
  (known_seats : List ℕ)
  (h_total : total_students = 60)
  (h_sample : sample_size = 4)
  (h_known : known_seats = [3, 18, 48])
  (h_systematic : systematic_sample total_students sample_size known_seats) :
  ∃ (fourth_seat : ℕ), fourth_seat = 33 ∧ systematic_sample total_students sample_size (fourth_seat :: known_seats) :=
by sorry

#check fourth_seat_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_seat_number_l660_66045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l660_66085

def mySequence : List ℕ := [2, 5, 11, 20, 32, 47]

def differences (s : List ℕ) : List ℕ :=
  List.zipWith (·-·) (s.tail) s

theorem sequence_property : 
  let s := mySequence
  let d := differences s
  (∀ i : Fin 3, d[i+1]! - d[i]! = d[1]! - d[0]!) ∧ 
  s[4]! = 32 := by
  sorry

#eval mySequence
#eval differences mySequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l660_66085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l660_66026

theorem power_equation_solution (x : ℝ) : (9 : ℝ)^x = 243 → x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l660_66026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_3_8_plus_6_7_l660_66018

theorem greatest_prime_factor_of_3_8_plus_6_7 :
  (Nat.factors (3^8 + 6^7)).maximum? = some 131 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_3_8_plus_6_7_l660_66018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_ticket_window_l660_66097

/-- Calculates the time needed to cover a remaining distance given an initial movement and total distance to cover -/
noncomputable def time_to_cover_remaining_distance (initial_distance : ℝ) (initial_time : ℝ) (remaining_distance_yards : ℝ) : ℝ :=
  let rate := initial_distance / initial_time
  let remaining_distance_feet := remaining_distance_yards * 3
  remaining_distance_feet / rate

/-- Proves that given the specific conditions, the time to cover the remaining distance is 135 minutes -/
theorem time_to_ticket_window : time_to_cover_remaining_distance 80 40 90 = 135 := by
  -- Unfold the definition of time_to_cover_remaining_distance
  unfold time_to_cover_remaining_distance
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_ticket_window_l660_66097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_difference_l660_66073

theorem rhombus_difference (n : ℕ) (h : n > 3) : 
  (3 * n * (n - 1)) / 2 - (3 * (n - 3) * (n - 2)) / 2 = 6 * n - 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_difference_l660_66073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_b_for_empty_solution_set_l660_66055

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + Real.sqrt a| - |x - Real.sqrt (1 - a)|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 1/2} = {x : ℝ | x ≥ -1/4} := by sorry

-- Part II
theorem range_of_b_for_empty_solution_set (a : ℝ) (h : a ∈ Set.Icc 0 1) :
  (∀ x : ℝ, f a x < b) → b > Real.sqrt (1 - a) + Real.sqrt a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_b_for_empty_solution_set_l660_66055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_box_cost_l660_66067

/-- Given Amber's purchase at the store, prove the cost of one box of tissues. -/
theorem tissue_box_cost (toilet_paper_rolls paper_towel_rolls tissue_boxes 
                         toilet_paper_cost paper_towel_cost total_cost result : ℝ) :
  (toilet_paper_rolls * toilet_paper_cost + paper_towel_rolls * paper_towel_cost + tissue_boxes * result = total_cost) →
  (toilet_paper_rolls = 10 ∧ paper_towel_rolls = 7 ∧ tissue_boxes = 3 ∧ 
   toilet_paper_cost = 1.5 ∧ paper_towel_cost = 2 ∧ total_cost = 35) →
  result = 2 := by
  intro h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_box_cost_l660_66067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_proof_l660_66041

noncomputable def median_of_set (a₁ a₂ a₃ a₄ a₅ : ℝ) : ℝ :=
  (0 - a₅) / 2

theorem median_proof (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h1 : a₁ < a₂) (h2 : a₂ < a₃) (h3 : a₃ < a₄) (h4 : a₄ < a₅) (h5 : a₅ < 0) :
  median_of_set a₁ a₂ a₃ a₄ a₅ = (0 - a₅) / 2 :=
by
  -- Unfold the definition of median_of_set
  unfold median_of_set
  -- The equality now holds by reflexivity
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_proof_l660_66041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_arrangement_proof_l660_66087

def gift_arrangement_count : Nat :=
  let box_count : Nat := 5
  let gift_count : Nat := 4
  let ways_to_choose_empty_box : Nat := box_count
  let ways_to_arrange_gifts : Nat := Nat.factorial gift_count
  ways_to_choose_empty_box * ways_to_arrange_gifts

#eval gift_arrangement_count -- This will evaluate to 120

theorem gift_arrangement_proof :
  gift_arrangement_count = 120 := by
  unfold gift_arrangement_count
  simp
  norm_num
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_arrangement_proof_l660_66087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectories_l660_66038

/-- Given an ellipse with semi-major axis a and semi-minor axis b, this theorem proves
    the trajectories of points D and E, where D is the foot of the perpendicular from
    the origin to a tangent line, and E is the midpoint of the chord formed by the
    tangent line's intersections with the ellipse. -/
theorem ellipse_trajectories (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (C₁ C₂ : Set (ℝ × ℝ)),
    C₁ = {(x, y) | x^2 + y^2 = (a^2 * b^2) / (a^2 + b^2)} ∧
    C₂ = {(x, y) | (a^2 + b^2) * a^2 * b^2 * ((x^2 / a^2) + (y^2 / b^2))^2 = b^4 * x^2 + a^4 * y^2} ∧
    (∀ (x y : ℝ), (x, y) ∈ C₁ ↔ ∃ (θ : ℝ), 
      x^2 / a^2 + y^2 / b^2 = 1 ∧
      x * (y * Real.cos θ - x * Real.sin θ) + y * (x * Real.cos θ + y * Real.sin θ) = 0) ∧
    (∀ (x y : ℝ), (x, y) ∈ C₂ ↔ ∃ (α : ℝ),
      x^2 / a^2 + y^2 / b^2 = 1 ∧
      x * Real.cos α + y * Real.sin α = 0 ∧
      ((x * Real.cos α + y * Real.sin α)^2 + (y * Real.cos α - x * Real.sin α)^2) / 4 + x^2 + y^2 = (a^2 + b^2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectories_l660_66038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_zeros_l660_66053

def f (r : ℝ) (x : ℝ) : ℝ := x^2 - 2*r*x + r

def g (r : ℝ) (s : ℝ) (x : ℝ) : ℝ := 27*x^3 - 27*r*x^2 + s*x - r^6

theorem polynomial_zeros (r s : ℝ) : 
  (∀ x, f r x = 0 → x ≥ 0) ∧
  (∀ x, g r s x = 0 → x ≥ 0) →
  ((r = 0 ∧ s = 0) ∨ (r = 1 ∧ s = 9)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_zeros_l660_66053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_seat_covers_all_l660_66075

/-- Represents the circular table with 10 seats. -/
def Table := Fin 10

/-- The function that determines the next seat to be shaded. -/
def nextSeat (n : Nat) : Table :=
  ⟨(1 + (3 * n * (n + 1) / 2)) % 10, by
    apply Nat.mod_lt
    · exact Nat.zero_lt_succ 9
  ⟩

/-- The set of shaded seats after n moves. -/
def shadedSeats (n : Nat) : Set Table :=
  {s | ∃ k < n, nextSeat k = s}

/-- The theorem stating that the 13th shaded seat is the first to cover all positions. -/
theorem thirteenth_seat_covers_all :
  (∀ s : Table, s ∈ shadedSeats 13) ∧
  (∀ m < 13, ∃ s : Table, s ∉ shadedSeats m) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_seat_covers_all_l660_66075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_ax_solution_l660_66025

theorem odd_ax_solution (a x y n m : ℕ) : 
  a > 0 → x > 0 → y > 0 → n > 0 → m > 0 →
  a * (x^n - x^m) = (a * x^m - 4) * y^2 →
  m % 2 = n % 2 →
  Odd (a * x) →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_ax_solution_l660_66025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_and_range_of_a_l660_66032

noncomputable def f (x : ℝ) := (x + 1) * Real.log (x + 1)

def g (a x : ℝ) := a * x^2 + x

theorem min_value_f_and_range_of_a :
  (∃ x, ∀ y, f y ≥ f x ∧ f x = -1/Real.exp 1) ∧
  (∀ a, (∀ x ≥ 0, f x ≤ g a x) ↔ a ≥ 1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_and_range_of_a_l660_66032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_are_false_l660_66052

-- Define the types for planes and lines
structure Plane : Type
structure Line : Type

-- Define the relations
def parallel_planes : Plane → Plane → Prop := sorry
def parallel_lines : Line → Line → Prop := sorry
def perpendicular_lines : Line → Line → Prop := sorry
def perpendicular_planes : Plane → Plane → Prop := sorry
def subset_plane_line : Plane → Line → Prop := sorry

theorem propositions_are_false :
  (∃ (α β : Plane) (l m : Line),
    parallel_planes α β ∧
    subset_plane_line α l ∧
    subset_plane_line β m ∧
    ¬ parallel_lines l m) ∧
  (∃ (α β : Plane) (l m : Line),
    perpendicular_lines l m ∧
    subset_plane_line α l ∧
    subset_plane_line β m ∧
    ¬ perpendicular_planes α β) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_are_false_l660_66052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l660_66021

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (a^2 * Real.exp a) / (Real.exp a - (a + 1) * x)

-- State the theorem
theorem range_of_f :
  ∀ x a : ℝ,
  x ∈ Set.Ico 0 1 →
  (2 - a) * Real.exp a = x * (2 + a) →
  ∃ y : ℝ, y ∈ Set.Ioo 2 4 ∧ f x a = y ∨ f x a = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l660_66021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_difference_l660_66092

/-- Given a function f(x) = bx + ln(x) where b is a real number,
    if a line with slope k passing through the origin is tangent to the curve y = f(x),
    then k - b = 1/e. -/
theorem tangent_line_slope_difference (b : ℝ) :
  ∃ (k x : ℝ) (f : ℝ → ℝ), 
    (∀ t, f t = b * t + Real.log t) ∧
    (k * x = f x) ∧
    (k = (f x - 0) / (x - 0)) ∧
    (k - b = 1 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_difference_l660_66092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l660_66065

-- Define the concept of a sequence
def Sequence := ℕ → ℝ

-- Define the concept of analogical reasoning
def AnalogicalReasoning (A B : Type) := A → B

-- Define the concept of syllogistic reasoning
def SyllogisticReasoning (P Q R : Prop) := (P → Q) → (Q → R) → (P → R)

-- Define placeholder types for geometric concepts
structure PlanarTriangle
structure SpatialTetrahedron
structure SpatialParallelepiped

-- Define the statements
def statement1 (a : Sequence) : Prop := 
  a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 3 → ∀ n : ℕ, a n = n

def statement2 : Prop := 
  ∃ (f : AnalogicalReasoning PlanarTriangle SpatialTetrahedron), True

def statement3 : Prop := 
  ∃ (f : AnalogicalReasoning PlanarTriangle SpatialParallelepiped), True

def statement4 : Prop := 
  ∃ (s : SyllogisticReasoning 
        (∀ m : ℕ, 3 ∣ m) 
        (∀ m : ℕ, 9 ∣ m) 
        (∀ m : ℕ, 3 ∣ m → 9 ∣ m)), 
    ¬(∀ m : ℕ, 3 ∣ m → 9 ∣ m)

theorem correct_statements : 
  (¬ ∀ a : Sequence, statement1 a) ∧ 
  statement2 ∧ 
  (¬ statement3) ∧ 
  statement4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l660_66065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_l660_66027

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points
variable (O A B P Q R S M N T : V)

-- Define ratios
variable (a b e f : ℝ)

-- Define the constant k for PR:PA and QS:QB
variable (k : ℝ)

-- Axioms based on the problem conditions
axiom ax1 : ∃ (t : ℝ), P = A + t • (R - A)
axiom ax2 : ∃ (t : ℝ), Q = B + t • (S - B)
axiom ax3 : ∃ (c : ℝ), (R - A) = c • (S - B)
axiom ax4 : ∃ (c : ℝ), (P - A) = c • (Q - B)
axiom ax5 : M - A = (e / (e + f)) • (B - A)
axiom ax6 : N - P = (e / (e + f)) • (Q - P)
axiom ax7 : T - R = (e / (e + f)) • (S - R)

-- Theorem to prove
theorem points_collinear :
  ∃ (t : ℝ), T = N + t • (M - N) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_l660_66027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_range_l660_66057

theorem trig_expression_range (α : ℝ) (h : 0 < α ∧ α < π/2) :
  ∃ x, x ∈ Set.Ioo 2 (3/2 + Real.sqrt 2) ∧
  x = (Real.sin α + Real.tan α) * (Real.cos α + (Real.cos α / Real.sin α)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_range_l660_66057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l660_66014

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (-2 * x)

-- State the theorem
theorem f_derivative :
  deriv f = fun x => -2 * Real.exp (-2 * x) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l660_66014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_graph_l660_66094

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem symmetric_sine_graph (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : ∀ x, f x φ = f ((π / 3) - x) φ) : 
  φ = π / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_graph_l660_66094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_megan_works_twenty_days_l660_66031

/-- Represents Megan's work schedule and earnings --/
structure WorkSchedule where
  hoursPerDay : ℚ
  hourlyRate : ℚ
  totalEarningsTwoMonths : ℚ

/-- Calculates the number of days Megan works per month --/
def daysWorkedPerMonth (schedule : WorkSchedule) : ℚ :=
  schedule.totalEarningsTwoMonths / (2 * schedule.hoursPerDay * schedule.hourlyRate)

/-- Theorem stating that Megan works 20 days per month --/
theorem megan_works_twenty_days (schedule : WorkSchedule)
  (h1 : schedule.hoursPerDay = 8)
  (h2 : schedule.hourlyRate = 15/2)
  (h3 : schedule.totalEarningsTwoMonths = 2400) :
  daysWorkedPerMonth schedule = 20 := by
  sorry

#eval daysWorkedPerMonth ⟨8, 15/2, 2400⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_megan_works_twenty_days_l660_66031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_representation_final_digit_base_842_final_digit_1_l660_66090

theorem base_representation_final_digit (n : ℕ) (lower upper : ℕ) : 
  (∀ b ∈ Finset.range (upper - lower + 1), 
    ¬(n % (b + lower) = 1)) → 
  (Finset.filter (λ b ↦ n % (b + lower) = 1) (Finset.range (upper - lower + 1))).card = 0 :=
by
  sorry

theorem base_842_final_digit_1 : 
  (Finset.filter (λ b ↦ 842 % (b + 3) = 1) (Finset.range 8)).card = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_representation_final_digit_base_842_final_digit_1_l660_66090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_zero_l660_66084

/-- Definition of a hyperbola Γ -/
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

/-- Definition of eccentricity for a hyperbola -/
noncomputable def Eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

/-- The dot product of two 2D vectors -/
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The main theorem -/
theorem hyperbola_dot_product_zero
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0)
  (he : Eccentricity a b = Real.sqrt 2)
  (A B C : ℝ × ℝ)
  (hA : A ∈ Hyperbola a b)
  (hB : B ∈ Hyperbola a b)
  (hC : C ∈ Hyperbola a b)
  (hright : A.1 = a ∧ A.2 = 0)
  (hparallel : B.2 = C.2) :
  DotProduct (B.1 - A.1, B.2 - A.2) (C.1 - A.1, C.2 - A.2) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_zero_l660_66084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l660_66020

theorem min_value_of_expression (a b : ℝ) (h : a - 2*b + 8 = 0) :
  (∀ x y : ℝ, x - 2*y + 8 = 0 → (2:ℝ)^x + (1/(4:ℝ))^y ≥ (2:ℝ)^a + (1/(4:ℝ))^b) →
  (2:ℝ)^a + (1/(4:ℝ))^b = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l660_66020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_decreasing_interval_minimum_a_for_no_zeros_l660_66009

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + x

theorem tangent_line_and_decreasing_interval (a : ℝ) :
  (∃ k : ℝ, k * (0 - 1) + g a 1 = 2 ∧ (deriv (g a)) 1 = k) →
  ∀ x ∈ Set.Ioo 0 2, (deriv (g a)) x < 0 :=
sorry

theorem minimum_a_for_no_zeros (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (1/2), f a x ≠ 0) ↔ a ≥ 2 - 4 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_decreasing_interval_minimum_a_for_no_zeros_l660_66009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem1_problem2_l660_66099

-- Problem 1
noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 2|

theorem problem1 (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a > f b) : a * b < 1 := by
  sorry

-- Problem 2
theorem problem2 (x a : ℝ) (h1 : 0 < x) (h2 : x < 1) (h3 : a > 0) (h4 : a ≠ 1) :
  |Real.log (1 - x) / Real.log a| > |Real.log (1 + x) / Real.log a| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem1_problem2_l660_66099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raviraj_distance_to_home_l660_66093

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the straight-line distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents Raviraj's journey -/
def raviraj_journey : Point :=
  Point.mk (-30) 0

theorem raviraj_distance_to_home : 
  distance (Point.mk 0 0) raviraj_journey = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raviraj_distance_to_home_l660_66093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l660_66091

/-- Represents a parabola -/
structure Parabola where
  /-- The coefficient in the standard equation y^2 = ax -/
  a : ℝ

/-- Defines the orientation of a parabola -/
def opens_left (p : Parabola) : Prop := p.a < 0

/-- Defines the distance from focus to directrix for a parabola -/
noncomputable def focus_directrix_distance (p : Parabola) : ℝ := abs p.a / 2

/-- The theorem stating the standard equation of the parabola C -/
theorem parabola_equation (C : Parabola) 
  (h1 : opens_left C) 
  (h2 : focus_directrix_distance C = 3) : 
  C.a = -6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l660_66091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_equal_rect_perimeter_greater_l660_66047

-- Define the side length of the square
noncomputable def square_side : ℝ := Real.sqrt 3 + 3

-- Define the dimensions of the rectangle
noncomputable def rect_length : ℝ := Real.sqrt 72 + 3 * Real.sqrt 6
noncomputable def rect_width : ℝ := Real.sqrt 2

-- Theorem stating that the areas are equal
theorem areas_equal :
  square_side ^ 2 = rect_length * rect_width := by
  sorry

-- Theorem stating that the rectangle's perimeter is greater
theorem rect_perimeter_greater :
  2 * (rect_length + rect_width) > 4 * square_side := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_equal_rect_perimeter_greater_l660_66047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arc_placement_l660_66019

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

/-- Represents an arc on a great circle of a sphere -/
structure Arc where
  startPoint : ℝ × ℝ × ℝ
  endPoint : ℝ × ℝ × ℝ
  length : ℝ

/-- Predicate to check if two arcs intersect -/
def intersect (a₁ a₂ : Arc) : Prop :=
  sorry

/-- Theorem stating that it's impossible to place n non-intersecting arcs on great circles 
    of a sphere if any arc's length exceeds π + 2π/n -/
theorem impossible_arc_placement (s : Sphere) (n : ℕ) (arcs : Finset Arc) :
  (∀ a ∈ arcs, a.length > Real.pi + 2 * Real.pi / ↑n) →
  (arcs.card = n) →
  (∀ a₁ a₂, a₁ ∈ arcs → a₂ ∈ arcs → a₁ ≠ a₂ → ¬ intersect a₁ a₂) →
  False :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arc_placement_l660_66019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_max_a_l660_66082

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (x^2 - 3*x + 2)

theorem f_monotonicity_and_max_a :
  -- Part 1: Monotonicity when a = 1
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/2 → f 1 x₁ < f 1 x₂) ∧
  (∀ x₁ x₂, 1/2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f 1 x₁ > f 1 x₂) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f 1 x₁ < f 1 x₂) ∧
  -- Part 2: Maximum value of a
  (∀ a, a > 0 ∧ (∀ x, x > 1 → f a x ≥ 0) → a ≤ 1) ∧
  (∀ x, x > 1 → f 1 x ≥ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_max_a_l660_66082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_see_again_proof_sum_of_numerator_and_denominator_l660_66015

/-- The time when Jenny and Kenny can see each other again -/
noncomputable def time_to_see_again : ℚ := 160 / 3

/-- Kenny's speed in feet per second -/
def kenny_speed : ℚ := 3

/-- Jenny's speed in feet per second -/
def jenny_speed : ℚ := 1

/-- Distance between parallel paths in feet -/
def path_distance : ℚ := 200

/-- Diameter of the circular building in feet -/
def building_diameter : ℚ := 100

/-- Initial distance between Jenny and Kenny when the building blocks their line of sight -/
def initial_blocked_distance : ℚ := 200

theorem time_to_see_again_proof :
  let t := time_to_see_again
  let relative_speed := kenny_speed - jenny_speed
  let half_path_distance := path_distance / 2
  building_diameter ^ 2 + (relative_speed * t) ^ 2 = (half_path_distance + building_diameter) ^ 2 :=
by sorry

/-- The sum of the numerator and denominator of time_to_see_again when expressed as a fraction in lowest terms -/
theorem sum_of_numerator_and_denominator :
  let n := 160  -- numerator
  let d := 3    -- denominator
  n + d = 163 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_see_again_proof_sum_of_numerator_and_denominator_l660_66015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l660_66089

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (|x + 1| + |x + 2| + |x - 3|)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 1/5 ∧ ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l660_66089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_right_triangle_area_l660_66005

theorem smallest_right_triangle_area :
  let a := 6
  let b := 8
  let area1 := a * (Real.sqrt (b^2 - a^2)) / 2
  let area2 := a * b / 2
  min area1 area2 = 6 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_right_triangle_area_l660_66005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_line_parallel_and_through_p_l660_66074

-- Define a general continuous function f in two variables
variable (f : ℝ → ℝ → ℝ)

-- Define a point P not on the line l: f(x, y) = 0
variable (x₀ y₀ : ℝ)
variable (h : f x₀ y₀ ≠ 0)

-- Define the new line g(x, y) = 0
def g (f : ℝ → ℝ → ℝ) (x₀ y₀ : ℝ) (x y : ℝ) : ℝ := f x y - f x₀ y₀

-- Theorem statement
theorem new_line_parallel_and_through_p (f : ℝ → ℝ → ℝ) (x₀ y₀ : ℝ) (h : f x₀ y₀ ≠ 0) :
  (∀ x y, g f x₀ y₀ x y = 0 → f x y = f x₀ y₀) ∧
  g f x₀ y₀ x₀ y₀ = 0 ∧
  (∀ x₁ y₁ x₂ y₂, f x₁ y₁ = f x₂ y₂ → g f x₀ y₀ x₁ y₁ = g f x₀ y₀ x₂ y₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_line_parallel_and_through_p_l660_66074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l660_66051

-- Define the polar coordinate system
noncomputable def polar_to_rect (ρ : ℝ) (θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define curve C₁
def C₁ (r : ℝ) (ρ θ : ℝ) : Prop := ρ * (ρ - 4 * Real.cos θ) = r^2 - 4

-- Define curve C₂
def C₂ (r θ : ℝ) (x y : ℝ) : Prop := x = 4 + Real.sqrt 3 * r * Real.cos θ ∧ y = Real.sqrt 3 * r * Real.sin θ

-- Define curve C₃ (implicitly as the intersection of C₁ and C₂)
def C₃ (x y : ℝ) : Prop := ∃ r θ, C₁ r (Real.sqrt (x^2 + y^2)) θ ∧ C₂ r θ x y

-- Define line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t / 2, t * Real.sqrt 3 / 2)

-- Theorem statement
theorem intersection_distance_difference :
  ∃ A B : ℝ × ℝ, 
    C₃ A.1 A.2 ∧ C₃ B.1 B.2 ∧ 
    (∃ t₁ t₂, line_l t₁ = A ∧ line_l t₂ = B) ∧
    Real.sqrt (A.1^2 + A.2^2) - Real.sqrt (B.1^2 + B.2^2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l660_66051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_digits_3_pow_100_l660_66083

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the representation of a positive real number
def positiveRealRepresentation (N : ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℤ), N = a * 10^n ∧ 1 ≤ a ∧ a < 10

-- Define the logarithm property
def logProperty (N : ℝ) : Prop :=
  ∃ (n : ℤ) (a : ℝ), log10 N = n + log10 a ∧ 0 ≤ log10 a ∧ log10 a < 1

-- Define the number of digits property
def numberOfDigits (N : ℝ) (d : ℕ) : Prop :=
  ∃ (n : ℤ), n > 0 ∧ log10 N = n + log10 (N / 10^n) ∧ d = n.toNat + 1

-- Approximate value of log10 3
def log10_3_approx : ℝ := 0.4771

-- Theorem statement
theorem number_of_digits_3_pow_100 :
  ∀ N : ℝ,
  positiveRealRepresentation N →
  logProperty N →
  (∀ d : ℕ, numberOfDigits N d → True) →
  log10 3 = log10_3_approx →
  numberOfDigits (3^100) 48 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_digits_3_pow_100_l660_66083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ratio_l660_66069

/-- Represents a point in 2D space. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on a circle with given radius and center at origin. -/
def Point.onCircle (p : Point) (r : ℝ) : Prop :=
  p.x^2 + p.y^2 = r^2

/-- Calculates the Euclidean distance between two points. -/
noncomputable def dist (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the length of the minor arc between two points on a circle. -/
noncomputable def minorArcLength (r : ℝ) (p1 p2 : Point) : ℝ :=
  sorry  -- Definition would depend on how we represent arcs

/-- Given a circle with radius r and points A, B, C on the circle,
    prove that AB/BC = √2 * sin(1) under specific conditions. -/
theorem circle_ratio (r : ℝ) (A B C : Point) 
  (h1 : A.onCircle r) (h2 : B.onCircle r) (h3 : C.onCircle r)
  (h4 : dist A B = dist A C) (h5 : dist A B > r) (h6 : minorArcLength r B C = 2*r) : 
  (dist A B) / (dist B C) = Real.sqrt 2 * Real.sin 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ratio_l660_66069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expanded_body_properties_l660_66043

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  volume : ℝ
  surfaceArea : ℝ
  edges : List (ℝ × ℝ)  -- List of (length, dihedral angle) pairs

/-- Calculates the volume of the expanded body -/
noncomputable def expandedVolume (p : ConvexPolyhedron) (d : ℝ) : ℝ :=
  p.volume + p.surfaceArea * d + 
  (1/2) * d^2 * (p.edges.map (λ (l, φ) => (Real.pi - φ) * l)).sum +
  (4/3) * Real.pi * d^3

/-- Calculates the surface area of the expanded body -/
noncomputable def expandedSurfaceArea (p : ConvexPolyhedron) (d : ℝ) : ℝ :=
  p.surfaceArea + 
  d * (p.edges.map (λ (l, φ) => (Real.pi - φ) * l)).sum +
  4 * Real.pi * d^2

/-- Theorem stating the correctness of the expanded volume and surface area calculations -/
theorem expanded_body_properties (p : ConvexPolyhedron) (d : ℝ) :
  (expandedVolume p d = p.volume + p.surfaceArea * d + 
   (1/2) * d^2 * (p.edges.map (λ (l, φ) => (Real.pi - φ) * l)).sum +
   (4/3) * Real.pi * d^3) ∧
  (expandedSurfaceArea p d = p.surfaceArea + 
   d * (p.edges.map (λ (l, φ) => (Real.pi - φ) * l)).sum +
   4 * Real.pi * d^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expanded_body_properties_l660_66043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_strange_pairs_l660_66060

-- Define the function F(n) that returns the greatest prime factor of n
noncomputable def F (n : ℕ) : ℕ := 
  Nat.factors n |>.maximum?  |>.getD 1

-- Define what it means to be a strange pair
def is_strange_pair (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  ∀ n : ℕ, n ≥ 2 → F n * F (n + 1) ≠ p * q

-- State the theorem
theorem infinitely_many_strange_pairs :
  ∀ k : ℕ, ∃ S : Finset (ℕ × ℕ),
    S.card > k ∧ 
    ∀ pair ∈ S, is_strange_pair pair.1 pair.2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_strange_pairs_l660_66060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_inequality_proof_l660_66076

-- Part I
theorem inequality_solution_set (x : ℝ) :
  2^x + 2^(|x|) ≥ 2 * Real.sqrt 2 ↔ x ≥ 1/2 ∨ x ≤ Real.log (Real.sqrt 2 - 1) / Real.log 2 :=
sorry

-- Part II
theorem inequality_proof (m n a b : ℝ) (hm : m > 0) (hn : n > 0) :
  a^2 / m + b^2 / n ≥ (a + b)^2 / (m + n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_inequality_proof_l660_66076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_ball_probability_nth_draw_l660_66071

/-- Represents the state of the bag after each draw -/
structure BagState where
  blackBalls : ℕ
  whiteBalls : ℕ

/-- The initial state of the bag -/
def initialState : BagState := ⟨1, 1⟩

/-- The process of drawing a ball and adding a black ball -/
def drawProcess (state : BagState) : BagState :=
  ⟨state.blackBalls + 1, state.whiteBalls⟩

/-- The probability of drawing a black ball on the n-th draw -/
noncomputable def blackBallProbability (n : ℕ) : ℝ :=
  1 - 1 / (2 ^ n)

/-- Theorem stating the probability of drawing a black ball on the n-th draw -/
theorem black_ball_probability_nth_draw (n : ℕ) :
  blackBallProbability n = 1 - 1 / (2 ^ n) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_ball_probability_nth_draw_l660_66071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_descending_order_proof_l660_66004

theorem descending_order_proof (a b c : ℝ) : 
  a = (3/5)^(-(1/3 : ℝ)) → b = (4/3)^(-(1/2 : ℝ)) → c = Real.log (3/5) → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_descending_order_proof_l660_66004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_length_correct_l660_66078

/-- A trapezoid ABCD with given properties -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  AB : ℝ
  CD : ℝ
  area_eq : area = 200
  altitude_eq : altitude = 10
  AB_eq : AB = 12
  CD_eq : CD = 22

/-- The length of BC in the trapezoid -/
noncomputable def BC_length (t : Trapezoid) : ℝ :=
  20 - Real.sqrt 11 - 6 * Real.sqrt 6

/-- Theorem stating that BC_length is correct for the given trapezoid -/
theorem BC_length_correct (t : Trapezoid) : 
  BC_length t = 20 - Real.sqrt 11 - 6 * Real.sqrt 6 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_length_correct_l660_66078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l660_66023

/-- Race parameters and conditions -/
structure RaceData where
  samTime : ℝ  -- Sam's time to run 100 m
  headStart : ℝ  -- Distance John starts ahead
  raceLength : ℝ  -- Total race distance

/-- Calculate John's time to run the full race distance -/
noncomputable def johnTime (data : RaceData) : ℝ :=
  (data.raceLength * data.samTime) / (data.raceLength - data.headStart)

/-- Calculate the time difference between Sam and John -/
noncomputable def timeDifference (data : RaceData) : ℝ :=
  johnTime data - data.samTime

/-- Theorem stating the time difference between Sam and John -/
theorem race_time_difference (data : RaceData) 
  (h1 : data.samTime = 13)
  (h2 : data.headStart = 35)
  (h3 : data.raceLength = 100) :
  timeDifference data = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l660_66023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frans_speed_calculation_l660_66039

/-- Fran's required speed to match Joann's distance -/
noncomputable def frans_speed (joanns_speed : ℝ) (joanns_time : ℝ) (frans_time : ℝ) : ℝ :=
  (joanns_speed * joanns_time) / frans_time

theorem frans_speed_calculation (joanns_speed : ℝ) (joanns_time : ℝ) (frans_time : ℝ)
  (h1 : joanns_speed = 15)
  (h2 : joanns_time = 4)
  (h3 : frans_time = 3.5) :
  frans_speed joanns_speed joanns_time frans_time = 120 / 7 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval frans_speed 15 4 3.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frans_speed_calculation_l660_66039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_real_part_of_fifth_power_l660_66046

noncomputable def z1 : ℂ := -2
noncomputable def z2 : ℂ := -Real.sqrt 3 + Complex.I
noncomputable def z3 : ℂ := -Real.sqrt 2 + Real.sqrt 2 * Complex.I
noncomputable def z4 : ℂ := 2 * Complex.I
noncomputable def z5 : ℂ := -1 + Real.sqrt 3 * Complex.I

theorem largest_real_part_of_fifth_power :
  (z2^5).re ≥ (z1^5).re ∧
  (z2^5).re ≥ (z3^5).re ∧
  (z2^5).re ≥ (z4^5).re ∧
  (z2^5).re ≥ (z5^5).re :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_real_part_of_fifth_power_l660_66046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_divisible_by_101_l660_66062

def a : ℕ → ℕ
  | 10 => 101
  | 0 => 101  -- Adding this case to cover Nat.zero
  | n+1 => 101 * a n + (n+1)

theorem least_divisible_by_101 : ∀ k : ℕ, k > 10 ∧ k < 134 → a k % 101 ≠ 0 ∧ a 134 % 101 = 0 := by
  sorry

#eval a 134 % 101  -- This will evaluate the result for n = 134

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_divisible_by_101_l660_66062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l660_66054

-- Define the expansion
def expansion (x : ℝ) (n : ℕ) : ℝ := (x^(2/3) + 3*x^2)^n

-- Define the ratio condition
def ratio_condition (n : ℕ) : Prop := (4^n) / (2^n) = 32

-- Theorem statement
theorem expansion_properties :
  ∃ (n : ℕ), 
    ratio_condition n ∧ 
    (n = 5) ∧
    (∀ x : ℝ, 
      -- Terms with largest binomial coefficient
      (Nat.choose n 2 * x^(10/3) * 9 * x^4 = 90 * x^6) ∧
      (Nat.choose n 3 * x^(4/3) * 27 * x^6 = 270 * x^(22/3)) ∧
      -- Term with largest coefficient
      (Nat.choose n 4 * x^(2/3) * 81 * x^8 = 405 * x^(26/3))) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l660_66054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_distance_l660_66056

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 4)^2 + (y + 1)^2 = 25

-- Define point M
def M : ℝ × ℝ := (0, 2)

-- Define line l'
def line_l' (x y a : ℝ) : Prop := 4*x - a*y + 2 = 0

-- Define the tangent line l
def line_l (x y : ℝ) : Prop := 4*x - 3*y + 6 = 0

-- State the theorem
theorem tangent_line_distance :
  ∃ a : ℝ, 
    (∀ x y : ℝ, line_l x y → my_circle x y → (x, y) = M) ∧ 
    (∀ x y : ℝ, line_l x y ↔ line_l' x y a) ∧
    (let d := |6 - 2| / Real.sqrt (4^2 + a^2);
     d = 4/5) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_distance_l660_66056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_theorem_l660_66029

-- Define the cubic polynomial and its roots
variable (a b c r s t : ℝ)

-- Define the polynomial
def cubic_polynomial (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Assume the polynomial has three real roots r, s, t
axiom root_r : cubic_polynomial a b c r = 0
axiom root_s : cubic_polynomial a b c s = 0
axiom root_t : cubic_polynomial a b c t = 0

-- Assume the roots are in descending order
axiom root_order : r ≥ s ∧ s ≥ t

-- Define k
def k (a b : ℝ) : ℝ := a^2 - 3*b

-- Theorem to prove
theorem cubic_polynomial_theorem :
  k a b ≥ 0 ∧ Real.sqrt (k a b) ≤ r - t :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_theorem_l660_66029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inverse_property_l660_66040

-- Define the function f
noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then 2 * c * x + d else 9 - 2 * x

-- State the theorem
theorem function_inverse_property (c d : ℝ) :
  (∀ x, f c d (f c d x) = x) → c + d = 4.25 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inverse_property_l660_66040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l660_66064

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := x^3 - (Real.sqrt 3 / 3) * x + 1/4

-- Define the slope of the tangent line
noncomputable def tangent_slope (x : ℝ) : ℝ := 3 * x^2 - Real.sqrt 3 / 3

-- Define the range of α
def alpha_range (α : ℝ) : Prop :=
  (0 ≤ α ∧ α < Real.pi / 2) ∨ (5 * Real.pi / 6 ≤ α ∧ α < Real.pi)

-- Theorem statement
theorem tangent_slope_range :
  ∀ x : ℝ, alpha_range (Real.arctan (tangent_slope x)) := by
  sorry

#check tangent_slope_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l660_66064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l660_66000

noncomputable def vector_projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_squared := v.1 * v.1 + v.2 * v.2
  let scalar := dot_product / magnitude_squared
  (scalar * v.1, scalar * v.2)

theorem projection_theorem (v : ℝ × ℝ) :
  vector_projection (0, 4) v = (-12/13, 4/13) →
  vector_projection (3, 2) v = (21/10, -7/10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l660_66000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_second_derivative_sign_l660_66059

open Real

-- Define the function f(x) = e^x + 1/x
noncomputable def f (x : ℝ) : ℝ := exp x + 1 / x

-- State the theorem
theorem f_second_derivative_sign 
  (x₀ : ℝ) 
  (h_x₀_pos : x₀ > 0) 
  (h_f_second_deriv_zero : (deriv^[2] f) x₀ = 0) 
  (m : ℝ) 
  (h_m : 0 < m ∧ m < x₀) 
  (n : ℝ) 
  (h_n : x₀ < n) : 
  (deriv^[2] f) m < 0 ∧ (deriv^[2] f) n > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_second_derivative_sign_l660_66059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l660_66066

/-- Calculates the average speed of a train given two segments of its journey -/
noncomputable def averageSpeed (x : ℝ) : ℝ :=
  let distance1 := x
  let speed1 := (30 : ℝ)
  let distance2 := 2 * x
  let speed2 := (20 : ℝ)
  let totalDistance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let totalTime := time1 + time2
  totalDistance / totalTime

theorem train_average_speed (x : ℝ) (h : x > 0) : averageSpeed x = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l660_66066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_f_g_l660_66034

noncomputable def f (x : ℝ) : ℝ := 2 * x

noncomputable def g (x : ℝ) : ℝ := -(3 * x - 1) / x

theorem product_f_g (x : ℝ) (hx : x ≠ 0) : f x * g x = 6 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_f_g_l660_66034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_theorem_l660_66068

/-- A quadrilateral inscribed in a circle with one side as diameter -/
structure InscribedQuadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  inscribed : Bool -- Placeholder for the inscribed property
  diameter : Bool -- Placeholder for the diameter property

/-- The length of a line segment between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: In an inscribed quadrilateral ABCD with diameter AD, 
    if AB = 5, AC = 6, and BD = 7, then CD = √38 -/
theorem inscribed_quadrilateral_theorem (quad : InscribedQuadrilateral) 
    (h1 : distance quad.A quad.B = 5)
    (h2 : distance quad.A quad.C = 6)
    (h3 : distance quad.B quad.D = 7) : 
  distance quad.C quad.D = Real.sqrt 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_theorem_l660_66068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_l660_66003

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 5)

-- State the theorem
theorem smallest_x_in_domain_of_f_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f (f x) = y) → x ≥ 30 ∧ 
  ∀ z : ℝ, z < 30 → ¬(∃ y : ℝ, f (f z) = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_l660_66003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_largest_root_bound_l660_66098

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := x^5 - 10*x^3 + a*x^2 + b*x + c

theorem max_largest_root_bound (a b c : ℝ) :
  (∀ x, f a b c x = 0 → x ∈ Set.univ) →
  ∃ m : ℝ, (∀ x, f a b c x = 0 → x ≤ m) ∧ m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_largest_root_bound_l660_66098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_triangle_area_l660_66096

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  right_angle : a^2 + b^2 = c^2

-- Define the areas
noncomputable def triangle_area (t : RightTriangle) : ℝ := (1/2) * t.a * t.b

noncomputable def semicircle_area (r : ℝ) : ℝ := (1/2) * Real.pi * r^2

noncomputable def shaded_area (t : RightTriangle) : ℝ :=
  semicircle_area (t.a/2) + semicircle_area (t.b/2) - semicircle_area (t.c/2)

-- State the theorem
theorem shaded_area_equals_triangle_area (t : RightTriangle) :
  shaded_area t = triangle_area t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_triangle_area_l660_66096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_rectangle_ratio_l660_66024

/-- Given a square of side length 4, with E and F as midpoints of adjacent sides,
    and AG perpendicular to BE, prove that when reassembled into a rectangle,
    the ratio of the rectangle's height to its base is 4. -/
theorem square_to_rectangle_ratio : 
  ∀ (A B C D E F G X Y Z : ℝ × ℝ),
  let square_side : ℝ := 4
  let square_area : ℝ := square_side ^ 2
  -- E and F are midpoints of adjacent sides
  norm (A.1 - E.1, A.2 - E.2) = square_side / 2 →
  norm (B.1 - F.1, B.2 - F.2) = square_side / 2 →
  -- AG is perpendicular to BE
  (G.1 - A.1) * (E.1 - B.1) + (G.2 - A.2) * (E.2 - B.2) = 0 →
  -- The reassembled rectangle has the same area as the original square
  (X.2 - Y.2) * (Y.1 - Z.1) = square_area →
  -- The ratio of height to base of the rectangle is 4
  (X.2 - Y.2) / (Y.1 - Z.1) = 4 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_rectangle_ratio_l660_66024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_circle_circumscribing_equilateral_triangle_l660_66050

/-- The area of a circle circumscribing an equilateral triangle with side length 4 is 16π/3 -/
theorem area_circle_circumscribing_equilateral_triangle :
  let side_length : ℝ := 4
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  let circle_radius : ℝ := side_length / Real.sqrt 3
  let circle_area : ℝ := Real.pi * circle_radius^2
  circle_area = (16 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_circle_circumscribing_equilateral_triangle_l660_66050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l660_66049

noncomputable def proj_vector (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (v.1 * w.1 + v.2 * w.2) / (w.1 * w.1 + w.2 * w.2)
  (scalar * w.1, scalar * w.2)

theorem projection_problem (k y : ℝ) :
  let v : ℝ × ℝ := (2 * k, y * k)
  let w : ℝ × ℝ := (4, 6)
  proj_vector v w = (-6, -9) → y = -23 ∧ k = 39 / 65 := by
  sorry

#check projection_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l660_66049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_row_probability_l660_66037

/-- The probability of selecting 2 students from the same row in a 3x2 arrangement --/
theorem same_row_probability (total_students : Nat) (rows : Nat) (cols : Nat) 
  (h1 : total_students = 6)
  (h2 : rows = 3)
  (h3 : cols = 2)
  (h4 : rows * cols = total_students) :
  (Nat.choose rows 1 * Nat.choose cols 2 : Rat) / Nat.choose total_students 2 = 1 / 5 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and might cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_row_probability_l660_66037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_land_area_calculation_l660_66007

/-- Represents the annual payment for the land in yuan per mu -/
noncomputable def annual_payment (wheat_price : ℝ) (land_area : ℝ) : ℝ :=
  (70 * land_area - 800) / wheat_price + 800 / land_area

/-- The land area in mu -/
def land_area : ℝ := 20

theorem land_area_calculation :
  (annual_payment 1.2 land_area = 70) ∧
  (annual_payment 1.6 land_area = 80) :=
by
  sorry

#eval land_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_land_area_calculation_l660_66007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l660_66080

open Set Real

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 - 5*x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = (Iio 2) ∪ (Ioo 2 3) ∪ (Ioi 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l660_66080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_base_area_l660_66028

/-- Represents the properties of a water tank -/
structure WaterTank where
  fill_rate : ℝ  -- Liters per minute
  height_increase_rate : ℝ  -- Centimeters per minute

/-- Calculates the base area of a water tank given its properties -/
noncomputable def calculate_base_area (tank : WaterTank) : ℝ :=
  (tank.fill_rate * 1000) / tank.height_increase_rate

/-- Theorem stating that a water tank with given properties has a base area of 100 square centimeters -/
theorem water_tank_base_area :
  let tank : WaterTank := { fill_rate := 1, height_increase_rate := 10 }
  calculate_base_area tank = 100 := by
  sorry

#check water_tank_base_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_base_area_l660_66028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_of_divisors_l660_66013

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def sum_of_squared_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d ↦ n % d = 0) (Finset.range (n + 1))).sum (λ d ↦ d * d)

theorem smallest_number_of_divisors (n : ℕ) :
  (is_divisible (n + 1) 24) →
  (is_divisible (sum_of_squared_divisors n) 48) →
  (∃ m, (Nat.divisors n).card = m ∧ ∀ k, (is_divisible (k + 1) 24) →
    (is_divisible (sum_of_squared_divisors k) 48) →
    (Nat.divisors k).card ≥ m) →
  (Nat.divisors n).card = 48 :=
by
  sorry

#check smallest_number_of_divisors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_of_divisors_l660_66013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l660_66095

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b) * (a * b + c^2) ≥ 4 * a * b * c ∧
  (a + b + c = 3 → Real.sqrt (a + 1) + Real.sqrt (b + 1) + Real.sqrt (c + 1) ≤ 3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l660_66095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_relation_l660_66011

/-- A geometric sequence with common ratio not equal to 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  hq : q ≠ 1
  is_geometric : ∀ n, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def S (seq : GeometricSequence) (n : ℕ) : ℝ :=
  (seq.a 1) * (1 - seq.q^n) / (1 - seq.q)

theorem geometric_sequence_sum_relation (seq : GeometricSequence) 
  (h : 2 * seq.a 2022 = seq.a 2023 + seq.a 2024) :
  S seq 2024 + S seq 2023 = 2 * S seq 2022 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_relation_l660_66011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l660_66001

/-- The area of a quadrilateral ABCD with specific angle and side length properties -/
theorem quadrilateral_area 
  (A B C D : ℝ) -- Angles of the quadrilateral
  (a b c d : ℝ) -- Side lengths of the quadrilateral
  (h1 : A = Real.pi - C) -- Angle condition
  (h2 : B = Real.pi - D) -- Angle condition
  (h3 : a + c = b + d) -- Side length condition
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) -- Positive side lengths
  : ∃ (T : ℝ), T = Real.sqrt (a * b * c * d) ∧ T^2 = (a * b * c * d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l660_66001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_is_5_sqrt_2_l660_66088

/-- A right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The length of tangent PA -/
  pa : ℝ
  /-- The length of tangent PB -/
  pb : ℝ
  /-- The length of tangent PC -/
  pc : ℝ
  /-- PA and PC are equal -/
  pa_eq_pc : pa = pc
  /-- PA, PB, PC are positive -/
  pa_pos : pa > 0
  pb_pos : pb > 0
  pc_pos : pc > 0

/-- The hypotenuse of the right-angled triangle -/
noncomputable def hypotenuse (t : RightTriangleWithInscribedCircle) : ℝ :=
  Real.sqrt (2 * (t.pa + t.pb)^2)

/-- The main theorem -/
theorem hypotenuse_is_5_sqrt_2 (t : RightTriangleWithInscribedCircle) 
    (h1 : t.pa = 2) (h2 : t.pb = 3) : hypotenuse t = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_is_5_sqrt_2_l660_66088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hand_movements_l660_66017

/-- Represents a standard clock -/
structure Clock where
  total_degrees : ℝ := 360
  minutes_per_hour : ℕ := 60
  hours_on_face : ℕ := 12

/-- Calculates the angular movement of the minute hand per minute -/
noncomputable def minute_hand_movement (c : Clock) : ℝ :=
  c.total_degrees / c.minutes_per_hour

/-- Calculates the angular movement of the hour hand per minute -/
noncomputable def hour_hand_movement (c : Clock) : ℝ :=
  (c.total_degrees / c.hours_on_face) / c.minutes_per_hour

/-- Theorem stating the angular movements of clock hands -/
theorem clock_hand_movements (c : Clock) :
  minute_hand_movement c = 6 ∧ hour_hand_movement c = 0.5 := by
  sorry

#check clock_hand_movements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hand_movements_l660_66017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_greater_sin_l660_66008

theorem cos_greater_sin : Real.cos (3 * π / 14) > Real.sin (-15 * π / 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_greater_sin_l660_66008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_five_l660_66072

/-- The radius of a sphere with surface area equal to the curved surface area of a right circular cylinder -/
noncomputable def sphere_radius (cylinder_height : ℝ) (cylinder_diameter : ℝ) : ℝ :=
  (cylinder_height * cylinder_diameter / 4) ^ (1/2)

/-- Theorem: The radius of the sphere is 5 cm when the cylinder height and diameter are both 10 cm -/
theorem sphere_radius_is_five :
  sphere_radius 10 10 = 5 := by
  -- Unfold the definition of sphere_radius
  unfold sphere_radius
  -- Simplify the expression
  simp [Real.rpow_def_of_pos]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_five_l660_66072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_and_parity_l660_66077

theorem perfect_square_and_parity (a b c : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b)) : 
  (∃ d : ℕ, c = d^2 ∧ Odd c) ∧ ¬(∃ c : ℕ, c > 0 ∧ Even c ∧ c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_and_parity_l660_66077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_cubic_properties_l660_66010

/-- A cubic function with specific root properties -/
structure SpecialCubic where
  b : ℝ
  c : ℝ
  d : ℝ
  f : ℝ → ℝ
  f_def : ∀ x, f x = x^3 + b*x^2 + c*x + d
  one_root : ∀ k, k ∈ (Set.Iic 0 ∪ Set.Ici 4) → (∃! x, f x = k)
  three_roots : ∀ k, k ∈ Set.Ioo 0 4 → (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = k ∧ f y = k ∧ f z = k)

/-- The derivative of the cubic function -/
def SpecialCubic.f' (sc : SpecialCubic) : ℝ → ℝ := 
  λ x ↦ 3*x^2 + 2*sc.b*x + sc.c

theorem special_cubic_properties (sc : SpecialCubic) : 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ sc.f' x₁ = 0 ∧ sc.f' x₂ = 0) ∧ 
  (∃ x, sc.f x = 4 ∧ sc.f' x = 0) ∧
  (∃ x, sc.f x = 0 ∧ sc.f' x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_cubic_properties_l660_66010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l660_66033

theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → m ≤ Real.tan x + 1) → 
  m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l660_66033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_STRQ_l660_66036

-- Define the triangle PQR
def triangle_PQR (P Q R : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  (px - qx)^2 + (py - qy)^2 = (px - rx)^2 + (py - ry)^2 ∧ 
  (qx - rx)^2 + (qy - ry)^2 = 300^2

-- Define points S and T
def point_S (P Q S : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2)

def point_T (P R T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ T = (t * P.1 + (1 - t) * R.1, t * P.2 + (1 - t) * R.2)

-- Define ST perpendicular to PR
def ST_perpendicular_PR (P R S T : ℝ × ℝ) : Prop :=
  (S.1 - T.1) * (P.1 - R.1) + (S.2 - T.2) * (P.2 - R.2) = 0

-- Define the lengths of ST, TR, and QS
def length_ST (S T : ℝ × ℝ) : Prop :=
  (S.1 - T.1)^2 + (S.2 - T.2)^2 = 120^2

def length_TR (T R : ℝ × ℝ) : Prop :=
  (T.1 - R.1)^2 + (T.2 - R.2)^2 = 271^2

def length_QS (Q S : ℝ × ℝ) : Prop :=
  (Q.1 - S.1)^2 + (Q.2 - S.2)^2 = 221^2

-- Define the area of a quadrilateral
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  let area_triangle (X Y Z : ℝ × ℝ) : ℝ :=
    (abs ((Y.1 - X.1) * (Z.2 - X.2) - (Z.1 - X.1) * (Y.2 - X.2))) / 2
  area_triangle A B C + area_triangle A C D

-- Theorem statement
theorem area_STRQ (P Q R S T : ℝ × ℝ) :
  triangle_PQR P Q R →
  point_S P Q S →
  point_T P R T →
  ST_perpendicular_PR P R S T →
  length_ST S T →
  length_TR T R →
  length_QS Q S →
  area_quadrilateral S T R Q = 46860 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_STRQ_l660_66036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_level_surfaces_are_spheres_function_value_at_P_l660_66081

-- Define the function
noncomputable def f (x₁ x₂ x₃ : ℝ) : ℝ := Real.sqrt (36 - x₁^2 - x₂^2 - x₃^2)

-- Define the level surface equation
def level_surface (C : ℝ) (x₁ x₂ x₃ : ℝ) : Prop :=
  x₁^2 + x₂^2 + x₃^2 = 36 - C^2

-- Theorem for level surfaces
theorem level_surfaces_are_spheres (C : ℝ) (h : 0 ≤ C ∧ C ≤ 6) :
  ∀ x₁ x₂ x₃, f x₁ x₂ x₃ = C ↔ level_surface C x₁ x₂ x₃ := by
  sorry

-- Theorem for function value at P(1, 1, 3)
theorem function_value_at_P : f 1 1 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_level_surfaces_are_spheres_function_value_at_P_l660_66081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rhombus_distinction_l660_66002

/-- A shape with four sides -/
class Quadrilateral :=
  (sides : Fin 4 → ℝ)
  (angles : Fin 4 → ℝ)
  (diagonals : Fin 2 → ℝ)

/-- A rectangle is a quadrilateral with specific properties -/
class Rectangle extends Quadrilateral :=
  (opposite_sides_equal : ∀ i : Fin 2, sides i = sides (i + 2))
  (opposite_sides_parallel : True)  -- We can't easily represent this in this simple model
  (right_angles : ∀ i : Fin 4, angles i = π / 2)
  (diagonals_bisect : True)  -- We can't easily represent this in this simple model
  (diagonals_equal : diagonals 0 = diagonals 1)

/-- A rhombus is a quadrilateral with specific properties -/
class Rhombus extends Quadrilateral :=
  (all_sides_equal : ∀ i j : Fin 4, sides i = sides j)
  (opposite_sides_parallel : True)  -- We can't easily represent this in this simple model
  (diagonals_bisect : True)  -- We can't easily represent this in this simple model
  (diagonals_perpendicular : True)  -- We can't easily represent this in this simple model
  (diagonals_bisect_angles : True)  -- We can't easily represent this in this simple model

/-- The theorem stating that equal diagonals distinguish rectangles from rhombuses -/
theorem rectangle_rhombus_distinction :
  ∃ (r : Rectangle) (h : Rhombus), r.diagonals 0 = r.diagonals 1 ∧ h.diagonals 0 ≠ h.diagonals 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rhombus_distinction_l660_66002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_one_l660_66058

/-- Represents a configuration of circles as described in the problem -/
structure CircleConfiguration where
  R : ℝ  -- Radius of the larger circle
  small_circle_area : ℝ  -- Area of one smaller circle

/-- The properties of the circle configuration -/
def circle_properties (c : CircleConfiguration) : Prop :=
  c.small_circle_area = Real.pi * (c.R / 2)^2 ∧  -- Area of smaller circle
  2 * c.small_circle_area = 1  -- Combined area of two smaller circles is 1

/-- The theorem to be proved -/
theorem shaded_area_equals_one (c : CircleConfiguration) 
  (h : circle_properties c) : 
  Real.pi * c.R^2 - 2 * c.small_circle_area = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_one_l660_66058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l660_66035

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the area function
noncomputable def area (t : Triangle) : ℝ := 
  1/2 * t.a * t.b * Real.sin t.C

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.c * Real.cos t.B = (3 * t.a - t.b) * Real.cos t.C)
  (h2 : t.c = 2 * Real.sqrt 6)
  (h3 : t.b - t.a = 2) : 
  Real.sin t.C = 2 * Real.sqrt 2 / 3 ∧ 
  area t = 5 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l660_66035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_square_roots_l660_66086

theorem simplify_square_roots :
  (Real.sqrt 12 = 2 * Real.sqrt 3) ∧ (Real.sqrt (1/2) = Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_square_roots_l660_66086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_l660_66044

/-- The area of wrapping paper needed for a box -/
theorem wrapping_paper_area (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  2 * (w + h)^2 = 8 * w * h :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_l660_66044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_property_l660_66030

theorem largest_integer_property (n : ℕ) (hn : Odd n) :
  ∃ (r : ℕ), r = Int.ceil (n / 2 : ℚ) ∧
  (∀ (a : Fin n → ℝ) (α : ℝ) (hα : Irrational α),
    ∃ (i : Fin r → Fin n),
      StrictMono i ∧
      (∀ (k l : Fin r), a (i k) - a (i l) ≠ 1 ∧ a (i k) - a (i l) ≠ α)) ∧
  (∀ (r' : ℕ), r' > r →
    ∃ (a : Fin n → ℝ) (α : ℝ) (hα : Irrational α),
      ∀ (i : Fin r' → Fin n),
        StrictMono i →
        ∃ (k l : Fin r'), a (i k) - a (i l) = 1 ∨ a (i k) - a (i l) = α) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_property_l660_66030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_with_inclination_l660_66012

/-- A line passing through the point (2, 4) with an angle of inclination of 45° has the equation x - y + 2 = 0 -/
theorem line_equation_through_point_with_inclination :
  ∀ (l : Set (ℝ × ℝ)),
  ((2, 4) ∈ l) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = 1) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x - y + 2 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_with_inclination_l660_66012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_exist_l660_66061

-- Define the fixed point F
noncomputable def F : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the fixed line
def fixed_line (x : ℝ) : Prop := x = 4 * Real.sqrt 3 / 3

-- Define the ratio of distances
noncomputable def distance_ratio (M : ℝ × ℝ) : Prop :=
  (Real.sqrt ((M.1 - F.1)^2 + M.2^2)) / |M.1 - 4 * Real.sqrt 3 / 3| = Real.sqrt 3 / 2

-- Define the curve C
def curve_C (x y : ℝ) : Prop := 9 * y^2 + 8 * Real.sqrt 3 * x - 36 = 0

-- Define the point P
noncomputable def P : ℝ × ℝ := (2 * Real.sqrt 3 / 3, 0)

-- Theorem statement
theorem equal_angles_exist :
  ∀ (A B : ℝ × ℝ),
  curve_C A.1 A.2 →
  curve_C B.1 B.2 →
  ∃ (k : ℝ),
  k ≠ 0 ∧
  (A.1 - F.1) = k * (A.2 - F.2) ∧
  (B.1 - F.1) = k * (B.2 - F.2) →
  (A.2 - P.2) / (A.1 - P.1) + (B.2 - P.2) / (B.1 - P.1) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_exist_l660_66061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_or_white_l660_66070

/-- The probability of selecting a red or white marble from a bag -/
theorem prob_red_or_white (total : ℕ) (blue : ℕ) (red : ℕ) (h_total : total = 20) (h_blue : blue = 6) (h_red : red = 9) :
  let white : ℕ := total - blue - red
  (red + white : ℚ) / total = 7 / 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_or_white_l660_66070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_irrational_l660_66006

-- Define the numbers
noncomputable def a : ℝ := 3.1415926
noncomputable def b : ℝ := -Real.rpow 9 (1/3)
noncomputable def c : ℝ := -Real.sqrt 0.25
noncomputable def d : ℝ := -9/13

-- State the theorem
theorem b_is_irrational :
  (∃ (q : ℚ), (q : ℝ) = a) ∧
  (∃ (q : ℚ), (q : ℝ) = c) ∧
  (∃ (q : ℚ), (q : ℝ) = d) →
  ¬∃ (q : ℚ), (q : ℝ) = b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_irrational_l660_66006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_k_range_l660_66022

-- Define the function f
noncomputable def f (k x : ℝ) : ℝ := (k * x) / (x^2 + 3 * k)

-- Define the solution set condition
def solution_set_condition (k m : ℝ) : Prop :=
  ∀ x, f k x > m ↔ x < -3 ∨ x > -2

-- Theorem 1
theorem solution_set_equivalence (k m : ℝ) (h : k > 0) (h_sol : solution_set_condition k m) :
  ∀ x, 5 * m * x^2 + (k / 2) * x + 3 > 0 ↔ -1 < x ∧ x < 3/2 :=
by sorry

-- Theorem 2
theorem k_range (k : ℝ) (h : k > 0) :
  (∃ x > 3, f k x > 1) → k > 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_k_range_l660_66022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_l660_66016

noncomputable section

theorem isosceles_right_triangle 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = Real.pi)
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h4 : a = b * Real.sin A / Real.sin B)
  (h5 : a = c * Real.sin A / Real.sin C)
  (h6 : c / Real.sin B + b / Real.sin C = 2 * a) :
  A = Real.pi / 2 ∧ b = c :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_l660_66016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_theorem_l660_66048

open Real

theorem triangle_abc_theorem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a * sin C = b * sin A ∧
  b * sin C = c * sin B ∧
  c * sin A = a * sin B ∧
  Real.sqrt 3 * a * sin B + b * cos A = c →
  (B = π/6 ∧
   (a = 2 * Real.sqrt 3 * c ∧ 
    1/2 * a * c * sin B = 2 * Real.sqrt 3 → 
    b = 6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_theorem_l660_66048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courses_taken_previous_year_l660_66063

theorem courses_taken_previous_year :
  ∃ (courses_previous_year : ℕ),
    let courses_last_year : ℕ := 6
    let avg_last_year : ℚ := 100
    let avg_previous_year : ℚ := 50
    let avg_two_years : ℚ := 77
    (courses_last_year * avg_last_year + courses_previous_year * avg_previous_year) / 
    (courses_last_year + courses_previous_year : ℚ) = avg_two_years ∧
    courses_previous_year = 5
  := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_courses_taken_previous_year_l660_66063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l660_66042

noncomputable def f (x : ℝ) : ℝ := 2 / (x + 3)

theorem function_properties :
  -- 1. The graph intersects with at least one coordinate axis
  (∃ x : ℝ, f x = 0 ∨ ∃ y : ℝ, f 0 = y) ∧
  -- 2. The graph is symmetric with respect to the line y = x + 3
  (∀ x y : ℝ, f x = y ↔ f (y - 3) = x + 3) ∧
  -- 3. The graph is symmetric with respect to the line y = -x - 3
  (∀ x y : ℝ, f x = y ↔ f (-y - 3) = -x - 3) ∧
  -- 4. The graph is symmetric with respect to the point (-3, 0)
  (∀ x y : ℝ, f x = y ↔ f (-x - 6) = -y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l660_66042
