import Mathlib

namespace NUMINAMATH_CALUDE_division_problem_l1295_129546

theorem division_problem :
  let dividend : Nat := 73648
  let divisor : Nat := 874
  let quotient : Nat := dividend / divisor
  let remainder : Nat := dividend % divisor
  (quotient = 84) ∧ 
  (remainder = 232) ∧ 
  (remainder + 375 = 607) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1295_129546


namespace NUMINAMATH_CALUDE_laundry_dry_cycle_time_l1295_129504

theorem laundry_dry_cycle_time 
  (total_loads : ℕ) 
  (wash_time_per_load : ℚ) 
  (total_time : ℚ) 
  (h1 : total_loads = 8) 
  (h2 : wash_time_per_load = 45 / 60) 
  (h3 : total_time = 14) : 
  (total_time - (total_loads : ℚ) * wash_time_per_load) / total_loads = 1 := by
  sorry

end NUMINAMATH_CALUDE_laundry_dry_cycle_time_l1295_129504


namespace NUMINAMATH_CALUDE_expression_evaluation_l1295_129528

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator := 4 + x * (2 + x) - 2^2
  let denominator := x - 2 + x^2
  numerator / denominator = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1295_129528


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1295_129538

theorem quadratic_inequality (x : ℝ) : x^2 - 50*x + 625 ≤ 25 ↔ 20 ≤ x ∧ x ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1295_129538


namespace NUMINAMATH_CALUDE_new_crew_weight_l1295_129521

/-- The combined weight of two new crew members in a sailboat scenario -/
theorem new_crew_weight (n : ℕ) (avg_increase w1 w2 : ℝ) : 
  n = 12 → 
  avg_increase = 2.2 →
  w1 = 78 →
  w2 = 65 →
  (n : ℝ) * avg_increase + w1 + w2 = 169.4 :=
by sorry

end NUMINAMATH_CALUDE_new_crew_weight_l1295_129521


namespace NUMINAMATH_CALUDE_series_sum_l1295_129586

open Real

/-- The sum of the series ∑_{n=1}^{∞} (sin^n x) / n for x ≠ π/2 + 2πk, where k is an integer -/
theorem series_sum (x : ℝ) (h : ∀ k : ℤ, x ≠ π / 2 + 2 * π * k) :
  ∑' n, (sin x) ^ n / n = -log (1 - sin x) :=
by sorry


end NUMINAMATH_CALUDE_series_sum_l1295_129586


namespace NUMINAMATH_CALUDE_annika_hiking_time_l1295_129520

/-- Annika's hiking problem -/
theorem annika_hiking_time (rate : ℝ) (initial_distance : ℝ) (total_distance : ℝ) : 
  rate = 12 →
  initial_distance = 2.75 →
  total_distance = 3.5 →
  (total_distance - initial_distance) * rate + total_distance * rate = 51 := by
sorry

end NUMINAMATH_CALUDE_annika_hiking_time_l1295_129520


namespace NUMINAMATH_CALUDE_statement_d_not_always_true_l1295_129553

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the given conditions
variable (m n : Line)
variable (α β : Plane)
variable (h1 : ¬ parallel m n)
variable (h2 : α ≠ β)

-- State the theorem
theorem statement_d_not_always_true :
  ¬ (∀ (m : Line) (α β : Plane),
    plane_perpendicular α β →
    contained_in m α →
    perpendicular m β) :=
by sorry

end NUMINAMATH_CALUDE_statement_d_not_always_true_l1295_129553


namespace NUMINAMATH_CALUDE_milan_phone_bill_l1295_129572

/-- Calculates the number of minutes billed given the total bill, monthly fee, and per-minute rate. -/
def minutes_billed (total_bill monthly_fee per_minute_rate : ℚ) : ℚ :=
  (total_bill - monthly_fee) / per_minute_rate

/-- Proves that given the specified conditions, the number of minutes billed is 178. -/
theorem milan_phone_bill : 
  let total_bill : ℚ := 23.36
  let monthly_fee : ℚ := 2
  let per_minute_rate : ℚ := 0.12
  minutes_billed total_bill monthly_fee per_minute_rate = 178 := by
  sorry

end NUMINAMATH_CALUDE_milan_phone_bill_l1295_129572


namespace NUMINAMATH_CALUDE_centroid_line_intersection_l1295_129592

/-- Given a triangle ABC with centroid G, and a line through G intersecting AB at M and AC at N,
    where AM = x * AB and AN = y * AC, prove that 1/x + 1/y = 3 -/
theorem centroid_line_intersection (A B C G M N : ℝ × ℝ) (x y : ℝ) :
  (G = (1/3 : ℝ) • (A + B + C)) →  -- G is the centroid
  (∃ (t : ℝ), M = A + t • (G - A) ∧ N = A + t • (G - A)) →  -- M and N are on the line through G
  (M = A + x • (B - A)) →  -- AM = x * AB
  (N = A + y • (C - A)) →  -- AN = y * AC
  (1 / x + 1 / y = 3) :=
by sorry

end NUMINAMATH_CALUDE_centroid_line_intersection_l1295_129592


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l1295_129562

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define points A, B, and M
def A : ℝ × ℝ := (-2, 2)
def B : ℝ × ℝ := (-5, 5)
def M : ℝ × ℝ := (-2, 9)

-- Define the line l: x + y + 3 = 0
def l (p : ℝ × ℝ) : Prop := p.1 + p.2 + 3 = 0

-- Theorem statement
theorem circle_and_tangent_lines :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    -- C lies on line l
    l C ∧
    -- Circle passes through A and B
    A ∈ Circle C r ∧ B ∈ Circle C r ∧
    -- Standard equation of the circle
    (∀ (x y : ℝ), (x, y) ∈ Circle C r ↔ (x + 5)^2 + (y - 2)^2 = 9) ∧
    -- Tangent lines through M
    (∀ (x y : ℝ),
      ((x = -2) ∨ (20 * x - 21 * y + 229 = 0)) ↔
      ((x, y) ∈ Circle C r → (x - M.1) * (x - C.1) + (y - M.2) * (y - C.2) = 0)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l1295_129562


namespace NUMINAMATH_CALUDE_power_difference_l1295_129507

theorem power_difference (a m n : ℝ) (hm : a^m = 9) (hn : a^n = 3) :
  a^(m - n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l1295_129507


namespace NUMINAMATH_CALUDE_min_sum_m_n_l1295_129561

theorem min_sum_m_n (m n : ℕ+) (h : 300 * m = n^3) : 
  ∀ (m' n' : ℕ+), 300 * m' = n'^3 → m + n ≤ m' + n' :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l1295_129561


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1295_129539

theorem system_of_equations_solution :
  ∃! (x y : ℚ), (5 * x - 3 * y = -7) ∧ (2 * x + 7 * y = -26) ∧ 
  (x = -127 / 41) ∧ (y = -116 / 41) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1295_129539


namespace NUMINAMATH_CALUDE_no_four_digit_sum_9_div_11_l1295_129537

/-- Represents a four-digit number -/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Theorem: There are no four-digit numbers whose digits add up to 9 and are divisible by 11 -/
theorem no_four_digit_sum_9_div_11 :
  ¬∃ (n : FourDigitNumber), sumOfDigits n.value = 9 ∧ n.value % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_four_digit_sum_9_div_11_l1295_129537


namespace NUMINAMATH_CALUDE_total_full_spots_l1295_129547

/-- Calculates the number of full parking spots in a multi-story parking garage -/
def fullParkingSpots : ℕ :=
  let totalLevels : ℕ := 7
  let firstLevelSpots : ℕ := 100
  let spotIncrease : ℕ := 50
  let firstLevelOpenSpots : ℕ := 58
  let openSpotDecrease : ℕ := 3
  let openSpotIncrease : ℕ := 10
  let switchLevel : ℕ := 4

  let totalFullSpots : ℕ := (List.range totalLevels).foldl
    (fun acc level =>
      let totalSpots := firstLevelSpots + level * spotIncrease
      let openSpots := if level < switchLevel - 1
        then firstLevelOpenSpots - level * openSpotDecrease
        else firstLevelOpenSpots - (switchLevel - 1) * openSpotDecrease + (level - switchLevel + 1) * openSpotIncrease
      acc + (totalSpots - openSpots))
    0

  totalFullSpots

/-- The theorem stating that the total number of full parking spots is 1329 -/
theorem total_full_spots : fullParkingSpots = 1329 := by
  sorry

end NUMINAMATH_CALUDE_total_full_spots_l1295_129547


namespace NUMINAMATH_CALUDE_trigonometric_problem_l1295_129599

theorem trigonometric_problem (α β : Real) 
  (h1 : 2 * Real.sin α = 2 * Real.sin (α / 2) ^ 2 - 1)
  (h2 : α ∈ Set.Ioo 0 Real.pi)
  (h3 : β ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h4 : 3 * Real.tan β ^ 2 - 2 * Real.tan β = 1) :
  (Real.sin (2 * α) + Real.cos (2 * α) = -1/5) ∧ 
  (α + β = 7 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l1295_129599


namespace NUMINAMATH_CALUDE_platform_length_l1295_129552

/-- Calculates the length of a platform given train parameters --/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 200 →
  train_speed_kmph = 54 →
  crossing_time = 25 →
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_speed_mps * crossing_time
  let platform_length := total_distance - train_length
  platform_length = 175 := by sorry

end NUMINAMATH_CALUDE_platform_length_l1295_129552


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1295_129535

theorem opposite_of_2023 : -(2023 : ℝ) = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1295_129535


namespace NUMINAMATH_CALUDE_compute_expression_l1295_129512

theorem compute_expression : (75 * 2424 + 25 * 2424) / 2 = 121200 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1295_129512


namespace NUMINAMATH_CALUDE_median_and_mode_are_23_l1295_129518

/-- Represents the shoe size distribution of a class --/
structure ShoeSizeDistribution where
  sizes : List Nat
  frequencies : List Nat
  total_students : Nat

/-- Calculates the median of a shoe size distribution --/
def median (dist : ShoeSizeDistribution) : Nat :=
  sorry

/-- Calculates the mode of a shoe size distribution --/
def mode (dist : ShoeSizeDistribution) : Nat :=
  sorry

/-- The shoe size distribution for the given class --/
def class_distribution : ShoeSizeDistribution :=
  { sizes := [20, 21, 22, 23, 24],
    frequencies := [2, 8, 9, 19, 2],
    total_students := 40 }

theorem median_and_mode_are_23 :
  median class_distribution = 23 ∧ mode class_distribution = 23 := by
  sorry

end NUMINAMATH_CALUDE_median_and_mode_are_23_l1295_129518


namespace NUMINAMATH_CALUDE_negation_equivalence_l1295_129550

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, (x₀ + 1 < 0) ∨ (x₀^2 - x₀ > 0)) ↔ 
  (∀ x : ℝ, (x + 1 ≥ 0) ∧ (x^2 - x ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1295_129550


namespace NUMINAMATH_CALUDE_sum_of_five_integers_l1295_129542

theorem sum_of_five_integers (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (4 - a) * (4 - b) * (4 - c) * (4 - d) * (4 - e) = 12 →
  a + b + c + d + e = 17 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_five_integers_l1295_129542


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l1295_129522

theorem inequality_not_always_true (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ¬ (∀ b, c * b^2 < a * b^2) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l1295_129522


namespace NUMINAMATH_CALUDE_industrial_machine_shirts_l1295_129566

/-- The number of shirts made by an industrial machine yesterday -/
def shirts_yesterday (x : ℕ) : Prop :=
  let shirts_per_minute : ℕ := 8
  let total_minutes : ℕ := 2
  let shirts_today : ℕ := 3
  let total_shirts : ℕ := shirts_per_minute * total_minutes
  x = total_shirts - shirts_today

theorem industrial_machine_shirts : shirts_yesterday 13 := by
  sorry

end NUMINAMATH_CALUDE_industrial_machine_shirts_l1295_129566


namespace NUMINAMATH_CALUDE_print_shop_charge_difference_l1295_129573

/-- The cost per color copy at print shop X -/
def cost_x : ℚ := 1.20

/-- The cost per color copy at print shop Y -/
def cost_y : ℚ := 1.70

/-- The number of color copies -/
def num_copies : ℕ := 70

/-- The difference in charge between print shop Y and print shop X for a given number of copies -/
def charge_difference (n : ℕ) : ℚ := n * (cost_y - cost_x)

theorem print_shop_charge_difference : 
  charge_difference num_copies = 35 := by sorry

end NUMINAMATH_CALUDE_print_shop_charge_difference_l1295_129573


namespace NUMINAMATH_CALUDE_social_gathering_handshakes_l1295_129559

/-- Represents a social gathering with specific conditions -/
structure SocialGathering where
  total_people : Nat
  group_a_size : Nat
  group_b_size : Nat
  group_b_connected : Nat
  group_b_isolated : Nat
  a_to_b_connections : Nat
  a_to_b_per_person : Nat

/-- Calculates the number of handshakes in the social gathering -/
def count_handshakes (g : SocialGathering) : Nat :=
  let a_to_b_handshakes := (g.group_a_size - g.a_to_b_connections) * g.group_b_size
  let b_connected_handshakes := g.group_b_connected * (g.group_b_connected - 1) / 2
  let b_isolated_handshakes := g.group_b_isolated * (g.group_b_isolated - 1) / 2 + g.group_b_isolated * g.group_b_connected
  a_to_b_handshakes + b_connected_handshakes + b_isolated_handshakes

/-- The main theorem stating the number of handshakes in the given social gathering -/
theorem social_gathering_handshakes :
  let g : SocialGathering := {
    total_people := 30,
    group_a_size := 15,
    group_b_size := 15,
    group_b_connected := 10,
    group_b_isolated := 5,
    a_to_b_connections := 5,
    a_to_b_per_person := 3
  }
  count_handshakes g = 255 := by
  sorry


end NUMINAMATH_CALUDE_social_gathering_handshakes_l1295_129559


namespace NUMINAMATH_CALUDE_even_implies_symmetric_at_most_one_intersection_l1295_129583

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define symmetry about a point
def symmetric_about (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- Theorem 1: If f(x+1) is even, then f(x) is symmetric about x = 1
theorem even_implies_symmetric :
  is_even (fun x ↦ f (x + 1)) → symmetric_about f 1 := by sorry

-- Theorem 2: Any function has at most one intersection with a vertical line
theorem at_most_one_intersection (a : ℝ) :
  ∃! y, f y = a := by sorry

end NUMINAMATH_CALUDE_even_implies_symmetric_at_most_one_intersection_l1295_129583


namespace NUMINAMATH_CALUDE_four_digit_int_solution_l1295_129510

/-- Represents a four-digit positive integer -/
structure FourDigitInt where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_pos : 0 < a
  a_lt_10 : a < 10
  b_lt_10 : b < 10
  c_lt_10 : c < 10
  d_lt_10 : d < 10

/-- Converts a FourDigitInt to a natural number -/
def FourDigitInt.toNat (n : FourDigitInt) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

theorem four_digit_int_solution :
  ∃! (n : FourDigitInt),
    n.a + n.b + n.c + n.d = 16 ∧
    n.b + n.c = 10 ∧
    n.a - n.d = 2 ∧
    n.toNat % 11 = 0 ∧
    n.toNat = 4642 := by
  sorry

#check four_digit_int_solution

end NUMINAMATH_CALUDE_four_digit_int_solution_l1295_129510


namespace NUMINAMATH_CALUDE_trig_identity_l1295_129588

-- Statement of the trigonometric identity
theorem trig_identity (α β γ : ℝ) :
  Real.sin α + Real.sin β + Real.sin γ = 4 * Real.cos (α/2) * Real.cos (β/2) * Real.cos (γ/2) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1295_129588


namespace NUMINAMATH_CALUDE_smallest_n_for_square_and_cube_l1295_129502

theorem smallest_n_for_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 7 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 → 
    (∃ (y : ℕ), 5 * x = y^2) → 
    (∃ (z : ℕ), 7 * x = z^3) → 
    x ≥ 245) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_square_and_cube_l1295_129502


namespace NUMINAMATH_CALUDE_tower_surface_area_calculation_l1295_129558

def cube_surface_area (s : ℕ) : ℕ := 6 * s^2

def tower_surface_area (edge_lengths : List ℕ) : ℕ :=
  let n := edge_lengths.length
  edge_lengths.enum.foldl (fun acc (i, s) => 
    if i = 0 
    then acc + cube_surface_area s
    else acc + cube_surface_area s - s^2
  ) 0

theorem tower_surface_area_calculation :
  tower_surface_area [4, 5, 6, 7, 8, 9, 10] = 1871 :=
sorry

end NUMINAMATH_CALUDE_tower_surface_area_calculation_l1295_129558


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1295_129500

theorem rectangular_box_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : b * c = 18) 
  (h3 : c * a = 10) : 
  a * b * c = 30 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1295_129500


namespace NUMINAMATH_CALUDE_z_ninth_power_l1295_129560

theorem z_ninth_power (z : ℂ) : z = (-Real.sqrt 3 + Complex.I) / 2 → z^9 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_z_ninth_power_l1295_129560


namespace NUMINAMATH_CALUDE_real_roots_iff_k_geq_quarter_l1295_129513

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ :=
  (k - 1)^2 * x^2 + (2*k + 1) * x + 1

-- Theorem statement
theorem real_roots_iff_k_geq_quarter :
  ∀ k : ℝ, (∃ x : ℝ, quadratic_equation k x = 0) ↔ k ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_iff_k_geq_quarter_l1295_129513


namespace NUMINAMATH_CALUDE_smallest_argument_in_circle_l1295_129516

theorem smallest_argument_in_circle (p : ℂ) : 
  (Complex.abs (p - 25 * Complex.I) ≤ 15) →
  Complex.arg p ≥ Complex.arg (12 + 16 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_smallest_argument_in_circle_l1295_129516


namespace NUMINAMATH_CALUDE_problem_solution_l1295_129581

theorem problem_solution (a b c : ℝ) 
  (h1 : (6 * a + 34) ^ (1/3 : ℝ) = 4)
  (h2 : (5 * a + b - 2) ^ (1/2 : ℝ) = 5)
  (h3 : c = 9 ^ (1/2 : ℝ)) :
  a = 5 ∧ b = 2 ∧ c = 3 ∧ (3 * a - b + c) ^ (1/2 : ℝ) = 4 ∨ (3 * a - b + c) ^ (1/2 : ℝ) = -4 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1295_129581


namespace NUMINAMATH_CALUDE_prime_divides_square_minus_prime_l1295_129577

theorem prime_divides_square_minus_prime (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ (q : ℕ) (n : ℕ+), q.Prime ∧ q < p ∧ p ∣ n.val^2 - q := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_square_minus_prime_l1295_129577


namespace NUMINAMATH_CALUDE_sum_of_ages_l1295_129594

/-- Given the ages and relationships of Beckett, Olaf, Shannen, and Jack, prove that the sum of their ages is 71 years. -/
theorem sum_of_ages (beckett olaf shannen jack : ℕ) : 
  beckett = 12 ∧ 
  olaf = beckett + 3 ∧ 
  shannen = olaf - 2 ∧ 
  jack = 2 * shannen + 5 → 
  beckett + olaf + shannen + jack = 71 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1295_129594


namespace NUMINAMATH_CALUDE_triangle_passing_theorem_l1295_129567

/-- A triangle represented by its side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- Whether a triangle can pass through another triangle -/
def can_pass_through (t1 t2 : Triangle) : Prop := sorry

theorem triangle_passing_theorem (T Q : Triangle) 
  (h_T_area : Triangle.area T < 4)
  (h_Q_area : Triangle.area Q = 3) :
  can_pass_through T Q := by sorry

end NUMINAMATH_CALUDE_triangle_passing_theorem_l1295_129567


namespace NUMINAMATH_CALUDE_option_A_correct_option_C_correct_l1295_129582

-- Define the set M
def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

-- Define the set B
def B : Set ℤ := {b | ∃ n : ℕ, b = 2*n + 1}

-- Theorem for option A
theorem option_A_correct : ∀ a₁ a₂ : ℤ, a₁ ∈ M → a₂ ∈ M → (a₁ * a₂) ∈ M := by
  sorry

-- Theorem for option C
theorem option_C_correct : B ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_option_A_correct_option_C_correct_l1295_129582


namespace NUMINAMATH_CALUDE_amy_bob_games_l1295_129549

/-- Represents the total number of players -/
def total_players : ℕ := 12

/-- Represents the number of players in each game -/
def players_per_game : ℕ := 6

/-- Represents the number of players that are always together (Chris and Dave) -/
def always_together : ℕ := 2

/-- Represents the number of specific players we're interested in (Amy and Bob) -/
def specific_players : ℕ := 2

/-- Theorem stating that the number of games where Amy and Bob play together
    is equal to the number of ways to choose 2 players from the remaining 8 players -/
theorem amy_bob_games :
  (total_players - specific_players - always_together).choose 2 =
  Nat.choose 8 2 := by sorry

end NUMINAMATH_CALUDE_amy_bob_games_l1295_129549


namespace NUMINAMATH_CALUDE_sin_to_cos_transformation_l1295_129555

theorem sin_to_cos_transformation (x : ℝ) :
  Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4) =
  Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_to_cos_transformation_l1295_129555


namespace NUMINAMATH_CALUDE_zhonghuan_cup_exam_l1295_129554

theorem zhonghuan_cup_exam (total : ℕ) (english : ℕ) (chinese : ℕ) (both : ℕ) 
  (h1 : total = 45)
  (h2 : english = 35)
  (h3 : chinese = 31)
  (h4 : both = 24) :
  total - (english + chinese - both) = 3 := by
  sorry

end NUMINAMATH_CALUDE_zhonghuan_cup_exam_l1295_129554


namespace NUMINAMATH_CALUDE_min_a_for_inequality_l1295_129509

theorem min_a_for_inequality (a : ℝ) : 
  (∀ x > a, 2 * x + 3 ≥ 7) ↔ a < 2 := by sorry

end NUMINAMATH_CALUDE_min_a_for_inequality_l1295_129509


namespace NUMINAMATH_CALUDE_minimum_point_l1295_129587

-- Define the function
def f (x : ℝ) : ℝ := 2 * |x - 4| - 2

-- State the theorem
theorem minimum_point :
  ∃! p : ℝ × ℝ, p.1 = 4 ∧ p.2 = -2 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
by sorry

end NUMINAMATH_CALUDE_minimum_point_l1295_129587


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1295_129557

theorem geometric_sequence_first_term (a b c : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ 16 = b * r ∧ c = 16 * r ∧ 128 = c * r) →
  a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1295_129557


namespace NUMINAMATH_CALUDE_catrionas_fish_count_catrionas_aquarium_l1295_129571

theorem catrionas_fish_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun goldfish angelfish guppies total =>
    (goldfish = 8) →
    (angelfish = goldfish + 4) →
    (guppies = 2 * angelfish) →
    (total = goldfish + angelfish + guppies) →
    (total = 44)

-- Proof
theorem catrionas_aquarium : catrionas_fish_count 8 12 24 44 := by
  sorry

end NUMINAMATH_CALUDE_catrionas_fish_count_catrionas_aquarium_l1295_129571


namespace NUMINAMATH_CALUDE_math_correct_percentage_l1295_129544

/-- Represents the number of questions in the math test -/
def math_questions : ℕ := 40

/-- Represents the number of questions in the English test -/
def english_questions : ℕ := 50

/-- Represents the percentage of English questions answered correctly -/
def english_correct_percentage : ℚ := 98 / 100

/-- Represents the total number of questions answered correctly across both tests -/
def total_correct : ℕ := 79

/-- Theorem stating that the percentage of math questions answered correctly is 75% -/
theorem math_correct_percentage :
  (total_correct - (english_correct_percentage * english_questions).num) / math_questions = 75 / 100 := by
  sorry

end NUMINAMATH_CALUDE_math_correct_percentage_l1295_129544


namespace NUMINAMATH_CALUDE_pairing_fraction_l1295_129570

theorem pairing_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) :
  n = 4 * s / 3 →
  (s / 3 + n / 4) / (s + n) = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_pairing_fraction_l1295_129570


namespace NUMINAMATH_CALUDE_system_solution_l1295_129576

theorem system_solution (a b : ℝ) : 
  (2 * 5 + b = a) ∧ (5 - 2 * b = 3) → a = 11 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_system_solution_l1295_129576


namespace NUMINAMATH_CALUDE_fibonacci_ratio_property_fibonacci_ratio_periodic_fibonacci_ratio_distinct_in_period_l1295_129514

/-- Fibonacci sequence -/
def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

/-- Ratio of consecutive Fibonacci numbers -/
def fibonacciRatio (n : ℕ) : ℚ :=
  if n = 0 then 0 else (fibonacci (n + 1) : ℚ) / (fibonacci n : ℚ)

theorem fibonacci_ratio_property (n : ℕ) (h : n > 1) :
  fibonacciRatio n = 1 + 1 / (fibonacciRatio (n - 1)) :=
sorry

theorem fibonacci_ratio_periodic :
  ∃ (p : ℕ) (h : p > 0), ∀ (n : ℕ), fibonacciRatio (n + p) = fibonacciRatio n :=
sorry

theorem fibonacci_ratio_distinct_in_period (p : ℕ) (h : p > 0) :
  ∀ (i j : ℕ), i < j → j < p → fibonacciRatio i ≠ fibonacciRatio j :=
sorry

end NUMINAMATH_CALUDE_fibonacci_ratio_property_fibonacci_ratio_periodic_fibonacci_ratio_distinct_in_period_l1295_129514


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_one_thirty_eight_satisfies_conditions_one_thirty_eight_is_greatest_main_result_l1295_129505

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 138 :=
by sorry

theorem one_thirty_eight_satisfies_conditions : 138 < 150 ∧ Nat.gcd 138 18 = 6 :=
by sorry

theorem one_thirty_eight_is_greatest : 
  ∀ m : ℕ, m < 150 ∧ Nat.gcd m 18 = 6 → m ≤ 138 :=
by sorry

theorem main_result : 
  (∃ n : ℕ, n < 150 ∧ Nat.gcd n 18 = 6 ∧ 
    ∀ m : ℕ, m < 150 ∧ Nat.gcd m 18 = 6 → m ≤ n) ∧
  (∀ n : ℕ, n < 150 ∧ Nat.gcd n 18 = 6 ∧ 
    (∀ m : ℕ, m < 150 ∧ Nat.gcd m 18 = 6 → m ≤ n) → n = 138) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_one_thirty_eight_satisfies_conditions_one_thirty_eight_is_greatest_main_result_l1295_129505


namespace NUMINAMATH_CALUDE_problem_solution_l1295_129579

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x - y^2 = 3) (h2 : x^2 + y^4 = 13) : 
  x = (3 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1295_129579


namespace NUMINAMATH_CALUDE_sum_of_digits_square_22222_l1295_129530

/-- The sum of the digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The square of 22222 -/
def square_22222 : ℕ := 22222 * 22222

theorem sum_of_digits_square_22222 : sum_of_digits square_22222 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_22222_l1295_129530


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l1295_129503

open Real

theorem negation_of_existence_proposition :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ 1 + sin x₀ = -x₀^2) ↔
  (∀ x : ℝ, x > 0 → 1 + sin x ≠ -x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l1295_129503


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_inequality_l1295_129536

theorem sqrt_equality_implies_inequality (b : ℝ) : 
  Real.sqrt ((3 - b)^2) = 3 - b → b ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_inequality_l1295_129536


namespace NUMINAMATH_CALUDE_election_votes_l1295_129517

theorem election_votes (total_votes : ℕ) 
  (winning_percentage : ℚ) (vote_majority : ℕ) : 
  winning_percentage = 70 / 100 → 
  vote_majority = 160 → 
  (winning_percentage * total_votes : ℚ) - 
    ((1 - winning_percentage) * total_votes : ℚ) = vote_majority → 
  total_votes = 400 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l1295_129517


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l1295_129574

-- Define the sets M and N
def M : Set ℝ := {x | -4 < x - 1 ∧ x - 1 ≤ 4}
def N : Set ℝ := {x | x^2 < 25}

-- Define the complement of M in ℝ
def C_R_M : Set ℝ := {x | x ∉ M}

-- State the theorem
theorem complement_M_intersect_N :
  (C_R_M ∩ N) = {x : ℝ | -5 < x ∧ x ≤ -3} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l1295_129574


namespace NUMINAMATH_CALUDE_max_value_2x_3y_l1295_129519

theorem max_value_2x_3y (x y : ℝ) (h : 3 * x^2 + y^2 ≤ 3) :
  ∃ (M : ℝ), M = Real.sqrt 31 ∧ 2*x + 3*y ≤ M ∧ ∀ (N : ℝ), (∀ (a b : ℝ), 3 * a^2 + b^2 ≤ 3 → 2*a + 3*b ≤ N) → M ≤ N :=
sorry

end NUMINAMATH_CALUDE_max_value_2x_3y_l1295_129519


namespace NUMINAMATH_CALUDE_disease_cases_1975_l1295_129565

/-- Calculates the number of disease cases in a given year, assuming a linear decrease -/
def diseaseCases (initialYear finalYear : ℕ) (initialCases finalCases : ℕ) (targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let annualDecrease := totalDecrease / totalYears
  let yearsPassed := targetYear - initialYear
  initialCases - (annualDecrease * yearsPassed)

theorem disease_cases_1975 :
  diseaseCases 1950 2000 500000 1000 1975 = 250500 := by
  sorry

end NUMINAMATH_CALUDE_disease_cases_1975_l1295_129565


namespace NUMINAMATH_CALUDE_jenny_meal_combinations_l1295_129527

/-- Represents the number of choices for each meal component -/
structure MealChoices where
  mainDishes : Nat
  drinks : Nat
  desserts : Nat
  sideDishes : Nat

/-- Calculates the total number of possible meal combinations -/
def totalMealCombinations (choices : MealChoices) : Nat :=
  choices.mainDishes * choices.drinks * choices.desserts * choices.sideDishes

/-- Theorem stating that Jenny can arrange 48 distinct possible meals -/
theorem jenny_meal_combinations :
  let jennyChoices : MealChoices := {
    mainDishes := 4,
    drinks := 2,
    desserts := 2,
    sideDishes := 3
  }
  totalMealCombinations jennyChoices = 48 := by
  sorry

end NUMINAMATH_CALUDE_jenny_meal_combinations_l1295_129527


namespace NUMINAMATH_CALUDE_triangle_square_perimeter_l1295_129543

theorem triangle_square_perimeter (d : ℕ) : 
  (∃ (s t : ℝ), 
    s > 0 ∧ 
    3 * t - 4 * s = 2016 ∧ 
    t - s = d) ↔ 
  d > 672 :=
sorry

end NUMINAMATH_CALUDE_triangle_square_perimeter_l1295_129543


namespace NUMINAMATH_CALUDE_square_field_area_l1295_129556

/-- Given a square field with side length s, prove that the area is 27889 square meters
    when the cost of barbed wire at 1.20 per meter for (4s - 2) meters equals 799.20. -/
theorem square_field_area (s : ℝ) : 
  (4 * s - 2) * 1.20 = 799.20 → s^2 = 27889 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l1295_129556


namespace NUMINAMATH_CALUDE_fraction_transformation_l1295_129590

theorem fraction_transformation (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 8 : ℚ) / (d + 8) = (1 : ℚ) / 3 →
  d = 25 := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l1295_129590


namespace NUMINAMATH_CALUDE_friday_zoo_visitors_l1295_129508

/-- The number of people who visited the zoo on Saturday -/
def saturday_visitors : ℕ := 3750

/-- The number of people who visited the zoo on Friday -/
def friday_visitors : ℕ := saturday_visitors / 3

/-- Theorem stating that 1250 people visited the zoo on Friday -/
theorem friday_zoo_visitors : friday_visitors = 1250 := by
  sorry

end NUMINAMATH_CALUDE_friday_zoo_visitors_l1295_129508


namespace NUMINAMATH_CALUDE_parabola_from_hyperbola_l1295_129595

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (positive_a : 0 < a)
  (positive_b : 0 < b)

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  p : ℝ
  (positive_p : 0 < p)

/-- The center of a hyperbola -/
def Hyperbola.center (h : Hyperbola) : ℝ × ℝ := (0, 0)

/-- The right focus of a hyperbola -/
def Hyperbola.right_focus (h : Hyperbola) : ℝ × ℝ := (h.a, 0)

/-- The equation of a parabola with vertex at the origin -/
def Parabola.equation (p : Parabola) (x y : ℝ) : Prop :=
  y^2 = 4 * p.p * x

theorem parabola_from_hyperbola (h : Hyperbola) 
    (h_eq : ∀ x y : ℝ, x^2 / 4 - y^2 / 5 = 1 ↔ (x / h.a)^2 - (y / h.b)^2 = 1) :
    ∃ p : Parabola, 
      p.equation = fun x y => y^2 = 12 * x ∧
      Parabola.equation p x y ↔ y^2 = 12 * x := by
  sorry

end NUMINAMATH_CALUDE_parabola_from_hyperbola_l1295_129595


namespace NUMINAMATH_CALUDE_sugar_calculation_l1295_129591

theorem sugar_calculation (initial_sugar : ℕ) (used_sugar : ℕ) (bought_sugar : ℕ) 
  (h1 : initial_sugar = 65)
  (h2 : used_sugar = 18)
  (h3 : bought_sugar = 50) :
  initial_sugar - used_sugar + bought_sugar = 97 := by
  sorry

end NUMINAMATH_CALUDE_sugar_calculation_l1295_129591


namespace NUMINAMATH_CALUDE_dividend_calculation_l1295_129585

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 15)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 140 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1295_129585


namespace NUMINAMATH_CALUDE_square_root_property_l1295_129533

theorem square_root_property (x : ℝ) : 
  (Real.sqrt (2*x + 3) = 3) → (2*x + 3)^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_root_property_l1295_129533


namespace NUMINAMATH_CALUDE_pizza_slices_remaining_l1295_129578

def initial_slices : ℕ := 15
def breakfast_slices : ℕ := 4
def lunch_slices : ℕ := 2
def snack_slices : ℕ := 2
def dinner_slices : ℕ := 5

theorem pizza_slices_remaining :
  initial_slices - breakfast_slices - lunch_slices - snack_slices - dinner_slices = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_remaining_l1295_129578


namespace NUMINAMATH_CALUDE_triangle_min_angle_le_60_l1295_129534

theorem triangle_min_angle_le_60 (α β γ : ℝ) :
  α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0 → min α (min β γ) ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_min_angle_le_60_l1295_129534


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1295_129529

/-- Given positive real numbers m and n, and perpendicular vectors (m, 1) and (1, n-1),
    the minimum value of 1/m + 2/n is 3 + 2√2. -/
theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) 
    (h_perp : m * 1 + 1 * (n - 1) = 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * 1 + 1 * (y - 1) = 0 → 1/x + 2/y ≥ 1/m + 2/n) →
  1/m + 2/n = 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1295_129529


namespace NUMINAMATH_CALUDE_range_of_a_l1295_129524

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 1 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, Real.exp (2*x) - 2 * Real.exp x + a ≥ 0

-- Define the theorem
theorem range_of_a : 
  ∀ a : ℝ, (p a ∧ q a) → a ∈ Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1295_129524


namespace NUMINAMATH_CALUDE_rectangle_triangle_ef_length_l1295_129532

/-- Given a rectangle ABCD with side lengths AB and BC, and a triangle DEF inside it
    where DE = DF and the area of DEF is one-third of ABCD's area,
    prove that EF has length 12 when AB = 9 and BC = 12. -/
theorem rectangle_triangle_ef_length
  (AB BC : ℝ)
  (DE DF EF : ℝ)
  (h_ab : AB = 9)
  (h_bc : BC = 12)
  (h_de_df : DE = DF)
  (h_area : (1/2) * DE * DF = (1/3) * AB * BC) :
  EF = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_ef_length_l1295_129532


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1295_129526

theorem fraction_multiplication : (2 : ℚ) / 3 * 4 / 7 * 5 / 9 * 11 / 13 = 440 / 2457 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1295_129526


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1295_129511

theorem complex_equation_solution (z : ℂ) : 2 + z = (2 - z) * Complex.I → z = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1295_129511


namespace NUMINAMATH_CALUDE_cost_of_gums_in_dollars_l1295_129563

-- Define the cost of one piece of gum in cents
def cost_of_one_gum : ℕ := 2

-- Define the number of pieces of gum
def number_of_gums : ℕ := 500

-- Define the conversion rate from cents to dollars
def cents_per_dollar : ℕ := 100

-- Theorem to prove
theorem cost_of_gums_in_dollars : 
  (number_of_gums * cost_of_one_gum) / cents_per_dollar = 10 := by
  sorry


end NUMINAMATH_CALUDE_cost_of_gums_in_dollars_l1295_129563


namespace NUMINAMATH_CALUDE_first_seven_primes_sum_mod_eighth_prime_l1295_129506

theorem first_seven_primes_sum_mod_eighth_prime : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17) % 19 = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_seven_primes_sum_mod_eighth_prime_l1295_129506


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l1295_129523

theorem permutation_equation_solution (m : ℕ) : 
  (m * (m - 1) * (m - 2) * (m - 3) * (m - 4) = 2 * m * (m - 1) * (m - 2)) → m = 5 :=
by sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l1295_129523


namespace NUMINAMATH_CALUDE_carol_distance_behind_anna_l1295_129515

/-- Represents the position of a runner in a race -/
structure Position :=
  (distance : ℝ)

/-- Represents a runner in the race -/
structure Runner :=
  (speed : ℝ)
  (position : Position)

/-- The race setup -/
structure Race :=
  (length : ℝ)
  (anna : Runner)
  (bridgit : Runner)
  (carol : Runner)

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  r.length = 100 ∧
  r.anna.speed > 0 ∧
  r.bridgit.speed > 0 ∧
  r.carol.speed > 0 ∧
  r.anna.speed > r.bridgit.speed ∧
  r.bridgit.speed > r.carol.speed ∧
  r.length - r.bridgit.position.distance = 16 ∧
  r.length - r.carol.position.distance = 25 + (r.length - r.bridgit.position.distance)

theorem carol_distance_behind_anna (r : Race) (h : race_conditions r) :
  r.length - r.carol.position.distance = 37 :=
sorry

end NUMINAMATH_CALUDE_carol_distance_behind_anna_l1295_129515


namespace NUMINAMATH_CALUDE_number_problem_l1295_129569

theorem number_problem (x n : ℝ) : x = 4 ∧ n * x + 3 = 10 * x - 17 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1295_129569


namespace NUMINAMATH_CALUDE_homework_problem_l1295_129575

theorem homework_problem (total : ℕ) (finished_ratio unfinished_ratio : ℕ) 
  (h_total : total = 65)
  (h_ratio : finished_ratio = 9 ∧ unfinished_ratio = 4) :
  (finished_ratio * total) / (finished_ratio + unfinished_ratio) = 45 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_l1295_129575


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1295_129584

-- Define the ratio between p and k
def ratio_p_k (p k : ℝ) : Prop := p / k = Real.sqrt 3

-- Define the line equation
def line_equation (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the tangency condition
def is_tangent (k : ℝ) : Prop := 2 / Real.sqrt (k^2 + 1) = 1

-- Define the theorem
theorem p_sufficient_not_necessary (p q : ℝ) : 
  (∃ k, ratio_p_k p k ∧ is_tangent k) → 
  (∃ k, ratio_p_k q k ∧ is_tangent k) → 
  (∃ k, ratio_p_k p k ∧ is_tangent k → ∃ k', ratio_p_k q k' ∧ is_tangent k') ∧ 
  ¬(∀ k, ratio_p_k q k ∧ is_tangent k → ∃ k', ratio_p_k p k' ∧ is_tangent k') :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1295_129584


namespace NUMINAMATH_CALUDE_no_self_power_divisibility_l1295_129589

theorem no_self_power_divisibility (n : ℕ) : n > 1 → ¬(n ∣ 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_no_self_power_divisibility_l1295_129589


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l1295_129568

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l1295_129568


namespace NUMINAMATH_CALUDE_helium_pressure_change_l1295_129545

/-- Boyle's law for ideal gases at constant temperature -/
axiom boyles_law {p1 p2 v1 v2 : ℝ} (h : p1 * v1 = p2 * v2) : 
  p1 * v1 = p2 * v2

theorem helium_pressure_change (p1 v1 p2 v2 : ℝ) 
  (h1 : p1 = 4) 
  (h2 : v1 = 3) 
  (h3 : v2 = 6) 
  (h4 : p1 * v1 = p2 * v2) : 
  p2 = 2 := by
  sorry

#check helium_pressure_change

end NUMINAMATH_CALUDE_helium_pressure_change_l1295_129545


namespace NUMINAMATH_CALUDE_f_continuous_not_bounded_variation_l1295_129580

/-- The function f(x) = x sin(1/x) for x ≠ 0 and f(0) = 0 -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x * Real.sin (1 / x) else 0

/-- The interval [0, 1] -/
def I : Set ℝ := Set.Icc 0 1

theorem f_continuous_not_bounded_variation :
  ContinuousOn f I ∧ ¬ BoundedVariationOn f I := by sorry

end NUMINAMATH_CALUDE_f_continuous_not_bounded_variation_l1295_129580


namespace NUMINAMATH_CALUDE_range_of_f_l1295_129531

def f (x : ℝ) := x^2 + 4*x + 6

theorem range_of_f : 
  ∀ y ∈ Set.Icc 2 6, ∃ x ∈ Set.Ico (-3) 0, f x = y ∧
  ∀ x ∈ Set.Ico (-3) 0, 2 ≤ f x ∧ f x < 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1295_129531


namespace NUMINAMATH_CALUDE_class_size_problem_l1295_129596

theorem class_size_problem (passing_score : ℝ) (class_average : ℝ) 
  (pass_average_before : ℝ) (fail_average_before : ℝ)
  (pass_average_after : ℝ) (fail_average_after : ℝ)
  (points_added : ℝ) :
  passing_score = 65 →
  class_average = 66 →
  pass_average_before = 71 →
  fail_average_before = 56 →
  pass_average_after = 75 →
  fail_average_after = 59 →
  points_added = 5 →
  ∃ (total_students : ℕ), 
    15 < total_students ∧ 
    total_students < 30 ∧
    total_students = 24 :=
by sorry

end NUMINAMATH_CALUDE_class_size_problem_l1295_129596


namespace NUMINAMATH_CALUDE_floor_plus_x_equation_l1295_129541

theorem floor_plus_x_equation :
  ∃! x : ℝ, ⌊x⌋ + x = 20.5 :=
by
  sorry

end NUMINAMATH_CALUDE_floor_plus_x_equation_l1295_129541


namespace NUMINAMATH_CALUDE_common_volume_for_ratios_l1295_129540

/-- The volume of the common part of two identical triangular pyramids -/
noncomputable def common_volume (V : ℝ) (r : ℝ) : ℝ := sorry

/-- Theorem stating the volume of the common part for different ratios -/
theorem common_volume_for_ratios (V : ℝ) (V_pos : V > 0) :
  (common_volume V (1/2) = 2/3 * V) ∧
  (common_volume V (3/4) = 1/2 * V) ∧
  (common_volume V (2/3) = 110/243 * V) ∧
  (common_volume V (4/5) = 12/25 * V) := by sorry

end NUMINAMATH_CALUDE_common_volume_for_ratios_l1295_129540


namespace NUMINAMATH_CALUDE_complex_square_condition_l1295_129501

theorem complex_square_condition (a b : ℝ) : 
  (∃ (x y : ℝ), (x + y * Complex.I) ^ 2 = 2 * Complex.I ∧ (x ≠ 1 ∨ y ≠ 1)) ∧ 
  ((1 : ℝ) + (1 : ℝ) * Complex.I) ^ 2 = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_condition_l1295_129501


namespace NUMINAMATH_CALUDE_equation_solution_l1295_129597

theorem equation_solution (x y : ℚ) 
  (eq1 : 3 * x + y = 6) 
  (eq2 : x + 3 * y = 8) : 
  9 * x^2 + 15 * x * y + 9 * y^2 = 1629 / 16 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1295_129597


namespace NUMINAMATH_CALUDE_student_average_greater_than_true_average_l1295_129548

theorem student_average_greater_than_true_average 
  (u v w x y : ℝ) 
  (h : u ≤ v ∧ v ≤ w ∧ w ≤ x ∧ x ≤ y) : 
  ((u + v + w) / 3 + x + y) / 3 > (u + v + w + x + y) / 5 := by
  sorry

end NUMINAMATH_CALUDE_student_average_greater_than_true_average_l1295_129548


namespace NUMINAMATH_CALUDE_alyssa_grew_nine_turnips_l1295_129525

/-- The number of turnips Keith grew -/
def keith_turnips : ℕ := 6

/-- The total number of turnips Keith and Alyssa grew together -/
def total_turnips : ℕ := 15

/-- The number of turnips Alyssa grew -/
def alyssa_turnips : ℕ := total_turnips - keith_turnips

theorem alyssa_grew_nine_turnips : alyssa_turnips = 9 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_grew_nine_turnips_l1295_129525


namespace NUMINAMATH_CALUDE_two_negative_solutions_iff_b_in_range_l1295_129564

/-- The equation 9^x + |3^x + b| = 5 has exactly two negative real number solutions if and only if b is in the open interval (-5.25, -5) -/
theorem two_negative_solutions_iff_b_in_range (b : ℝ) : 
  (∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ x ≠ y ∧ 
    (9^x + |3^x + b| = 5) ∧ 
    (9^y + |3^y + b| = 5) ∧
    (∀ z : ℝ, z < 0 → z ≠ x → z ≠ y → 9^z + |3^z + b| ≠ 5)) ↔ 
  -5.25 < b ∧ b < -5 :=
sorry

end NUMINAMATH_CALUDE_two_negative_solutions_iff_b_in_range_l1295_129564


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1295_129593

theorem absolute_value_equation_solution : 
  ∃ (x : ℝ), (|x - 3| = 5 - 2*x) ∧ (x = 8/3 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1295_129593


namespace NUMINAMATH_CALUDE_power_function_property_l1295_129598

/-- A power function is a function of the form f(x) = x^a for some real number a -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

theorem power_function_property (f : ℝ → ℝ) (h1 : IsPowerFunction f) (h2 : f 4 = 2) :
  f (1/4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l1295_129598


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1295_129551

/-- A hyperbola is defined by its standard equation and properties -/
structure Hyperbola where
  /-- The standard equation of the hyperbola: y²/a² - x²/b² = 1 -/
  equation : ℝ → ℝ → Prop
  /-- The hyperbola passes through a given point -/
  passes_through : ℝ × ℝ → Prop
  /-- The asymptotic equations of the hyperbola -/
  asymptotic_equations : (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop)

/-- Theorem: Given a hyperbola that passes through (√3, 4) with asymptotic equations 2x ± y = 0,
    its standard equation is y²/4 - x² = 1 -/
theorem hyperbola_standard_equation (h : Hyperbola) :
  h.passes_through (Real.sqrt 3, 4) ∧
  h.asymptotic_equations = ((fun x y => 2*x = y), (fun x y => 2*x = -y)) →
  h.equation = fun x y => y^2/4 - x^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1295_129551
