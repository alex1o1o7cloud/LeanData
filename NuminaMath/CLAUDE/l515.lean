import Mathlib

namespace NUMINAMATH_CALUDE_square_difference_equality_l515_51530

theorem square_difference_equality : 1103^2 - 1097^2 - 1101^2 + 1099^2 = 8800 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l515_51530


namespace NUMINAMATH_CALUDE_ages_sum_l515_51522

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 128 → a + b + c = 18 := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l515_51522


namespace NUMINAMATH_CALUDE_yellow_balls_count_l515_51519

theorem yellow_balls_count (total_balls : ℕ) (yellow_probability : ℚ) 
  (h1 : total_balls = 40)
  (h2 : yellow_probability = 3/10) :
  (yellow_probability * total_balls : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l515_51519


namespace NUMINAMATH_CALUDE_iv_bottle_capacity_l515_51578

/-- Calculates the total capacity of an IV bottle given initial volume, flow rate, and elapsed time. -/
def totalCapacity (initialVolume : ℝ) (flowRate : ℝ) (elapsedTime : ℝ) : ℝ :=
  initialVolume + flowRate * elapsedTime

/-- Theorem stating that given the specified conditions, the total capacity of the IV bottle is 150 mL. -/
theorem iv_bottle_capacity :
  let initialVolume : ℝ := 100
  let flowRate : ℝ := 2.5
  let elapsedTime : ℝ := 12
  totalCapacity initialVolume flowRate elapsedTime = 150 := by
  sorry

#eval totalCapacity 100 2.5 12

end NUMINAMATH_CALUDE_iv_bottle_capacity_l515_51578


namespace NUMINAMATH_CALUDE_percentage_relation_l515_51552

theorem percentage_relation (x y z : ℝ) (hx : x = 0.06 * z) (hy : y = 0.18 * z) (hz : z > 0) :
  x / y * 100 = 100 / 3 :=
sorry

end NUMINAMATH_CALUDE_percentage_relation_l515_51552


namespace NUMINAMATH_CALUDE_expand_and_simplify_l515_51582

theorem expand_and_simplify (x : ℝ) : (x + 3) * (4 * x - 8) + x^2 = 5 * x^2 + 4 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l515_51582


namespace NUMINAMATH_CALUDE_tan_390_deg_l515_51565

/-- Proves that the tangent of 390 degrees is equal to √3/3 -/
theorem tan_390_deg : Real.tan (390 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_390_deg_l515_51565


namespace NUMINAMATH_CALUDE_revenue_not_increased_l515_51589

/-- The revenue function for the current year -/
def revenue (x : ℝ) : ℝ := 4*x^3 - 20*x^2 + 33*x - 17

/-- The previous year's revenue -/
def previous_revenue : ℝ := 20

theorem revenue_not_increased (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) : 
  revenue x ≤ previous_revenue := by
  sorry

#check revenue_not_increased

end NUMINAMATH_CALUDE_revenue_not_increased_l515_51589


namespace NUMINAMATH_CALUDE_M_sufficient_not_necessary_for_N_l515_51540

def M : Set ℝ := {x | x^2 < 3*x}
def N : Set ℝ := {x | |x - 1| < 2}

theorem M_sufficient_not_necessary_for_N :
  (∀ a : ℝ, a ∈ M → a ∈ N) ∧ (∃ b : ℝ, b ∈ N ∧ b ∉ M) := by sorry

end NUMINAMATH_CALUDE_M_sufficient_not_necessary_for_N_l515_51540


namespace NUMINAMATH_CALUDE_farmers_wheat_cleaning_l515_51558

/-- The total number of acres to be cleaned -/
def total_acres : ℕ := 480

/-- The original cleaning rate in acres per day -/
def original_rate : ℕ := 80

/-- The new cleaning rate with machinery in acres per day -/
def new_rate : ℕ := 90

/-- The number of acres cleaned on the last day -/
def last_day_acres : ℕ := 30

/-- The number of days taken to clean all acres -/
def days : ℕ := 6

theorem farmers_wheat_cleaning :
  (days - 1) * new_rate + last_day_acres = total_acres ∧
  days * original_rate = total_acres := by sorry

end NUMINAMATH_CALUDE_farmers_wheat_cleaning_l515_51558


namespace NUMINAMATH_CALUDE_range_of_m_for_always_nonnegative_quadratic_l515_51571

theorem range_of_m_for_always_nonnegative_quadratic :
  {m : ℝ | ∀ x : ℝ, x^2 + m*x + 2*m + 5 ≥ 0} = {m : ℝ | -2 ≤ m ∧ m ≤ 10} := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_always_nonnegative_quadratic_l515_51571


namespace NUMINAMATH_CALUDE_inscribed_ngon_existence_l515_51597

/-- An n-gon inscribed in a circle with sides parallel to n given lines -/
structure InscribedNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ
  lines : Fin n → ℝ × ℝ → Prop

/-- The measure of the angle at a vertex of the n-gon -/
def angle (ngon : InscribedNGon n) (i : Fin n) : ℝ := sorry

/-- The sum of odd-indexed angles -/
def sumOddAngles (ngon : InscribedNGon n) : ℝ := sorry

/-- The sum of even-indexed angles -/
def sumEvenAngles (ngon : InscribedNGon n) : ℝ := sorry

/-- The existence of an inscribed n-gon with sides parallel to given lines -/
def existsInscribedNGon (n : ℕ) (center : ℝ × ℝ) (radius : ℝ) (lines : Fin n → ℝ × ℝ → Prop) : Prop := sorry

theorem inscribed_ngon_existence (n : ℕ) (center : ℝ × ℝ) (radius : ℝ) (lines : Fin n → ℝ × ℝ → Prop) :
  (n % 2 = 1 ∧ existsInscribedNGon n center radius lines) ∨
  (n % 2 = 0 ∧ (existsInscribedNGon n center radius lines ↔
    ∃ (ngon : InscribedNGon n), sumOddAngles ngon = sumEvenAngles ngon)) :=
sorry

end NUMINAMATH_CALUDE_inscribed_ngon_existence_l515_51597


namespace NUMINAMATH_CALUDE_negative_inequality_l515_51549

theorem negative_inequality (a b : ℝ) (h : a > b) : -a < -b := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l515_51549


namespace NUMINAMATH_CALUDE_carols_age_l515_51512

theorem carols_age (bob_age carol_age : ℕ) : 
  bob_age + carol_age = 66 →
  carol_age = 3 * bob_age + 2 →
  carol_age = 50 := by
sorry

end NUMINAMATH_CALUDE_carols_age_l515_51512


namespace NUMINAMATH_CALUDE_circle_tangent_at_origin_l515_51527

/-- A circle in the xy-plane -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- The equation of the circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  x^2 + y^2 + c.D * x + c.E * y + c.F = 0

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- A circle is tangent to the x-axis at the origin -/
def tangent_at_origin (c : Circle) : Prop :=
  c.equation origin.x origin.y ∧
  ∀ (p : Point), p.y = 0 → p = origin ∨ ¬c.equation p.x p.y

theorem circle_tangent_at_origin (c : Circle) :
  tangent_at_origin c → c.E ≠ 0 ∧ c.D = 0 ∧ c.F = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_at_origin_l515_51527


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l515_51593

theorem opposite_of_negative_three : -((-3) : ℤ) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l515_51593


namespace NUMINAMATH_CALUDE_min_side_c_value_l515_51583

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the minimum value of c is approximately 2.25 -/
theorem min_side_c_value (a b c : ℝ) (A B C : ℝ) : 
  b = 2 →
  c * Real.cos B + b * Real.cos C = 4 * a * Real.sin B * Real.sin C →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ c ≥ 2.25 - ε :=
sorry

end NUMINAMATH_CALUDE_min_side_c_value_l515_51583


namespace NUMINAMATH_CALUDE_inequality_solution_count_l515_51513

theorem inequality_solution_count : 
  (Finset.filter (fun x => (x - 2)^2 ≤ 4) (Finset.range 100)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_count_l515_51513


namespace NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l515_51516

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation for a line being contained in a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Theorem statement
theorem perp_planes_necessary_not_sufficient 
  (α β : Plane) (m : Line) 
  (h_subset : subset_line_plane m α) :
  (∀ m α β, perp_line_plane m β → perp_planes α β) ∧ 
  (∃ m α β, perp_planes α β ∧ ¬perp_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l515_51516


namespace NUMINAMATH_CALUDE_implicit_function_derivative_specific_point_derivative_l515_51537

noncomputable section

/-- The implicit function defined by 10x^3 + 4x^2y + y^2 = 0 -/
def f (x y : ℝ) : ℝ := 10 * x^3 + 4 * x^2 * y + y^2

/-- The derivative of the implicit function -/
def f_derivative (x y : ℝ) : ℝ := (-15 * x^2 - 4 * x * y) / (2 * x^2 + y)

theorem implicit_function_derivative (x y : ℝ) (h : f x y = 0) :
  deriv (fun y => f x y) y = f_derivative x y := by sorry

theorem specific_point_derivative :
  f_derivative (-2) 4 = -7/3 := by sorry

end

end NUMINAMATH_CALUDE_implicit_function_derivative_specific_point_derivative_l515_51537


namespace NUMINAMATH_CALUDE_johns_total_time_l515_51508

/-- Represents the time John spent on various activities related to his travels and book writing --/
structure TravelTime where
  southAmerica : ℕ  -- Time spent exploring South America (in years)
  africa : ℕ        -- Time spent exploring Africa (in years)
  manuscriptTime : ℕ -- Time spent compiling notes into a manuscript (in months)
  editingTime : ℕ   -- Time spent finalizing the book with an editor (in months)

/-- Calculates the total time John spent on his adventures, note writing, and book creation --/
def totalTime (t : TravelTime) : ℕ :=
  -- Convert exploration time to months and add note-writing time
  (t.southAmerica * 12 + t.southAmerica * 6) +
  -- Convert Africa exploration time to months and add note-writing time
  (t.africa * 12 + t.africa * 4) +
  -- Add manuscript compilation and editing time
  t.manuscriptTime + t.editingTime

/-- Theorem stating that John's total time spent is 100 months --/
theorem johns_total_time :
  ∀ t : TravelTime,
    t.southAmerica = 3 ∧
    t.africa = 2 ∧
    t.manuscriptTime = 8 ∧
    t.editingTime = 6 →
    totalTime t = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_johns_total_time_l515_51508


namespace NUMINAMATH_CALUDE_seating_arrangement_l515_51551

structure Person where
  name : String
  is_sitting : Prop

def M : Person := ⟨"M", false⟩
def I : Person := ⟨"I", true⟩
def P : Person := ⟨"P", true⟩
def A : Person := ⟨"A", false⟩

theorem seating_arrangement :
  (¬M.is_sitting) →
  (¬M.is_sitting → I.is_sitting) →
  (I.is_sitting → P.is_sitting) →
  (¬A.is_sitting) →
  (I.is_sitting ∧ P.is_sitting) := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l515_51551


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l515_51585

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 11

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties :
  (∀ x, f x ≤ 13) ∧  -- Maximum value is 13
  f 3 = 5 ∧          -- f(3) = 5
  f (-1) = 5 ∧       -- f(-1) = 5
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) -- f is a quadratic function
  :=
by
  sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_quadratic_function_properties_l515_51585


namespace NUMINAMATH_CALUDE_min_printers_purchase_l515_51596

theorem min_printers_purchase (cost1 cost2 : ℕ) (h1 : cost1 = 350) (h2 : cost2 = 200) :
  ∃ (x y : ℕ), 
    x * cost1 = y * cost2 ∧ 
    x + y = 11 ∧
    ∀ (a b : ℕ), a * cost1 = b * cost2 → a + b ≥ 11 :=
by sorry

end NUMINAMATH_CALUDE_min_printers_purchase_l515_51596


namespace NUMINAMATH_CALUDE_roger_coins_count_l515_51599

/-- Calculates the total number of coins given the number of piles of quarters,
    piles of dimes, and coins per pile. -/
def totalCoins (quarterPiles dimePiles coinsPerPile : ℕ) : ℕ :=
  (quarterPiles + dimePiles) * coinsPerPile

/-- Theorem stating that with 3 piles of quarters, 3 piles of dimes,
    and 7 coins per pile, the total number of coins is 42. -/
theorem roger_coins_count :
  totalCoins 3 3 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_roger_coins_count_l515_51599


namespace NUMINAMATH_CALUDE_xyz_sum_root_l515_51546

theorem xyz_sum_root (x y z : ℝ) 
  (h1 : y + z = 22 / 2)
  (h2 : z + x = 24 / 2)
  (h3 : x + y = 26 / 2) :
  Real.sqrt (x * y * z * (x + y + z)) = 3 * Real.sqrt 70 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_root_l515_51546


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l515_51573

theorem smallest_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 150)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prop : ∃ (k : ℚ), a = 3 * k ∧ b = 5 * k ∧ c = (7/2) * k)
  (h_sum : a + b + c = total) : 
  min a (min b c) = 900 / 23 := by
sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l515_51573


namespace NUMINAMATH_CALUDE_probability_one_white_one_red_l515_51539

theorem probability_one_white_one_red (total : ℕ) (white : ℕ) (red : ℕ) :
  total = white + red →
  total = 15 →
  white = 10 →
  red = 5 →
  (white.choose 1 * red.choose 1 : ℚ) / total.choose 2 = 10 / 21 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_white_one_red_l515_51539


namespace NUMINAMATH_CALUDE_stratified_sampling_business_personnel_l515_51538

theorem stratified_sampling_business_personnel 
  (total_employees : ℕ) 
  (business_personnel : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 160) 
  (h2 : business_personnel = 120) 
  (h3 : sample_size = 20) :
  (business_personnel * sample_size) / total_employees = 15 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_business_personnel_l515_51538


namespace NUMINAMATH_CALUDE_ratio_IJ_IF_is_14_13_l515_51550

/-- A structure representing the geometric configuration described in the problem -/
structure TriangleConfiguration where
  /-- Point F -/
  F : ℝ × ℝ
  /-- Point G -/
  G : ℝ × ℝ
  /-- Point H -/
  H : ℝ × ℝ
  /-- Point I -/
  I : ℝ × ℝ
  /-- Point J -/
  J : ℝ × ℝ
  /-- FGH is a right triangle with right angle at H -/
  FGH_right_at_H : (F.1 - H.1) * (G.1 - H.1) + (F.2 - H.2) * (G.2 - H.2) = 0
  /-- FG = 5 -/
  FG_length : (F.1 - G.1)^2 + (F.2 - G.2)^2 = 25
  /-- GH = 12 -/
  GH_length : (G.1 - H.1)^2 + (G.2 - H.2)^2 = 144
  /-- FHI is a right triangle with right angle at F -/
  FHI_right_at_F : (H.1 - F.1) * (I.1 - F.1) + (H.2 - F.2) * (I.2 - F.2) = 0
  /-- FI = 15 -/
  FI_length : (F.1 - I.1)^2 + (F.2 - I.2)^2 = 225
  /-- H and I are on opposite sides of FG -/
  H_I_opposite_sides : ((G.1 - F.1) * (H.2 - F.2) - (G.2 - F.2) * (H.1 - F.1)) *
                       ((G.1 - F.1) * (I.2 - F.2) - (G.2 - F.2) * (I.1 - F.1)) < 0
  /-- IJ is parallel to FG -/
  IJ_parallel_FG : (J.1 - I.1) * (G.2 - F.2) = (J.2 - I.2) * (G.1 - F.1)
  /-- J is on the extension of GH -/
  J_on_GH_extended : ∃ t : ℝ, J.1 = G.1 + t * (H.1 - G.1) ∧ J.2 = G.2 + t * (H.2 - G.2)

/-- The main theorem stating that the ratio IJ/IF is equal to 14/13 -/
theorem ratio_IJ_IF_is_14_13 (config : TriangleConfiguration) :
  let IJ := ((config.I.1 - config.J.1)^2 + (config.I.2 - config.J.2)^2).sqrt
  let IF := ((config.I.1 - config.F.1)^2 + (config.I.2 - config.F.2)^2).sqrt
  IJ / IF = 14 / 13 :=
sorry

end NUMINAMATH_CALUDE_ratio_IJ_IF_is_14_13_l515_51550


namespace NUMINAMATH_CALUDE_smallest_m_is_12_l515_51517

-- Define the set T
def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

-- Define the property we want to prove
def has_nth_root_of_unity (n : ℕ) : Prop :=
  ∃ z ∈ T, z^n = 1

-- The main theorem
theorem smallest_m_is_12 :
  (∃ m : ℕ, m > 0 ∧ ∀ n ≥ m, has_nth_root_of_unity n) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ n ≥ m, has_nth_root_of_unity n) → m ≥ 12) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_12_l515_51517


namespace NUMINAMATH_CALUDE_constant_expression_l515_51525

theorem constant_expression (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 10)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 20) :
  5 * (a^3 + b^3 + c^3 + d^3 + e^3) - (a^4 + b^4 + c^4 + d^4 + e^4) = 120 := by
  sorry

end NUMINAMATH_CALUDE_constant_expression_l515_51525


namespace NUMINAMATH_CALUDE_negation_constant_arithmetic_sequence_l515_51510

theorem negation_constant_arithmetic_sequence :
  ¬(∀ s : ℕ → ℝ, (∀ n : ℕ, s n = s 0) → (∃ d : ℝ, ∀ n : ℕ, s (n + 1) = s n + d)) ↔
  (∃ s : ℕ → ℝ, (∀ n : ℕ, s n = s 0) ∧ ¬(∃ d : ℝ, ∀ n : ℕ, s (n + 1) = s n + d)) :=
by sorry

end NUMINAMATH_CALUDE_negation_constant_arithmetic_sequence_l515_51510


namespace NUMINAMATH_CALUDE_max_min_product_l515_51576

theorem max_min_product (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a + b + c = 8) (h5 : a * b + b * c + c * a = 16) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 16 / 9 ∧
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 16 / 9 :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l515_51576


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l515_51518

theorem smallest_prime_divisor_of_sum (p : ℕ) : 
  Prime p ∧ p ∣ (3^11 + 5^13) → p = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l515_51518


namespace NUMINAMATH_CALUDE_forty_percent_of_fifty_percent_l515_51536

theorem forty_percent_of_fifty_percent (x : ℝ) : (0.4 * (0.5 * x)) = (0.2 * x) := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_fifty_percent_l515_51536


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l515_51541

theorem square_perimeter_from_area (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 400 → 
  area = side ^ 2 → 
  perimeter = 4 * side → 
  perimeter = 80 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l515_51541


namespace NUMINAMATH_CALUDE_conference_seating_arrangement_l515_51534

theorem conference_seating_arrangement :
  ∃! y : ℕ, ∃ x : ℕ,
    (9 * x + 10 * y = 73) ∧
    (0 < x) ∧ (0 < y) ∧
    y = 1 := by
  sorry

end NUMINAMATH_CALUDE_conference_seating_arrangement_l515_51534


namespace NUMINAMATH_CALUDE_exam_mode_l515_51563

/-- Represents a score in the music theory exam -/
structure Score where
  value : ℕ
  deriving Repr

/-- Represents the frequency of each score -/
def ScoreFrequency := Score → ℕ

/-- The set of all scores in the exam -/
def ExamScores : Set Score := sorry

/-- The frequency distribution of scores in the exam -/
def examFrequency : ScoreFrequency := sorry

/-- Definition of mode: the score that appears most frequently -/
def isMode (s : Score) (freq : ScoreFrequency) (scores : Set Score) : Prop :=
  ∀ t ∈ scores, freq s ≥ freq t

/-- The mode of the exam scores is 88 -/
theorem exam_mode :
  ∃ s : Score, s.value = 88 ∧ isMode s examFrequency ExamScores := by sorry

end NUMINAMATH_CALUDE_exam_mode_l515_51563


namespace NUMINAMATH_CALUDE_seashells_after_giving_away_starfish_count_indeterminate_l515_51504

/-- Proves that the number of seashells after giving some away is correct -/
theorem seashells_after_giving_away 
  (initial_seashells : ℕ) 
  (seashells_given_away : ℕ) 
  (final_seashells : ℕ) 
  (h1 : initial_seashells = 49)
  (h2 : seashells_given_away = 13)
  (h3 : final_seashells = 36) :
  final_seashells = initial_seashells - seashells_given_away :=
by sorry

/-- The number of starfish cannot be determined from the given information -/
theorem starfish_count_indeterminate 
  (initial_seashells : ℕ) 
  (seashells_given_away : ℕ) 
  (final_seashells : ℕ) 
  (starfish : ℕ) 
  (h1 : initial_seashells = 49)
  (h2 : seashells_given_away = 13)
  (h3 : final_seashells = 36) :
  ∃ (n : ℕ), starfish = n :=
by sorry

end NUMINAMATH_CALUDE_seashells_after_giving_away_starfish_count_indeterminate_l515_51504


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l515_51521

/-- The focal length of a hyperbola with equation x²/4 - y²/5 = 1 is 6 -/
theorem hyperbola_focal_length : ∃ (a b c : ℝ),
  (a^2 = 4 ∧ b^2 = 5) →
  (c^2 = a^2 + b^2) →
  (2 * c = 6) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l515_51521


namespace NUMINAMATH_CALUDE_min_value_of_expression_l515_51545

theorem min_value_of_expression (x y : ℝ) (h1 : x > 1) (h2 : x - y = 1) :
  x + 1/y ≥ 3 ∧ ∃ (x0 y0 : ℝ), x0 > 1 ∧ x0 - y0 = 1 ∧ x0 + 1/y0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l515_51545


namespace NUMINAMATH_CALUDE_product_45_360_trailing_zeros_l515_51587

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- Theorem: The product of 45 and 360 has 2 trailing zeros -/
theorem product_45_360_trailing_zeros : trailingZeros (45 * 360) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_45_360_trailing_zeros_l515_51587


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l515_51577

/-- Triangle ABC with inscribed rectangle PQRS --/
structure TriangleWithRectangle where
  /-- Side lengths of triangle ABC --/
  AB : ℝ
  BC : ℝ
  CA : ℝ
  /-- Coefficient of x in the area formula --/
  a : ℝ
  /-- Coefficient of x^2 in the area formula --/
  b : ℝ
  /-- The area of rectangle PQRS is given by a * x - b * x^2 --/
  area_formula : ∀ x, 0 ≤ x → x ≤ BC → 0 ≤ a * x - b * x^2

/-- The main theorem --/
theorem inscribed_rectangle_area_coefficient
  (t : TriangleWithRectangle)
  (h1 : t.AB = 13)
  (h2 : t.BC = 24)
  (h3 : t.CA = 15) :
  t.b = 13 / 48 :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l515_51577


namespace NUMINAMATH_CALUDE_square_number_difference_l515_51501

theorem square_number_difference (n k l : ℕ) :
  (∃ x : ℕ, x^2 < n ∧ n < (x+1)^2) →  -- n is between consecutive squares
  (∃ x : ℕ, n - k = x^2) →            -- n - k is a square number
  (∃ x : ℕ, n + l = x^2) →            -- n + l is a square number
  (∃ x : ℕ, n - k - l = x^2) :=        -- n - k - l is a square number
by sorry

end NUMINAMATH_CALUDE_square_number_difference_l515_51501


namespace NUMINAMATH_CALUDE_original_number_is_fifteen_l515_51543

theorem original_number_is_fifteen : 
  ∃ x : ℝ, 3 * (2 * x + 5) = 105 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_fifteen_l515_51543


namespace NUMINAMATH_CALUDE_horner_method_correct_l515_51528

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x -/
def f_coeffs : List ℝ := [5, 4, 3, 2, 1, 0]

theorem horner_method_correct :
  horner_eval f_coeffs 3 = 1641 := by
  sorry

#eval horner_eval f_coeffs 3

end NUMINAMATH_CALUDE_horner_method_correct_l515_51528


namespace NUMINAMATH_CALUDE_circle_radius_from_arc_length_l515_51526

/-- Given a circle where the arc length corresponding to a central angle of 135° is 3π,
    prove that the radius of the circle is 4. -/
theorem circle_radius_from_arc_length :
  ∀ r : ℝ, (135 / 180 * π * r = 3 * π) → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_arc_length_l515_51526


namespace NUMINAMATH_CALUDE_fence_pole_count_l515_51569

/-- Represents a rectangular fence with an internal divider -/
structure RectangularFence where
  longer_side : ℕ
  shorter_side : ℕ
  has_internal_divider : Bool

/-- Calculates the total number of poles needed for a rectangular fence with an internal divider -/
def total_poles (fence : RectangularFence) : ℕ :=
  let perimeter_poles := 2 * (fence.longer_side + fence.shorter_side) - 4
  let internal_poles := if fence.has_internal_divider then fence.shorter_side - 1 else 0
  perimeter_poles + internal_poles

/-- Theorem stating that a rectangular fence with 35 poles on the longer side, 
    27 poles on the shorter side, and an internal divider needs 146 poles in total -/
theorem fence_pole_count : 
  let fence := RectangularFence.mk 35 27 true
  total_poles fence = 146 := by sorry

end NUMINAMATH_CALUDE_fence_pole_count_l515_51569


namespace NUMINAMATH_CALUDE_saree_price_calculation_l515_51561

theorem saree_price_calculation (P : ℝ) : 
  P * (1 - 0.20) * (1 - 0.10) = 108 → P = 150 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l515_51561


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l515_51554

/-- The function f(x) = 3 + a^(x-1) always passes through the point (1, 4) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f := λ x : ℝ => 3 + a^(x - 1)
  f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l515_51554


namespace NUMINAMATH_CALUDE_project_time_ratio_l515_51562

/-- Given a project where three people (Pat, Kate, and Mark) charged time, 
    prove that the ratio of time charged by Pat to Mark is 1:3 -/
theorem project_time_ratio (total_hours kate_hours pat_hours mark_hours : ℕ) : 
  total_hours = 216 →
  pat_hours = 2 * kate_hours →
  mark_hours = kate_hours + 120 →
  total_hours = kate_hours + pat_hours + mark_hours →
  pat_hours * 3 = mark_hours := by
  sorry

end NUMINAMATH_CALUDE_project_time_ratio_l515_51562


namespace NUMINAMATH_CALUDE_new_year_day_frequency_new_year_day_sunday_more_frequent_l515_51581

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a date -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Function to determine if a year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Function to get the day of the week for a given date -/
noncomputable def getDayOfWeek (date : Date) : DayOfWeek :=
  sorry

/-- Function to count occurrences of a specific day of the week as New Year's Day over 400 years -/
noncomputable def countNewYearDay (day : DayOfWeek) (startYear : ℕ) : ℕ :=
  sorry

/-- Theorem stating that New Year's Day falls on Sunday more frequently than on Monday over a 400-year cycle -/
theorem new_year_day_frequency (startYear : ℕ) :
  countNewYearDay DayOfWeek.Sunday startYear > countNewYearDay DayOfWeek.Monday startYear :=
by
  sorry

/-- Given condition: 23 October 1948 was a Saturday -/
axiom oct_23_1948_saturday : getDayOfWeek ⟨1948, 10, 23⟩ = DayOfWeek.Saturday

/-- Theorem to prove the frequency of New Year's Day on Sunday vs Monday -/
theorem new_year_day_sunday_more_frequent :
  ∃ startYear, countNewYearDay DayOfWeek.Sunday startYear > countNewYearDay DayOfWeek.Monday startYear :=
by
  sorry

end NUMINAMATH_CALUDE_new_year_day_frequency_new_year_day_sunday_more_frequent_l515_51581


namespace NUMINAMATH_CALUDE_max_three_digit_operation_l515_51548

theorem max_three_digit_operation :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 2 * (200 + n) ≤ 2398 :=
by sorry

end NUMINAMATH_CALUDE_max_three_digit_operation_l515_51548


namespace NUMINAMATH_CALUDE_quiz_score_theorem_l515_51570

/-- Represents a quiz score configuration -/
structure QuizScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- The quiz scoring system -/
def quizScoring (qs : QuizScore) : ℚ :=
  4 * qs.correct + 1.5 * qs.unanswered

/-- Predicate for valid quiz configurations -/
def isValidQuizScore (qs : QuizScore) : Prop :=
  qs.correct + qs.unanswered + qs.incorrect = 30

/-- Predicate for scores achievable in exactly three ways -/
def hasExactlyThreeConfigurations (score : ℚ) : Prop :=
  ∃ (c1 c2 c3 : QuizScore),
    c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
    isValidQuizScore c1 ∧ isValidQuizScore c2 ∧ isValidQuizScore c3 ∧
    quizScoring c1 = score ∧ quizScoring c2 = score ∧ quizScoring c3 = score ∧
    ∀ c, isValidQuizScore c ∧ quizScoring c = score → c = c1 ∨ c = c2 ∨ c = c3

theorem quiz_score_theorem :
  ∃ score, 0 ≤ score ∧ score ≤ 120 ∧ hasExactlyThreeConfigurations score := by
  sorry

end NUMINAMATH_CALUDE_quiz_score_theorem_l515_51570


namespace NUMINAMATH_CALUDE_temperature_difference_l515_51511

theorem temperature_difference (N : ℝ) : 
  (∃ (M L : ℝ),
    -- Noon conditions
    M = L + N ∧
    -- 6:00 PM conditions
    (M - 11) - (L + 5) = 6 ∨ (M - 11) - (L + 5) = -6) →
  N = 22 ∨ N = 10 :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_l515_51511


namespace NUMINAMATH_CALUDE_length_AX_in_tangent_circles_configuration_l515_51500

/-- Two circles with radii r₁ and r₂ that are externally tangent -/
structure ExternallyTangentCircles (r₁ r₂ : ℝ) :=
  (center_distance : ℝ)
  (tangent_point : ℝ × ℝ)
  (external_tangent_length : ℝ)
  (h_center_distance : center_distance = r₁ + r₂)

/-- The configuration of two externally tangent circles with their common tangents -/
structure TangentCirclesConfiguration (r₁ r₂ : ℝ) extends ExternallyTangentCircles r₁ r₂ :=
  (common_external_tangent_point_A : ℝ × ℝ)
  (common_external_tangent_point_B : ℝ × ℝ)
  (common_internal_tangent_intersection : ℝ × ℝ)

/-- The theorem stating the length of AX in the given configuration -/
theorem length_AX_in_tangent_circles_configuration 
  (config : TangentCirclesConfiguration 20 13) : 
  ∃ (AX : ℝ), AX = 2 * Real.sqrt 65 :=
sorry

end NUMINAMATH_CALUDE_length_AX_in_tangent_circles_configuration_l515_51500


namespace NUMINAMATH_CALUDE_left_shoe_probability_l515_51594

/-- The probability of randomly picking a left shoe from a shoe cabinet with 3 pairs of shoes is 1/2. -/
theorem left_shoe_probability (num_pairs : ℕ) (h : num_pairs = 3) :
  (num_pairs : ℚ) / (2 * num_pairs : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_left_shoe_probability_l515_51594


namespace NUMINAMATH_CALUDE_angle_B_is_30_degrees_l515_51559

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the law of sines
def lawOfSines (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B

-- Define the theorem
theorem angle_B_is_30_degrees (t : Triangle) 
  (h1 : t.A = 45 * π / 180)
  (h2 : t.a = 6)
  (h3 : t.b = 3 * Real.sqrt 2) :
  t.B = 30 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_angle_B_is_30_degrees_l515_51559


namespace NUMINAMATH_CALUDE_students_liking_sports_l515_51592

theorem students_liking_sports (B C : Finset Nat) : 
  (B.card = 9) → 
  (C.card = 8) → 
  ((B ∩ C).card = 6) → 
  ((B ∪ C).card = 11) := by
sorry

end NUMINAMATH_CALUDE_students_liking_sports_l515_51592


namespace NUMINAMATH_CALUDE_remaining_money_l515_51502

def initial_amount : ℚ := 10
def candy_bar_cost : ℚ := 2
def chocolate_cost : ℚ := 3
def soda_cost : ℚ := 1.5
def gum_cost : ℚ := 1.25

theorem remaining_money :
  initial_amount - candy_bar_cost - chocolate_cost - soda_cost - gum_cost = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l515_51502


namespace NUMINAMATH_CALUDE_fraction_of_number_l515_51580

theorem fraction_of_number : (7 : ℚ) / 25 * 89473 = 25052.44 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_number_l515_51580


namespace NUMINAMATH_CALUDE_total_spent_equals_79_09_l515_51588

def shorts_price : Float := 15.00
def jacket_price : Float := 14.82
def shirt_price : Float := 12.51
def shoes_price : Float := 21.67
def hat_price : Float := 8.75
def belt_price : Float := 6.34

def total_spent : Float := shorts_price + jacket_price + shirt_price + shoes_price + hat_price + belt_price

theorem total_spent_equals_79_09 : total_spent = 79.09 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_79_09_l515_51588


namespace NUMINAMATH_CALUDE_total_cupcakes_l515_51507

theorem total_cupcakes (cupcakes_per_event : ℕ) (number_of_events : ℕ) 
  (h1 : cupcakes_per_event = 156) 
  (h2 : number_of_events = 12) : 
  cupcakes_per_event * number_of_events = 1872 := by
sorry

end NUMINAMATH_CALUDE_total_cupcakes_l515_51507


namespace NUMINAMATH_CALUDE_max_value_inequality_l515_51555

theorem max_value_inequality (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  x * y * z * (x + y + z + w) / ((x + y + z)^2 * (y + z + w)^2) ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l515_51555


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l515_51503

theorem line_segment_endpoint (x : ℝ) :
  x < 0 ∧
  ((x - 1)^2 + (8 - 3)^2).sqrt = 15 →
  x = 1 - 10 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l515_51503


namespace NUMINAMATH_CALUDE_simplify_and_find_ratio_l515_51533

theorem simplify_and_find_ratio (k : ℝ) : 
  (6 * k + 18) / 6 = k + 3 ∧ (1 : ℝ) / 3 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_find_ratio_l515_51533


namespace NUMINAMATH_CALUDE_infinite_pairs_geometric_progression_l515_51584

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricProgression (seq : Fin 4 → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ i : Fin 3, seq (i + 1) = seq i * r

/-- There are infinitely many pairs of real numbers (a,b) such that 12, a, b, ab form a geometric progression. -/
theorem infinite_pairs_geometric_progression :
  {(a, b) : ℝ × ℝ | IsGeometricProgression (λ i => match i with
    | 0 => 12
    | 1 => a
    | 2 => b
    | 3 => a * b)} = Set.univ := by
  sorry


end NUMINAMATH_CALUDE_infinite_pairs_geometric_progression_l515_51584


namespace NUMINAMATH_CALUDE_no_base_for_256_with_4_digits_l515_51520

theorem no_base_for_256_with_4_digits :
  ¬ ∃ b : ℕ, b ≥ 2 ∧ b^3 ≤ 256 ∧ 256 < b^4 := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_256_with_4_digits_l515_51520


namespace NUMINAMATH_CALUDE_reconstruct_diagonals_l515_51595

/-- Represents a convex polygon with labeled vertices -/
structure LabeledPolygon where
  vertices : Finset ℕ
  labels : vertices → ℕ

/-- Represents a set of non-intersecting diagonals in a polygon -/
def Diagonals (p : LabeledPolygon) := Finset (Finset ℕ)

/-- Checks if a set of diagonals divides a polygon into triangles -/
def divides_into_triangles (p : LabeledPolygon) (d : Diagonals p) : Prop := sorry

/-- Checks if a set of diagonals matches the vertex labels -/
def matches_labels (p : LabeledPolygon) (d : Diagonals p) : Prop := sorry

/-- Main theorem: For any labeled convex polygon, there exists a unique set of diagonals
    that divides it into triangles and matches the labels -/
theorem reconstruct_diagonals (p : LabeledPolygon) : 
  ∃! d : Diagonals p, divides_into_triangles p d ∧ matches_labels p d := by sorry

end NUMINAMATH_CALUDE_reconstruct_diagonals_l515_51595


namespace NUMINAMATH_CALUDE_simplify_expression_l515_51529

theorem simplify_expression (w : ℝ) : 
  2 * w + 3 - 4 * w - 6 + 7 * w + 9 - 8 * w - 12 + 3 * (2 * w - 1) = 3 * w - 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l515_51529


namespace NUMINAMATH_CALUDE_quadratic_root_value_l515_51560

theorem quadratic_root_value (a : ℝ) : 
  ((a + 1) * 1^2 - 1 + a^2 - 2*a - 2 = 0) → a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l515_51560


namespace NUMINAMATH_CALUDE_salary_january_l515_51524

def employee_salary (jan feb mar apr may jun : ℕ) : Prop :=
  -- Average salary for Jan, Feb, Mar, Apr, May is Rs. 9,000
  (jan + feb + mar + apr + may) / 5 = 9000 ∧
  -- Average salary for Feb, Mar, Apr, May, Jun is Rs. 10,000
  (feb + mar + apr + may + jun) / 5 = 10000 ∧
  -- Employee receives a bonus of Rs. 1,500 in March
  ∃ (base_mar : ℕ), mar = base_mar + 1500 ∧
  -- Employee receives a deduction of Rs. 1,000 in June
  ∃ (base_jun : ℕ), jun = base_jun - 1000 ∧
  -- Salary for May is Rs. 7,500
  may = 7500 ∧
  -- No pay increase or decrease in the given time frame
  ∃ (base : ℕ), feb = base ∧ apr = base ∧ base_mar = base ∧ base_jun = base

theorem salary_january :
  ∀ (jan feb mar apr may jun : ℕ),
  employee_salary jan feb mar apr may jun →
  jan = 4500 :=
by sorry

end NUMINAMATH_CALUDE_salary_january_l515_51524


namespace NUMINAMATH_CALUDE_cl2_moles_required_l515_51515

-- Define the reaction components
structure Reaction where
  c2h6 : ℝ
  cl2 : ℝ
  c2cl6 : ℝ
  hcl : ℝ

-- Define the balanced equation ratios
def balancedRatio : Reaction := {
  c2h6 := 1,
  cl2 := 6,
  c2cl6 := 1,
  hcl := 6
}

-- Define the given reaction
def givenReaction : Reaction := {
  c2h6 := 2,
  cl2 := 0,  -- This is what we need to prove
  c2cl6 := 2,
  hcl := 12
}

-- Theorem statement
theorem cl2_moles_required (r : Reaction) :
  r.c2h6 = givenReaction.c2h6 ∧
  r.c2cl6 = givenReaction.c2cl6 ∧
  r.hcl = givenReaction.hcl →
  r.cl2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_cl2_moles_required_l515_51515


namespace NUMINAMATH_CALUDE_abs_and_opposite_l515_51509

theorem abs_and_opposite :
  (abs (-2) = 2) ∧ (-(1/2) = -1/2) := by sorry

end NUMINAMATH_CALUDE_abs_and_opposite_l515_51509


namespace NUMINAMATH_CALUDE_smallest_k_product_equals_sum_l515_51532

theorem smallest_k_product_equals_sum (k : ℕ) : k = 10 ↔ 
  (k ≥ 3 ∧ 
   ∃ a b : ℕ, a ∈ Finset.range k ∧ b ∈ Finset.range k ∧ a ≠ b ∧
   a * b = (k * (k + 1) / 2) - a - b ∧
   ∀ m : ℕ, m ≥ 3 → m < k → 
     ¬∃ x y : ℕ, x ∈ Finset.range m ∧ y ∈ Finset.range m ∧ x ≠ y ∧
     x * y = (m * (m + 1) / 2) - x - y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_product_equals_sum_l515_51532


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l515_51575

/-- Given a circle with area M and circumference N, if M/N = 15, then the radius is 30 -/
theorem circle_radius_from_area_circumference_ratio (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) :
  ∃ r : ℝ, r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l515_51575


namespace NUMINAMATH_CALUDE_a_minus_b_greater_than_one_l515_51574

theorem a_minus_b_greater_than_one (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hf : ∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    r₁^3 + a*r₁^2 + 2*b*r₁ - 1 = 0 ∧
    r₂^3 + a*r₂^2 + 2*b*r₂ - 1 = 0 ∧
    r₃^3 + a*r₃^2 + 2*b*r₃ - 1 = 0)
  (hg : ∀ x : ℝ, 2*x^2 + 2*b*x + a ≠ 0) :
  a - b > 1 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_greater_than_one_l515_51574


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l515_51598

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l515_51598


namespace NUMINAMATH_CALUDE_composite_polynomial_l515_51567

theorem composite_polynomial (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℕ, 1 < k ∧ k < n^(5*n-1) + n^(5*n-2) + n^(5*n-3) + n + 1 ∧ 
  (n^(5*n-1) + n^(5*n-2) + n^(5*n-3) + n + 1) % k = 0 :=
by sorry

end NUMINAMATH_CALUDE_composite_polynomial_l515_51567


namespace NUMINAMATH_CALUDE_two_digit_average_decimal_l515_51505

theorem two_digit_average_decimal (m n : ℕ) : 
  10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100 →  -- m and n are 2-digit positive integers
  (m + n) / 2 = m + n / 100 →             -- their average equals the decimal representation
  max m n = 50 :=                         -- the larger of the two is 50
by sorry

end NUMINAMATH_CALUDE_two_digit_average_decimal_l515_51505


namespace NUMINAMATH_CALUDE_dave_remaining_tickets_l515_51566

/-- Given that Dave had 13 tickets initially and used 6 tickets,
    prove that he has 7 tickets left. -/
theorem dave_remaining_tickets :
  let initial_tickets : ℕ := 13
  let used_tickets : ℕ := 6
  initial_tickets - used_tickets = 7 := by
sorry

end NUMINAMATH_CALUDE_dave_remaining_tickets_l515_51566


namespace NUMINAMATH_CALUDE_solution_set_equality_l515_51553

theorem solution_set_equality (x : ℝ) : 
  Set.Icc (-1 : ℝ) (7/3 : ℝ) = { x | |x - 1| + |2*x - 1| ≤ 5 } := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l515_51553


namespace NUMINAMATH_CALUDE_sum_of_ages_is_32_l515_51590

/-- Viggo's age when his brother was 2 years old -/
def viggos_initial_age : ℕ := 2 * 2 + 10

/-- The current age of Viggo's younger brother -/
def brothers_current_age : ℕ := 10

/-- The number of years that have passed since the initial condition -/
def years_passed : ℕ := brothers_current_age - 2

/-- Viggo's current age -/
def viggos_current_age : ℕ := viggos_initial_age + years_passed

/-- The sum of Viggo's and his younger brother's current ages -/
def sum_of_ages : ℕ := viggos_current_age + brothers_current_age

theorem sum_of_ages_is_32 : sum_of_ages = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_is_32_l515_51590


namespace NUMINAMATH_CALUDE_smaller_fraction_l515_51542

theorem smaller_fraction (x y : ℝ) (sum_eq : x + y = 5/6) (prod_eq : x * y = 1/8) :
  min x y = (5 - Real.sqrt 7) / 12 := by sorry

end NUMINAMATH_CALUDE_smaller_fraction_l515_51542


namespace NUMINAMATH_CALUDE_expected_deliveries_l515_51591

theorem expected_deliveries (packages_yesterday : ℕ) (success_rate : ℚ) :
  packages_yesterday = 80 →
  success_rate = 90 / 100 →
  (packages_yesterday * 2 : ℚ) * success_rate = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_expected_deliveries_l515_51591


namespace NUMINAMATH_CALUDE_power_fraction_equality_l515_51564

theorem power_fraction_equality : (40 ^ 56) / (10 ^ 28) = 160 ^ 28 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l515_51564


namespace NUMINAMATH_CALUDE_smallest_prime_sum_l515_51514

theorem smallest_prime_sum (a b c d : ℕ) : 
  (Prime a ∧ Prime b ∧ Prime c ∧ Prime d) →
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  Prime (a + b + c + d) →
  (Prime (a + b) ∧ Prime (a + c) ∧ Prime (a + d) ∧ 
   Prime (b + c) ∧ Prime (b + d) ∧ Prime (c + d)) →
  (Prime (a + b + c) ∧ Prime (a + b + d) ∧ 
   Prime (a + c + d) ∧ Prime (b + c + d)) →
  a + b + c + d ≥ 31 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_sum_l515_51514


namespace NUMINAMATH_CALUDE_snowman_volume_l515_51506

theorem snowman_volume (π : ℝ) (h : π > 0) : 
  let sphere_volume (r : ℝ) := (4 / 3) * π * r^3
  sphere_volume 4 + sphere_volume 6 + sphere_volume 8 = (3168 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_snowman_volume_l515_51506


namespace NUMINAMATH_CALUDE_divisibility_of_2023_power_l515_51568

theorem divisibility_of_2023_power (n : ℕ) : 
  ∃ (k : ℕ), 2023^2023 - 2023^2021 = k * 2022 * 2023 * 2024 :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_2023_power_l515_51568


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l515_51531

/-- Given a line y = kx + b tangent to the curve y = x³ + ax + 1 at the point (2, 3), prove that b = -15 -/
theorem tangent_line_b_value (k a : ℝ) : 
  (3 = 2 * k + b) →  -- Line equation at (2, 3)
  (3 = 8 + 2 * a + 1) →  -- Curve equation at (2, 3)
  (k = 3 * 2^2 + a) →  -- Slope of the tangent line equals derivative of the curve at x = 2
  (b = -15) := by sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l515_51531


namespace NUMINAMATH_CALUDE_find_x_l515_51586

theorem find_x : ∃ x : ℚ, (3 * x - 6 + 4) / 7 = 15 ∧ x = 107 / 3 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l515_51586


namespace NUMINAMATH_CALUDE_aaron_cards_total_l515_51556

/-- Given that Aaron initially has 5 cards and finds 62 more, 
    prove that he ends up with 67 cards in total. -/
theorem aaron_cards_total (initial_cards : ℕ) (found_cards : ℕ) : 
  initial_cards = 5 → found_cards = 62 → initial_cards + found_cards = 67 := by
  sorry

end NUMINAMATH_CALUDE_aaron_cards_total_l515_51556


namespace NUMINAMATH_CALUDE_min_circumcircle_area_l515_51572

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define points A and B as tangent points on circle C
def tangent_points (xA yA xB yB : ℝ) : Prop :=
  circle_C xA yA ∧ circle_C xB yB

-- Define the theorem
theorem min_circumcircle_area (xP yP xA yA xB yB : ℝ) 
  (h_P : point_P xP yP) 
  (h_AB : tangent_points xA yA xB yB) :
  ∃ (r : ℝ), r > 0 ∧ r^2 * π = 5*π/4 ∧ 
  ∀ (r' : ℝ), r' > 0 → (∃ (xP' yP' xA' yA' xB' yB' : ℝ),
    point_P xP' yP' ∧ 
    tangent_points xA' yA' xB' yB' ∧ 
    r'^2 * π ≥ 5*π/4) :=
sorry

end NUMINAMATH_CALUDE_min_circumcircle_area_l515_51572


namespace NUMINAMATH_CALUDE_pencils_left_problem_l515_51523

def pencils_left (total_pencils : ℕ) (num_students : ℕ) : ℕ :=
  total_pencils - (num_students * (total_pencils / num_students))

theorem pencils_left_problem :
  pencils_left 42 12 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_pencils_left_problem_l515_51523


namespace NUMINAMATH_CALUDE_vector_problem_l515_51535

/-- Given two perpendicular vectors a and c, and two parallel vectors b and c in ℝ², 
    prove that x = 4, y = -8, and the magnitude of a + b is 10. -/
theorem vector_problem (x y : ℝ) 
  (a b c : ℝ × ℝ)
  (ha : a = (x, 2))
  (hb : b = (4, y))
  (hc : c = (1, -2))
  (hac_perp : a.1 * c.1 + a.2 * c.2 = 0)
  (hbc_par : b.1 * c.2 = b.2 * c.1) :
  x = 4 ∧ y = -8 ∧ Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l515_51535


namespace NUMINAMATH_CALUDE_compound_statement_properties_l515_51544

/-- Given two propositions p and q, prove the compound statement properties --/
theorem compound_statement_properties (p q : Prop) 
  (hp : p ↔ (8 + 7 = 16)) 
  (hq : q ↔ (π > 3)) : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬p := by sorry

end NUMINAMATH_CALUDE_compound_statement_properties_l515_51544


namespace NUMINAMATH_CALUDE_binary_101101_conversion_l515_51547

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, x) => acc + (if x then 2^i else 0)) 0

def decimal_to_base7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_101101_conversion :
  (binary_to_decimal binary_101101 = 45) ∧
  (decimal_to_base7 45 = [6, 3]) := by sorry

end NUMINAMATH_CALUDE_binary_101101_conversion_l515_51547


namespace NUMINAMATH_CALUDE_polygon_angles_theorem_l515_51579

theorem polygon_angles_theorem (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 = 2 * 360) →
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_polygon_angles_theorem_l515_51579


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l515_51557

theorem system_of_equations_solution :
  ∃ (x y : ℝ), x + 2*y = 3 ∧ x - 4*y = 9 → x = 5 ∧ y = -1 := by
  sorry

#check system_of_equations_solution

end NUMINAMATH_CALUDE_system_of_equations_solution_l515_51557
