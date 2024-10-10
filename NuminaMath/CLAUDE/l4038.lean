import Mathlib

namespace bluray_price_l4038_403805

/-- The price of a Blu-ray movie given the following conditions:
  * 8 DVDs cost $12 each
  * There are 4 Blu-ray movies
  * The average price of all 12 movies is $14
-/
theorem bluray_price :
  ∀ (x : ℝ),
  (8 * 12 + 4 * x) / 12 = 14 →
  x = 18 :=
by sorry

end bluray_price_l4038_403805


namespace no_real_solutions_l4038_403834

theorem no_real_solutions (k : ℝ) : 
  (∀ x : ℝ, x^2 ≠ 5*x + k) ↔ k < -25/4 := by sorry

end no_real_solutions_l4038_403834


namespace contractor_daily_wage_l4038_403888

/-- Contractor's daily wage problem -/
theorem contractor_daily_wage (total_days : ℕ) (absent_days : ℕ) (fine_per_day : ℚ) (total_pay : ℚ) :
  total_days = 30 →
  absent_days = 2 →
  fine_per_day = 15/2 →
  total_pay = 685 →
  ∃ (daily_wage : ℚ), 
    daily_wage * (total_days - absent_days : ℚ) - fine_per_day * absent_days = total_pay ∧
    daily_wage = 25 :=
by sorry

end contractor_daily_wage_l4038_403888


namespace sequence_identity_l4038_403852

def StrictlyIncreasing (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem sequence_identity (a : ℕ → ℕ) 
  (h_increasing : StrictlyIncreasing a)
  (h_upper_bound : ∀ n : ℕ, a n ≤ n + 2020)
  (h_divisibility : ∀ n : ℕ, (a (n + 1)) ∣ (n^3 * (a n) - 1)) :
  ∀ n : ℕ, a n = n :=
sorry

end sequence_identity_l4038_403852


namespace f_is_odd_l4038_403882

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of being an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of a function not being identically zero
def NotIdenticallyZero (f : ℝ → ℝ) : Prop := ∃ x, f x ≠ 0

-- Main theorem
theorem f_is_odd 
  (f : ℝ → ℝ) 
  (h1 : IsEven (fun x => (x^3 - 2*x) * f x))
  (h2 : NotIdenticallyZero f) : 
  IsOdd f := by
  sorry


end f_is_odd_l4038_403882


namespace line_segment_ratio_l4038_403862

/-- Given seven points O, A, B, C, D, E, F on a straight line, with P on CD,
    prove that OP = (3a + 2d) / 5 when AP:PD = 2:3 and BP:PC = 3:4 -/
theorem line_segment_ratio (a b c d e f : ℝ) :
  let O : ℝ := 0
  let A : ℝ := a
  let B : ℝ := b
  let C : ℝ := c
  let D : ℝ := d
  let E : ℝ := e
  let F : ℝ := f
  ∀ P : ℝ,
    c ≤ P ∧ P ≤ d →
    (A - P) / (P - D) = 2 / 3 →
    (B - P) / (P - C) = 3 / 4 →
    P = (3 * a + 2 * d) / 5 :=
by sorry

end line_segment_ratio_l4038_403862


namespace fourth_term_coefficient_of_binomial_expansion_l4038_403824

theorem fourth_term_coefficient_of_binomial_expansion :
  let n : ℕ := 7
  let k : ℕ := 3
  let coef : ℕ := n.choose k * 2^k
  coef = 280 := by sorry

end fourth_term_coefficient_of_binomial_expansion_l4038_403824


namespace simplify_expression_l4038_403897

theorem simplify_expression (x : ℝ) : (3*x + 25) - (2*x - 5) = x + 30 := by
  sorry

end simplify_expression_l4038_403897


namespace batsman_average_l4038_403848

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) :
  total_innings = 25 →
  last_innings_score = 175 →
  average_increase = 6 →
  (∃ (previous_average : ℕ),
    (previous_average * (total_innings - 1) + last_innings_score) / total_innings =
    previous_average + average_increase) →
  (((total_innings - 1) * ((last_innings_score / average_increase) - total_innings) +
    last_innings_score) / total_innings) = 31 :=
by sorry

end batsman_average_l4038_403848


namespace calculate_responses_needed_l4038_403839

/-- The percentage of people who respond to a questionnaire -/
def response_rate : ℝ := 0.60

/-- The minimum number of questionnaires that should be mailed -/
def min_questionnaires : ℕ := 1250

/-- The number of responses needed -/
def responses_needed : ℕ := 750

/-- Theorem: Given the response rate and minimum number of questionnaires,
    the number of responses needed is 750 -/
theorem calculate_responses_needed : 
  ⌊response_rate * min_questionnaires⌋ = responses_needed := by
  sorry

end calculate_responses_needed_l4038_403839


namespace unit_conversions_l4038_403843

/-- Conversion factor from cubic decimeters to cubic meters -/
def cubic_dm_to_m : ℚ := 1 / 1000

/-- Conversion factor from seconds to minutes -/
def sec_to_min : ℚ := 1 / 60

/-- Conversion factor from minutes to hours -/
def min_to_hour : ℚ := 1 / 60

/-- Conversion factor from square centimeters to square decimeters -/
def sq_cm_to_sq_dm : ℚ := 1 / 100

/-- Conversion factor from milliliters to liters -/
def ml_to_l : ℚ := 1 / 1000

/-- Theorem stating the correctness of unit conversions -/
theorem unit_conversions :
  (35 * cubic_dm_to_m = 7 / 200) ∧
  (53 * sec_to_min = 53 / 60) ∧
  (5 * min_to_hour = 1 / 12) ∧
  (1 * sq_cm_to_sq_dm = 1 / 100) ∧
  (450 * ml_to_l = 9 / 20) := by
  sorry

end unit_conversions_l4038_403843


namespace equal_milk_water_ratio_l4038_403829

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- The ratio of two quantities -/
def ratio (a b : ℚ) : ℚ := a / b

/-- Mixture p with milk to water ratio 5:4 -/
def mixture_p : Mixture := { milk := 5, water := 4 }

/-- Mixture q with milk to water ratio 2:7 -/
def mixture_q : Mixture := { milk := 2, water := 7 }

/-- Combine two mixtures in a given ratio -/
def combine_mixtures (m1 m2 : Mixture) (r : ℚ) : Mixture :=
  { milk := m1.milk * r + m2.milk,
    water := m1.water * r + m2.water }

/-- Theorem stating that mixing p and q in ratio 5:1 results in equal milk and water -/
theorem equal_milk_water_ratio :
  let result := combine_mixtures mixture_p mixture_q (5/1)
  ratio result.milk result.water = 1 := by sorry

end equal_milk_water_ratio_l4038_403829


namespace population_equal_in_14_years_l4038_403821

/-- The number of years it takes for two villages' populations to be equal -/
def years_to_equal_population (initial_x initial_y decline_rate_x growth_rate_y : ℕ) : ℕ :=
  (initial_x - initial_y) / (decline_rate_x + growth_rate_y)

/-- Theorem stating that it takes 14 years for the populations to be equal -/
theorem population_equal_in_14_years :
  years_to_equal_population 70000 42000 1200 800 = 14 := by
  sorry

#eval years_to_equal_population 70000 42000 1200 800

end population_equal_in_14_years_l4038_403821


namespace function_condition_implies_a_range_l4038_403866

/-- Given a function f and a positive real number a, proves that if the given condition holds, then a ≥ 1 -/
theorem function_condition_implies_a_range (a : ℝ) (h_a : a > 0) :
  (∀ (x : ℝ), x > 0 → ∃ (f : ℝ → ℝ), f x = a * Real.log x + (1/2) * x^2) →
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
    (a * Real.log x₁ + (1/2) * x₁^2 - (a * Real.log x₂ + (1/2) * x₂^2)) / (x₁ - x₂) ≥ 2) →
  a ≥ 1 := by
  sorry

end function_condition_implies_a_range_l4038_403866


namespace mollys_age_problem_l4038_403871

theorem mollys_age_problem (current_age : ℕ) (years_ahead : ℕ) (multiplier : ℕ) : 
  current_age = 12 →
  years_ahead = 18 →
  multiplier = 5 →
  ∃ (years_ago : ℕ), current_age + years_ahead = multiplier * (current_age - years_ago) ∧ years_ago = 6 := by
  sorry

end mollys_age_problem_l4038_403871


namespace line_point_z_coordinate_l4038_403899

/-- Given a line passing through two points in 3D space, 
    find the z-coordinate of a point on the line with a specific x-coordinate. -/
theorem line_point_z_coordinate 
  (p1 : ℝ × ℝ × ℝ) 
  (p2 : ℝ × ℝ × ℝ) 
  (x : ℝ) 
  (h1 : p1 = (1, 3, 2)) 
  (h2 : p2 = (4, 2, -1)) 
  (h3 : x = 7) : 
  ∃ (y z : ℝ), (∃ (t : ℝ), 
    (1 + 3*t, 3 - t, 2 - 3*t) = (x, y, z)) ∧ z = -4 :=
sorry

end line_point_z_coordinate_l4038_403899


namespace banana_arrangement_count_l4038_403860

/-- The number of distinct arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of occurrences of the letter 'A' in "BANANA" -/
def a_count : ℕ := 3

/-- The number of occurrences of the letter 'N' in "BANANA" -/
def n_count : ℕ := 2

/-- The number of occurrences of the letter 'B' in "BANANA" -/
def b_count : ℕ := 1

theorem banana_arrangement_count :
  banana_arrangements = (Nat.factorial total_letters) / ((Nat.factorial a_count) * (Nat.factorial n_count)) :=
by sorry

end banana_arrangement_count_l4038_403860


namespace sisters_age_when_kolya_was_her_current_age_l4038_403809

/- Define the current ages of the brother, sister, and Kolya -/
variable (x y k : ℕ)

/- Define the time differences -/
variable (t₁ t₂ : ℕ)

/- First condition: When Kolya was as old as they both are now, the sister was as old as the brother is now -/
axiom condition1 : k - t₁ = x + y ∧ y - t₁ = x

/- Second condition: When Kolya was as old as the sister is now, the sister's age was to be determined -/
axiom condition2 : k - t₂ = y

/- The theorem to prove -/
theorem sisters_age_when_kolya_was_her_current_age : y - t₂ = 0 :=
sorry

end sisters_age_when_kolya_was_her_current_age_l4038_403809


namespace exactly_one_shot_probability_l4038_403868

/-- The probability that exactly one person makes a shot given the probabilities of A and B making shots. -/
theorem exactly_one_shot_probability (p_a p_b : ℝ) (h_a : p_a = 0.8) (h_b : p_b = 0.6) :
  p_a * (1 - p_b) + (1 - p_a) * p_b = 0.44 := by
  sorry

end exactly_one_shot_probability_l4038_403868


namespace point_guard_footage_l4038_403810

/-- Represents the number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Represents the number of players on the basketball team -/
def num_players : ℕ := 5

/-- Represents the average number of minutes each player should get in the highlight film -/
def avg_minutes_per_player : ℕ := 2

/-- Represents the total seconds of footage for the shooting guard, small forward, power forward, and center -/
def other_players_footage : ℕ := 470

/-- Theorem stating that the point guard's footage is 130 seconds -/
theorem point_guard_footage : 
  (num_players * avg_minutes_per_player * seconds_per_minute) - other_players_footage = 130 := by
sorry

end point_guard_footage_l4038_403810


namespace triangle_expression_simplification_l4038_403804

theorem triangle_expression_simplification (a b c : ℝ) 
  (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) : 
  |a + b - c| - |b - a - c| = 2*b - 2*c := by
  sorry

end triangle_expression_simplification_l4038_403804


namespace trigonometric_values_signs_l4038_403838

theorem trigonometric_values_signs :
  (∃ x, x = Real.sin (-1000 * π / 180) ∧ x > 0) ∧
  (∃ y, y = Real.cos (-2200 * π / 180) ∧ y > 0) ∧
  (∃ z, z = Real.tan (-10) ∧ z < 0) ∧
  (∃ w, w = (Real.sin (7 * π / 10) * Real.cos π) / Real.tan (17 * π / 9) ∧ w > 0) :=
by sorry

end trigonometric_values_signs_l4038_403838


namespace audio_channel_bandwidth_l4038_403857

/-- Represents the parameters for an audio channel --/
structure AudioChannelParams where
  session_duration : ℕ  -- in minutes
  sampling_rate : ℕ     -- in Hz
  bit_depth : ℕ         -- in bits
  metadata_size : ℕ     -- in bytes
  metadata_per : ℕ      -- in kilobits of audio
  is_stereo : Bool

/-- Calculates the required bandwidth for an audio channel --/
def calculate_bandwidth (params : AudioChannelParams) : ℝ :=
  sorry

/-- Theorem stating the required bandwidth for the given audio channel parameters --/
theorem audio_channel_bandwidth 
  (params : AudioChannelParams)
  (h1 : params.session_duration = 51)
  (h2 : params.sampling_rate = 63)
  (h3 : params.bit_depth = 17)
  (h4 : params.metadata_size = 47)
  (h5 : params.metadata_per = 5)
  (h6 : params.is_stereo = true) :
  ∃ (ε : ℝ), ε > 0 ∧ abs (calculate_bandwidth params - 2.25) < ε :=
sorry

end audio_channel_bandwidth_l4038_403857


namespace f_geq_g_l4038_403886

/-- Given positive real numbers a, b, c, and a real number α, 
    we define functions f and g as follows:
    f(α) = abc(a^α + b^α + c^α)
    g(α) = a^(α+2)(b+c-a) + b^(α+2)(a-b+c) + c^(α+2)(a+b-c)
    This theorem states that f(α) ≥ g(α) for all real α. -/
theorem f_geq_g (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let f := fun (α : ℝ) ↦ a * b * c * (a^α + b^α + c^α)
  let g := fun (α : ℝ) ↦ a^(α+2)*(b+c-a) + b^(α+2)*(a-b+c) + c^(α+2)*(a+b-c)
  ∀ α, f α ≥ g α :=
by sorry

end f_geq_g_l4038_403886


namespace equal_powers_implies_equality_l4038_403845

theorem equal_powers_implies_equality (a b : ℝ) : 
  0 < a → 0 < b → a^b = b^a → a < 1 → a = b := by
sorry

end equal_powers_implies_equality_l4038_403845


namespace product_96_104_l4038_403875

theorem product_96_104 : 96 * 104 = 9984 := by
  sorry

end product_96_104_l4038_403875


namespace parabola_intersection_fixed_points_l4038_403820

/-- The parabola y^2 = 2px -/
def Parabola (p : ℝ) : Set (ℝ × ℝ) :=
  {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}

/-- The fixed point A -/
def A (t : ℝ) : ℝ × ℝ := (t, 0)

/-- The line x = -t -/
def VerticalLine (t : ℝ) : Set (ℝ × ℝ) :=
  {xy : ℝ × ℝ | xy.1 = -t}

/-- The circle with diameter MN -/
def CircleMN (M N : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {xy : ℝ × ℝ | (xy.1 - (M.1 + N.1) / 2)^2 + (xy.2 - (M.2 + N.2) / 2)^2 = 
    ((N.1 - M.1)^2 + (N.2 - M.2)^2) / 4}

theorem parabola_intersection_fixed_points (p t : ℝ) (hp : p > 0) (ht : t > 0) :
  ∀ (B C M N : ℝ × ℝ),
    B ∈ Parabola p → C ∈ Parabola p →
    (∃ (k : ℝ), B.1 = k * B.2 + t ∧ C.1 = k * C.2 + t) →
    M ∈ VerticalLine t → N ∈ VerticalLine t →
    (∃ (r : ℝ), M.2 = r * M.1 ∧ B.2 = r * B.1) →
    (∃ (s : ℝ), N.2 = s * N.1 ∧ C.2 = s * C.1) →
    ((-t - Real.sqrt (2 * p * t), 0) ∈ CircleMN M N) ∧
    ((-t + Real.sqrt (2 * p * t), 0) ∈ CircleMN M N) := by
  sorry

end parabola_intersection_fixed_points_l4038_403820


namespace number_of_children_l4038_403877

theorem number_of_children (pencils_per_child : ℕ) (total_pencils : ℕ) (h1 : pencils_per_child = 2) (h2 : total_pencils = 22) :
  total_pencils / pencils_per_child = 11 := by
  sorry

end number_of_children_l4038_403877


namespace existence_of_special_set_l4038_403806

theorem existence_of_special_set (n : ℕ) (hn : n ≥ 3) :
  ∃ (S : Finset ℕ),
    (Finset.card S = 2 * n) ∧
    (∀ m : ℕ, 2 ≤ m ∧ m ≤ n →
      ∃ (A : Finset ℕ),
        A ⊆ S ∧
        Finset.card A = m ∧
        2 * (A.sum id) = S.sum id) :=
  sorry

end existence_of_special_set_l4038_403806


namespace smallest_common_multiple_9_6_l4038_403846

theorem smallest_common_multiple_9_6 : ∃ n : ℕ+, (∀ m : ℕ+, 9 ∣ m ∧ 6 ∣ m → n ≤ m) ∧ 9 ∣ n ∧ 6 ∣ n := by
  sorry

end smallest_common_multiple_9_6_l4038_403846


namespace simplify_nested_roots_l4038_403864

theorem simplify_nested_roots (a : ℝ) (ha : a > 0) :
  (((a^16)^(1/5))^(1/4))^6 * (((a^16)^(1/4))^(1/5))^6 = a^(48/5) :=
sorry

end simplify_nested_roots_l4038_403864


namespace highest_point_parabola_l4038_403802

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -2 * x^2 + 28 * x + 418

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 7

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y : ℝ := parabola vertex_x

theorem highest_point_parabola :
  ∀ x : ℝ, parabola x ≤ vertex_y :=
by sorry

end highest_point_parabola_l4038_403802


namespace complex_division_simplification_l4038_403898

theorem complex_division_simplification :
  (2 - Complex.I) / (3 + 4 * Complex.I) = 2/25 - 11/25 * Complex.I := by
  sorry

end complex_division_simplification_l4038_403898


namespace perpendicular_lines_relationship_l4038_403896

-- Define a type for lines in 3D space
def Line3D := ℝ × ℝ × ℝ → Prop

-- Define perpendicularity of lines
def perpendicular (l₁ l₂ : Line3D) : Prop := sorry

-- Define parallel lines
def parallel (l₁ l₂ : Line3D) : Prop := sorry

-- Define skew lines
def skew (l₁ l₂ : Line3D) : Prop := sorry

theorem perpendicular_lines_relationship (a b c : Line3D) 
  (h1 : perpendicular a b) (h2 : perpendicular b c) :
  ¬ (parallel a c ∨ perpendicular a c ∨ skew a c) → False := by sorry

end perpendicular_lines_relationship_l4038_403896


namespace paperback_ratio_l4038_403851

/-- Represents the number and types of books Thabo owns -/
structure BookCollection where
  total : ℕ
  paperback_fiction : ℕ
  paperback_nonfiction : ℕ
  hardcover_nonfiction : ℕ

/-- The properties of Thabo's book collection -/
def thabos_books : BookCollection where
  total := 220
  paperback_fiction := 120
  paperback_nonfiction := 60
  hardcover_nonfiction := 40

/-- Theorem stating the ratio of paperback fiction to paperback nonfiction books -/
theorem paperback_ratio (b : BookCollection) 
  (h1 : b.total = 220)
  (h2 : b.paperback_fiction + b.paperback_nonfiction + b.hardcover_nonfiction = b.total)
  (h3 : b.paperback_nonfiction = b.hardcover_nonfiction + 20)
  (h4 : b.hardcover_nonfiction = 40) :
  b.paperback_fiction / b.paperback_nonfiction = 2 := by
  sorry

#check paperback_ratio thabos_books

end paperback_ratio_l4038_403851


namespace factor_expression_l4038_403889

theorem factor_expression (z : ℝ) :
  75 * z^24 + 225 * z^48 = 75 * z^24 * (1 + 3 * z^24) := by
  sorry

end factor_expression_l4038_403889


namespace dilation_result_l4038_403861

/-- Dilation of a complex number -/
def dilation (c k z : ℂ) : ℂ := c + k * (z - c)

theorem dilation_result :
  let c : ℂ := 1 - 3*I
  let k : ℂ := 3
  let z : ℂ := -2 + I
  dilation c k z = -8 + 9*I := by sorry

end dilation_result_l4038_403861


namespace binary_multiplication_theorem_l4038_403858

/-- Converts a list of binary digits to a natural number -/
def binary_to_nat (digits : List Bool) : Nat :=
  digits.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0

/-- Converts a natural number to a list of binary digits -/
def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true, false, true, true]  -- 1101101₂
  let b := [true, true, false, true]                     -- 1011₂
  let product := [true, true, true, true, false, false, true, false, false, false, true]  -- 10001001111₂
  binary_to_nat a * binary_to_nat b = binary_to_nat product := by
  sorry

#eval binary_to_nat [true, false, true, true, false, true, true]  -- Should output 109
#eval binary_to_nat [true, true, false, true]  -- Should output 11
#eval binary_to_nat [true, true, true, true, false, false, true, false, false, false, true]  -- Should output 1103

end binary_multiplication_theorem_l4038_403858


namespace circle_tangent_to_parabola_directrix_l4038_403876

/-- The value of p for which a circle (x-1)^2 + y^2 = 4 is tangent to the directrix of a parabola y^2 = 2px -/
theorem circle_tangent_to_parabola_directrix (p : ℝ) : 
  p > 0 → 
  (∃ (x y : ℝ), (x - 1)^2 + y^2 = 4 ∧ y^2 = 2*p*x) →
  (∀ (x y : ℝ), (x - 1)^2 + y^2 = 4 → x ≥ -p/2) →
  (∃ (x y : ℝ), (x - 1)^2 + y^2 = 4 ∧ x = -p/2) →
  p = 2 :=
by sorry

end circle_tangent_to_parabola_directrix_l4038_403876


namespace largest_N_equals_n_l4038_403803

theorem largest_N_equals_n (n : ℕ) (hn : n ≥ 2) :
  ∃ N : ℕ, N > 0 ∧
  (∀ M : ℕ, M > N →
    ¬∃ (a : ℕ → ℝ), a 0 + a 1 = -1 / n ∧
    ∀ k : ℕ, 1 ≤ k ∧ k ≤ M - 1 →
      (a k + a (k - 1)) * (a k + a (k + 1)) = a (k - 1) - a (k + 1)) ∧
  (∃ (a : ℕ → ℝ), a 0 + a 1 = -1 / n ∧
    ∀ k : ℕ, 1 ≤ k ∧ k ≤ N - 1 →
      (a k + a (k - 1)) * (a k + a (k + 1)) = a (k - 1) - a (k + 1)) ∧
  N = n :=
sorry

end largest_N_equals_n_l4038_403803


namespace curve_C_symmetry_l4038_403855

/-- The curve C is defined by the equation x^2*y + x*y^2 = 1 --/
def C (x y : ℝ) : Prop := x^2*y + x*y^2 = 1

/-- A point (x, y) is symmetric to (a, b) with respect to the line y=x --/
def symmetric_y_eq_x (x y a b : ℝ) : Prop := x = b ∧ y = a

theorem curve_C_symmetry :
  (∀ x y : ℝ, C x y → C y x) ∧ 
  (∃ x y : ℝ, C x y ∧ ¬C x (-y)) ∧ 
  (∃ x y : ℝ, C x y ∧ ¬C (-x) y) ∧ 
  (∃ x y : ℝ, C x y ∧ ¬C (-x) (-y)) ∧ 
  (∃ x y : ℝ, C x y ∧ ¬C (-y) (-x)) :=
sorry

end curve_C_symmetry_l4038_403855


namespace smaller_solution_of_quadratic_l4038_403894

theorem smaller_solution_of_quadratic (x : ℝ) : 
  x^2 + 7*x - 30 = 0 ∧ (∀ y : ℝ, y^2 + 7*y - 30 = 0 → y ≥ x) → x = -10 :=
by sorry

end smaller_solution_of_quadratic_l4038_403894


namespace log_equation_proof_l4038_403830

theorem log_equation_proof : -2 * Real.log 10 / Real.log 5 - Real.log 0.25 / Real.log 5 + 2 = 0 := by
  sorry

end log_equation_proof_l4038_403830


namespace parabola_reflection_translation_sum_l4038_403831

/-- Given a parabola y = ax^2 + bx, prove that after reflecting about the y-axis
    and translating one parabola 4 units right and the other 4 units left,
    the sum of the resulting parabolas' equations is y = 2ax^2 - 8b. -/
theorem parabola_reflection_translation_sum (a b : ℝ) :
  let f (x : ℝ) := a * x^2 + b * (x - 4)
  let g (x : ℝ) := a * x^2 - b * (x + 4)
  ∀ x, (f + g) x = 2 * a * x^2 - 8 * b :=
by sorry

end parabola_reflection_translation_sum_l4038_403831


namespace g_four_to_four_l4038_403827

/-- Given two functions f and g satisfying certain conditions, prove that [g(4)]^4 = 16 -/
theorem g_four_to_four (f g : ℝ → ℝ) 
  (h1 : ∀ x ≥ 1, f (g x) = x^2)
  (h2 : ∀ x ≥ 1, g (f x) = x^4)
  (h3 : g 16 = 16) : 
  (g 4)^4 = 16 := by
sorry

end g_four_to_four_l4038_403827


namespace square_perimeter_square_perimeter_holds_l4038_403893

/-- The perimeter of a square with side length 7 meters is 28 meters. -/
theorem square_perimeter : ℝ → Prop :=
  fun side_length =>
    side_length = 7 → 4 * side_length = 28

/-- The theorem holds for the given side length. -/
theorem square_perimeter_holds : square_perimeter 7 := by
  sorry

end square_perimeter_square_perimeter_holds_l4038_403893


namespace double_plus_five_l4038_403873

theorem double_plus_five (x : ℝ) (h : x = 6) : 2 * x + 5 = 17 := by
  sorry

end double_plus_five_l4038_403873


namespace complementary_angles_ratio_l4038_403881

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary
  a / b = 3 / 2 →  -- The ratio of the angles is 3:2
  b = 36 :=  -- The smaller angle is 36°
by
  sorry

end complementary_angles_ratio_l4038_403881


namespace no_integer_solution_l4038_403854

theorem no_integer_solution : ∀ x y : ℤ, x^2 + 3*x*y - 2*y^2 ≠ 122 := by
  sorry

end no_integer_solution_l4038_403854


namespace min_value_quadratic_expression_l4038_403818

theorem min_value_quadratic_expression :
  ∀ x y : ℝ, x^2 + 6*x*y + 9*y^2 ≥ 0 :=
by sorry

end min_value_quadratic_expression_l4038_403818


namespace problem_solution_l4038_403885

theorem problem_solution : ∃ x : ℝ, 0.75 * x = x / 3 + 110 ∧ x = 264 := by
  sorry

end problem_solution_l4038_403885


namespace pencil_difference_l4038_403874

/-- The number of pencils each person has -/
structure PencilCounts where
  candy : ℕ
  caleb : ℕ
  calen : ℕ

/-- The conditions of the problem -/
def problem_conditions (p : PencilCounts) : Prop :=
  p.candy = 9 ∧
  p.calen = p.caleb + 5 ∧
  p.caleb < 2 * p.candy ∧
  p.calen - 10 = 10

/-- The theorem to be proved -/
theorem pencil_difference (p : PencilCounts) 
  (h : problem_conditions p) : 2 * p.candy - p.caleb = 3 := by
  sorry


end pencil_difference_l4038_403874


namespace baking_scoop_size_l4038_403844

theorem baking_scoop_size (total_ingredients : ℚ) (num_scoops : ℕ) (scoop_size : ℚ) :
  total_ingredients = 3.75 ∧ num_scoops = 15 ∧ total_ingredients = num_scoops * scoop_size →
  scoop_size = 1/4 := by
  sorry

end baking_scoop_size_l4038_403844


namespace harmonic_series_inequality_l4038_403884

/-- The harmonic series function -/
def f (n : ℕ) : ℚ := (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

/-- The main theorem: f(2^n) > (n+2)/2 for all n ≥ 1 -/
theorem harmonic_series_inequality (n : ℕ) (h : n ≥ 1) : f (2^n) > (n + 2 : ℚ) / 2 := by
  sorry

end harmonic_series_inequality_l4038_403884


namespace three_million_squared_l4038_403835

theorem three_million_squared :
  (3000000 : ℕ) * 3000000 = 9000000000000 := by
  sorry

end three_million_squared_l4038_403835


namespace ellipse_intersection_midpoints_line_slope_l4038_403822

/-- Definition of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of the parallel lines -/
def parallel_line (x y m : ℝ) : Prop := y = (1/4) * x + m

/-- Definition of a point being the midpoint of two other points -/
def is_midpoint (x y x1 y1 x2 y2 : ℝ) : Prop :=
  x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2

/-- The main theorem -/
theorem ellipse_intersection_midpoints_line_slope :
  ∀ (l : ℝ → ℝ),
  (∀ x y m x1 y1 x2 y2 : ℝ,
    ellipse x1 y1 ∧ ellipse x2 y2 ∧
    parallel_line x1 y1 m ∧ parallel_line x2 y2 m ∧
    is_midpoint x y x1 y1 x2 y2 →
    y = l x) →
  ∃ k, ∀ x, l x = -2 * x + k :=
sorry

end ellipse_intersection_midpoints_line_slope_l4038_403822


namespace arithmetic_sequence_sum_l4038_403837

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∃ d, ∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 + a 2 = 1 →
  a 3 + a 4 = 4 →
  a 4 + a 5 = 8 :=
by
  sorry

end arithmetic_sequence_sum_l4038_403837


namespace laura_charge_account_l4038_403832

/-- Represents the simple interest calculation for a charge account -/
def simple_interest_charge_account (principal : ℝ) (interest_rate : ℝ) (time : ℝ) (total_owed : ℝ) : Prop :=
  total_owed = principal + (principal * interest_rate * time)

theorem laura_charge_account :
  ∀ (principal : ℝ),
    simple_interest_charge_account principal 0.05 1 36.75 →
    principal = 35 := by
  sorry

end laura_charge_account_l4038_403832


namespace imaginary_part_of_z_l4038_403865

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2 * Complex.I) * z = Complex.I) : 
  z.im = 1 / 5 := by
sorry

end imaginary_part_of_z_l4038_403865


namespace shaded_area_fraction_l4038_403812

/-- An equilateral triangle ABC divided into 9 smaller equilateral triangles -/
structure TriangleABC where
  /-- The side length of the large equilateral triangle ABC -/
  side : ℝ
  /-- The side length of each smaller equilateral triangle -/
  small_side : ℝ
  /-- The number of smaller triangles that make up triangle ABC -/
  num_small_triangles : ℕ
  /-- The number of smaller triangles that are half shaded -/
  num_half_shaded : ℕ
  /-- Condition: The large triangle is divided into 9 smaller triangles -/
  h_num_small : num_small_triangles = 9
  /-- Condition: Two smaller triangles are half shaded -/
  h_num_half : num_half_shaded = 2
  /-- Condition: The side length of the large triangle is 3 times the small triangle -/
  h_side : side = 3 * small_side

/-- The shaded area is 2/9 of the total area of triangle ABC -/
theorem shaded_area_fraction (t : TriangleABC) : 
  (t.num_half_shaded : ℝ) / 2 / t.num_small_triangles = 2 / 9 := by
  sorry

end shaded_area_fraction_l4038_403812


namespace arithmetic_geometric_sequence_property_l4038_403828

-- Define the arithmetic sequence a_n
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the geometric sequence b_n
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_property
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_geometric : geometric_sequence b)
  (h_a_sum : a 1001 + a 1015 = Real.pi)
  (h_b_prod : b 6 * b 9 = 2) :
  Real.tan ((a 1 + a 2015) / (1 + b 7 * b 8)) = Real.sqrt 3 :=
by sorry

end arithmetic_geometric_sequence_property_l4038_403828


namespace tetrahedron_properties_l4038_403867

/-- Given a tetrahedron with vertices A₁, A₂, A₃, A₄ in ℝ³ -/
def A₁ : ℝ × ℝ × ℝ := (-1, -5, 2)
def A₂ : ℝ × ℝ × ℝ := (-6, 0, -3)
def A₃ : ℝ × ℝ × ℝ := (3, 6, -3)
def A₄ : ℝ × ℝ × ℝ := (-10, 6, 7)

/-- Calculate the volume of the tetrahedron -/
def tetrahedron_volume (A₁ A₂ A₃ A₄ : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Calculate the height from A₄ to face A₁A₂A₃ -/
def tetrahedron_height (A₁ A₂ A₃ A₄ : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem tetrahedron_properties :
  tetrahedron_volume A₁ A₂ A₃ A₄ = 190 ∧
  tetrahedron_height A₁ A₂ A₃ A₄ = 2 * Real.sqrt 38 := by
  sorry

end tetrahedron_properties_l4038_403867


namespace misha_earnings_l4038_403811

theorem misha_earnings (current_amount target_amount : ℕ) 
  (h1 : current_amount = 34) 
  (h2 : target_amount = 47) : 
  target_amount - current_amount = 13 := by
  sorry

end misha_earnings_l4038_403811


namespace product_a2_a6_l4038_403836

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ := S n - S (n-1)

theorem product_a2_a6 : a 2 * a 6 = 64 := by
  sorry

end product_a2_a6_l4038_403836


namespace pizza_slices_theorem_l4038_403825

/-- Given a number of pizzas and slices per pizza, calculate the total number of slices -/
def total_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  num_pizzas * slices_per_pizza

/-- Theorem: With 14 pizzas and 2 slices per pizza, the total number of slices is 28 -/
theorem pizza_slices_theorem : total_slices 14 2 = 28 := by
  sorry

end pizza_slices_theorem_l4038_403825


namespace semicircle_area_with_inscribed_rectangle_l4038_403808

theorem semicircle_area_with_inscribed_rectangle (r : ℝ) (h : r = 3 / 2) : 
  (π * r^2) / 2 = 9 * π / 8 := by
  sorry

end semicircle_area_with_inscribed_rectangle_l4038_403808


namespace apple_cost_price_l4038_403879

/-- Proves that given a selling price of 15 and a loss of 1/6th of the cost price, the cost price of the apple is 18. -/
theorem apple_cost_price (selling_price : ℚ) (loss_fraction : ℚ) : 
  selling_price = 15 ∧ loss_fraction = 1/6 → 
  ∃ (cost_price : ℚ), cost_price = 18 ∧ selling_price = cost_price * (1 - loss_fraction) :=
by sorry

end apple_cost_price_l4038_403879


namespace pollen_grain_diameter_scientific_notation_l4038_403880

theorem pollen_grain_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0000065 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 6.5 ∧ n = -6 := by
  sorry

end pollen_grain_diameter_scientific_notation_l4038_403880


namespace chess_tournament_participants_l4038_403859

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 210 → n = 21 := by
  sorry

end chess_tournament_participants_l4038_403859


namespace absolute_value_inequality_l4038_403895

theorem absolute_value_inequality (x : ℝ) :
  |((x^2 - 5*x + 4) / 3)| < 1 ↔ (5 - Real.sqrt 21) / 2 < x ∧ x < (5 + Real.sqrt 21) / 2 := by
  sorry

end absolute_value_inequality_l4038_403895


namespace square_difference_l4038_403817

theorem square_difference (n : ℕ) (h : n = 50) : n^2 - (n-1)^2 = 2*n - 1 := by
  sorry

#check square_difference

end square_difference_l4038_403817


namespace unsold_books_percentage_l4038_403892

/-- Calculates the percentage of unsold books in a bookshop -/
theorem unsold_books_percentage 
  (initial_stock : ℕ) 
  (monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales : ℕ) : 
  initial_stock = 700 →
  monday_sales = 50 →
  tuesday_sales = 82 →
  wednesday_sales = 60 →
  thursday_sales = 48 →
  friday_sales = 40 →
  (initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) / initial_stock * 100 = 60 := by
  sorry

end unsold_books_percentage_l4038_403892


namespace smallest_consecutive_number_l4038_403850

theorem smallest_consecutive_number (x : ℕ) : 
  (∃ (a b c d : ℕ), x + a + b + c + d = 225 ∧ 
   a = x + 1 ∧ b = x + 2 ∧ c = x + 3 ∧ d = x + 4 ∧
   ∃ (k : ℕ), x = 7 * k) → 
  x = 42 := by
sorry

end smallest_consecutive_number_l4038_403850


namespace anne_twice_sister_height_l4038_403800

/-- Represents the heights of Anne, her sister, and Bella -/
structure Heights where
  anne : ℝ
  sister : ℝ
  bella : ℝ

/-- The conditions of the problem -/
def HeightConditions (h : Heights) : Prop :=
  ∃ (n : ℝ),
    h.anne = n * h.sister ∧
    h.bella = 3 * h.anne ∧
    h.anne = 80 ∧
    h.bella - h.sister = 200

/-- The theorem stating that under the given conditions, 
    Anne's height is twice her sister's height -/
theorem anne_twice_sister_height (h : Heights) 
  (hc : HeightConditions h) : h.anne = 2 * h.sister := by
  sorry

end anne_twice_sister_height_l4038_403800


namespace total_items_l4038_403842

def num_children : ℕ := 12
def pencils_per_child : ℕ := 5
def erasers_per_child : ℕ := 3
def skittles_per_child : ℕ := 13
def crayons_per_child : ℕ := 7

theorem total_items :
  num_children * pencils_per_child = 60 ∧
  num_children * erasers_per_child = 36 ∧
  num_children * skittles_per_child = 156 ∧
  num_children * crayons_per_child = 84 := by
  sorry

end total_items_l4038_403842


namespace driver_net_pay_rate_l4038_403853

/-- Calculates the net rate of pay for a driver given specific conditions --/
theorem driver_net_pay_rate (hours : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (pay_per_mile : ℝ) (gas_price : ℝ) :
  hours = 3 →
  speed = 50 →
  fuel_efficiency = 25 →
  pay_per_mile = 0.60 →
  gas_price = 2.50 →
  let distance := hours * speed
  let gas_used := distance / fuel_efficiency
  let earnings := distance * pay_per_mile
  let gas_cost := gas_used * gas_price
  let net_earnings := earnings - gas_cost
  net_earnings / hours = 25 := by
  sorry

end driver_net_pay_rate_l4038_403853


namespace todd_money_left_l4038_403883

def initial_amount : ℕ := 20
def candy_bars : ℕ := 4
def cost_per_bar : ℕ := 2

theorem todd_money_left : 
  initial_amount - (candy_bars * cost_per_bar) = 12 := by sorry

end todd_money_left_l4038_403883


namespace length_of_AF_l4038_403814

/-- Given a plot ABCD with known dimensions, prove the length of AF --/
theorem length_of_AF (CE ED AE : ℝ) (area_ABCD : ℝ) 
  (h1 : CE = 40)
  (h2 : ED = 50)
  (h3 : AE = 120)
  (h4 : area_ABCD = 7200) :
  ∃ AF : ℝ, AF = 128 := by
  sorry

end length_of_AF_l4038_403814


namespace total_cleaner_needed_l4038_403801

def cleaner_per_dog : ℕ := 6
def cleaner_per_cat : ℕ := 4
def cleaner_per_rabbit : ℕ := 1

def num_dogs : ℕ := 6
def num_cats : ℕ := 3
def num_rabbits : ℕ := 1

theorem total_cleaner_needed :
  cleaner_per_dog * num_dogs + cleaner_per_cat * num_cats + cleaner_per_rabbit * num_rabbits = 49 :=
by
  sorry

end total_cleaner_needed_l4038_403801


namespace smallest_number_satisfying_conditions_l4038_403863

theorem smallest_number_satisfying_conditions : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((y + 3).ModEq 0 7 ∧ (y - 5).ModEq 0 8)) ∧
  (x + 3).ModEq 0 7 ∧ 
  (x - 5).ModEq 0 8 := by
  sorry

end smallest_number_satisfying_conditions_l4038_403863


namespace trig_identity_l4038_403872

theorem trig_identity (x : Real) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin ((5*π)/6 - x) + (Real.cos ((π/3) - x))^2 = 5/16 := by
  sorry

end trig_identity_l4038_403872


namespace problem_proof_l4038_403823

theorem problem_proof : -1^2023 + (-8) / (-4) - |(-5)| = -4 := by
  sorry

end problem_proof_l4038_403823


namespace trigonometric_identity_l4038_403826

theorem trigonometric_identity (α : Real) : 
  (2 * Real.tan (π / 4 - α)) / (1 - Real.tan (π / 4 - α)^2) * 
  (Real.sin α * Real.cos α) / (Real.cos α^2 - Real.sin α^2) = 1 / 2 := by
  sorry

end trigonometric_identity_l4038_403826


namespace subset_of_intersection_eq_union_l4038_403869

theorem subset_of_intersection_eq_union {A B C : Set α} 
  (hA : A.Nonempty) (hB : B.Nonempty) (hC : C.Nonempty) 
  (h : A ∩ B = B ∪ C) : C ⊆ B := by
  sorry

end subset_of_intersection_eq_union_l4038_403869


namespace sqrt_15_minus_1_range_l4038_403891

theorem sqrt_15_minus_1_range : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := by
  have h1 : 9 < 15 := by sorry
  have h2 : 15 < 16 := by sorry
  sorry

end sqrt_15_minus_1_range_l4038_403891


namespace odd_power_minus_self_div_24_l4038_403819

theorem odd_power_minus_self_div_24 (n : ℕ) (h : Odd n) (h' : n > 0) :
  ∃ k : ℤ, (n^n : ℤ) - n = 24 * k := by
  sorry

end odd_power_minus_self_div_24_l4038_403819


namespace triangle_problem_l4038_403847

noncomputable section

/-- Represents a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector m in the problem -/
def m (t : Triangle) : ℝ × ℝ := (Real.cos t.A, Real.sin t.A)

/-- Vector n in the problem -/
def n (t : Triangle) : ℝ × ℝ := (Real.cos t.A, -Real.sin t.A)

/-- Dot product of vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The theorem to be proved -/
theorem triangle_problem (t : Triangle) 
    (h_acute : 0 < t.A ∧ t.A < π / 2)
    (h_dot : dot_product (m t) (n t) = 1 / 2)
    (h_a : t.a = Real.sqrt 5) :
  t.A = π / 6 ∧ 
  Real.arccos (dot_product (m t) (n t) / (Real.sqrt ((m t).1^2 + (m t).2^2) * Real.sqrt ((n t).1^2 + (n t).2^2))) = π / 3 ∧
  (let max_area := (10 + 5 * Real.sqrt 3) / 4
   ∀ b c, t.b = b → t.c = c → 1 / 2 * b * c * Real.sin t.A ≤ max_area) :=
sorry

end triangle_problem_l4038_403847


namespace ali_baba_max_coins_l4038_403849

/-- Represents the state of the coin distribution game -/
structure GameState :=
  (piles : List Nat)
  (total_coins : Nat)

/-- Represents a move in the game -/
structure Move :=
  (chosen_piles : List Nat)
  (coins_removed : List Nat)

/-- Ali Baba's strategy -/
def aliBabaStrategy (state : GameState) : Move :=
  sorry

/-- Thief's strategy -/
def thiefStrategy (state : GameState) (move : Move) : List Nat :=
  sorry

/-- Simulate one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Check if the game should end -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Calculate Ali Baba's final score -/
def calculateScore (state : GameState) : Nat :=
  sorry

/-- Main theorem: Ali Baba can secure at most 72 coins -/
theorem ali_baba_max_coins :
  ∀ (initial_state : GameState),
    initial_state.total_coins = 100 ∧ 
    initial_state.piles.length = 10 ∧ 
    (∀ pile ∈ initial_state.piles, pile = 10) →
    calculateScore (playRound initial_state) ≤ 72 :=
  sorry

end ali_baba_max_coins_l4038_403849


namespace no_extreme_points_iff_m_in_range_l4038_403841

/-- A cubic function parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + x + 2023

/-- The derivative of f with respect to x -/
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + 1

/-- Predicate for f having no extreme points -/
def has_no_extreme_points (m : ℝ) : Prop :=
  ∀ x : ℝ, f_derivative m x ≠ 0 ∨ 
    (∀ y : ℝ, y < x → f_derivative m y > 0) ∧ 
    (∀ y : ℝ, y > x → f_derivative m y > 0)

theorem no_extreme_points_iff_m_in_range :
  ∀ m : ℝ, has_no_extreme_points m ↔ -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 :=
sorry

end no_extreme_points_iff_m_in_range_l4038_403841


namespace inequality_equivalence_l4038_403887

theorem inequality_equivalence (x y : ℝ) : 
  (y - 2*x < Real.sqrt (4*x^2 - 4*x + 1)) ↔ 
  ((x < 1/2 ∧ y < 1) ∨ (x ≥ 1/2 ∧ y < 4*x - 1)) :=
by sorry

end inequality_equivalence_l4038_403887


namespace average_study_time_difference_l4038_403816

-- Define the list of daily differences
def daily_differences : List Int := [15, -5, 25, 0, -15, 10, 20]

-- Define the number of days
def num_days : Nat := 7

-- Theorem to prove
theorem average_study_time_difference :
  (daily_differences.sum : ℚ) / num_days = 50 / 7 := by
  sorry

end average_study_time_difference_l4038_403816


namespace f_not_monotonic_iff_k_in_range_l4038_403815

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the property of being not monotonic on an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- Theorem statement
theorem f_not_monotonic_iff_k_in_range (k : ℝ) :
  not_monotonic f (k - 1) (k + 1) ↔ (-3 < k ∧ k < -1) ∨ (1 < k ∧ k < 3) :=
sorry

end f_not_monotonic_iff_k_in_range_l4038_403815


namespace share_price_calculation_l4038_403890

/-- Proves the price of shares given dividend rate, face value, and return on investment -/
theorem share_price_calculation (dividend_rate : ℝ) (face_value : ℝ) (roi : ℝ) : 
  dividend_rate = 0.125 →
  face_value = 60 →
  roi = 0.25 →
  dividend_rate * face_value = roi * (face_value * dividend_rate / roi) := by
  sorry

#check share_price_calculation

end share_price_calculation_l4038_403890


namespace triangle_ratio_l4038_403807

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  A = π / 3 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end triangle_ratio_l4038_403807


namespace unique_solution_condition_l4038_403833

theorem unique_solution_condition (a b c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = c * x + b) ↔ c ≠ 4 := by
  sorry

end unique_solution_condition_l4038_403833


namespace expression_equals_two_l4038_403856

theorem expression_equals_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt (a * b / 2) + Real.sqrt 8) / Real.sqrt ((a * b + 16) / 8 + Real.sqrt (a * b)) = 2 := by
  sorry

end expression_equals_two_l4038_403856


namespace average_age_increase_l4038_403870

/-- Proves that adding a 28-year-old student to a class of 9 students with an average age of 8 years increases the overall average age by 2 years -/
theorem average_age_increase (total_students : ℕ) (initial_students : ℕ) (initial_average : ℝ) (new_student_age : ℕ) :
  total_students = 10 →
  initial_students = 9 →
  initial_average = 8 →
  new_student_age = 28 →
  (initial_students * initial_average + new_student_age) / total_students - initial_average = 2 := by
  sorry

end average_age_increase_l4038_403870


namespace multiple_of_six_l4038_403813

theorem multiple_of_six (n : ℤ) : 
  (∃ k : ℤ, n = 6 * k) → (∃ m : ℤ, n = 2 * m) ∧ (∃ p : ℤ, n = 3 * p) := by
  sorry

end multiple_of_six_l4038_403813


namespace max_area_four_squares_l4038_403878

/-- The maximum area covered by 4 squares with side length 2 when arranged to form a larger square -/
theorem max_area_four_squares (n : ℕ) (side_length : ℝ) (h1 : n = 4) (h2 : side_length = 2) :
  n * side_length^2 - (n - 1) = 13 :=
sorry

end max_area_four_squares_l4038_403878


namespace pony_jeans_discount_rate_l4038_403840

theorem pony_jeans_discount_rate
  (fox_price : ℝ)
  (pony_price : ℝ)
  (total_savings : ℝ)
  (fox_quantity : ℕ)
  (pony_quantity : ℕ)
  (total_discount_rate : ℝ)
  (h1 : fox_price = 15)
  (h2 : pony_price = 18)
  (h3 : total_savings = 8.64)
  (h4 : fox_quantity = 3)
  (h5 : pony_quantity = 2)
  (h6 : total_discount_rate = 22) :
  ∃ (fox_discount : ℝ) (pony_discount : ℝ),
    fox_discount + pony_discount = total_discount_rate ∧
    fox_quantity * fox_price * (fox_discount / 100) + pony_quantity * pony_price * (pony_discount / 100) = total_savings ∧
    pony_discount = 14 :=
by sorry

end pony_jeans_discount_rate_l4038_403840
