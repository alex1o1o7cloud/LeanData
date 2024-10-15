import Mathlib

namespace NUMINAMATH_CALUDE_ara_final_height_theorem_l1675_167501

/-- Represents the growth and heights of Shea and Ara -/
structure HeightGrowth where
  initial_height : ℝ
  shea_growth_percent : ℝ
  ara_growth_fraction : ℝ
  shea_final_height : ℝ

/-- Calculates Ara's final height based on the given conditions -/
def calculate_ara_height (hg : HeightGrowth) : ℝ :=
  let shea_growth := hg.initial_height * hg.shea_growth_percent
  let ara_growth := shea_growth * hg.ara_growth_fraction
  hg.initial_height + ara_growth

/-- Theorem stating that Ara's final height is approximately 60.67 inches -/
theorem ara_final_height_theorem (hg : HeightGrowth) 
  (h1 : hg.initial_height > 0)
  (h2 : hg.shea_growth_percent = 0.25)
  (h3 : hg.ara_growth_fraction = 1/3)
  (h4 : hg.shea_final_height = 70) :
  ∃ ε > 0, |calculate_ara_height hg - 60.67| < ε :=
sorry

end NUMINAMATH_CALUDE_ara_final_height_theorem_l1675_167501


namespace NUMINAMATH_CALUDE_sequence_sum_l1675_167521

/-- Given an 8-term sequence where C = 10 and the sum of any three consecutive terms is 40,
    prove that A + H = 30 -/
theorem sequence_sum (A B C D E F G H : ℝ) 
  (hC : C = 10)
  (hABC : A + B + C = 40)
  (hBCD : B + C + D = 40)
  (hCDE : C + D + E = 40)
  (hDEF : D + E + F = 40)
  (hEFG : E + F + G = 40)
  (hFGH : F + G + H = 40) :
  A + H = 30 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l1675_167521


namespace NUMINAMATH_CALUDE_condition_analysis_l1675_167549

theorem condition_analysis (a b c : ℝ) (h : a > b ∧ b > c) :
  (∀ a b c, a + b + c = 0 → a * b > a * c) ∧
  (∃ a b c, a * b > a * c ∧ a + b + c ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_analysis_l1675_167549


namespace NUMINAMATH_CALUDE_total_flour_used_l1675_167580

-- Define the amount of wheat flour used
def wheat_flour : ℝ := 0.2

-- Define the amount of white flour used
def white_flour : ℝ := 0.1

-- Theorem stating the total amount of flour used
theorem total_flour_used : wheat_flour + white_flour = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_total_flour_used_l1675_167580


namespace NUMINAMATH_CALUDE_gcd_90_150_l1675_167562

theorem gcd_90_150 : Nat.gcd 90 150 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_150_l1675_167562


namespace NUMINAMATH_CALUDE_johns_allowance_is_150_cents_l1675_167526

def johns_allowance (A : ℚ) : Prop :=
  let arcade_spent : ℚ := 3 / 5 * A
  let remaining_after_arcade : ℚ := A - arcade_spent
  let toy_store_spent : ℚ := 1 / 3 * remaining_after_arcade
  let remaining_after_toy_store : ℚ := remaining_after_arcade - toy_store_spent
  remaining_after_toy_store = 40 / 100

theorem johns_allowance_is_150_cents :
  ∃ A : ℚ, johns_allowance A ∧ A = 150 / 100 :=
sorry

end NUMINAMATH_CALUDE_johns_allowance_is_150_cents_l1675_167526


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l1675_167516

theorem vector_difference_magnitude (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 1) 
  (h3 : ‖a + b‖ = ‖a - b‖) : 
  ‖a - b‖ = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l1675_167516


namespace NUMINAMATH_CALUDE_calculation_proof_l1675_167568

theorem calculation_proof :
  (4 + (-2)^3 * 5 - (-0.28) / 4 = -35.93) ∧
  (-1^4 - 1/6 * (2 - (-3)^2) = 1/6) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1675_167568


namespace NUMINAMATH_CALUDE_bill_drew_eight_squares_l1675_167500

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- The number of triangles Bill drew -/
def num_triangles : ℕ := 12

/-- The number of pentagons Bill drew -/
def num_pentagons : ℕ := 4

/-- The total number of lines Bill drew -/
def total_lines : ℕ := 88

/-- Theorem: Bill drew 8 squares -/
theorem bill_drew_eight_squares :
  ∃ (num_squares : ℕ),
    num_squares * square_sides + 
    num_triangles * triangle_sides + 
    num_pentagons * pentagon_sides = total_lines ∧
    num_squares = 8 := by
  sorry

end NUMINAMATH_CALUDE_bill_drew_eight_squares_l1675_167500


namespace NUMINAMATH_CALUDE_square_equation_solution_l1675_167566

theorem square_equation_solution : ∃! (M : ℕ), M > 0 ∧ 14^2 * 35^2 = 70^2 * M^2 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l1675_167566


namespace NUMINAMATH_CALUDE_square_of_sum_equality_l1675_167505

theorem square_of_sum_equality : 31^2 + 2*(31)*(5 + 3) + (5 + 3)^2 = 1521 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_equality_l1675_167505


namespace NUMINAMATH_CALUDE_cake_mix_buyers_l1675_167522

/-- Proof of the number of buyers purchasing cake mix -/
theorem cake_mix_buyers (total : ℕ) (muffin : ℕ) (both : ℕ) (neither_prob : ℚ) 
  (h1 : total = 100)
  (h2 : muffin = 40)
  (h3 : both = 17)
  (h4 : neither_prob = 27 / 100) : 
  ∃ cake : ℕ, cake = 50 ∧ 
    cake + muffin - both = total - (neither_prob * total).num := by
  sorry

end NUMINAMATH_CALUDE_cake_mix_buyers_l1675_167522


namespace NUMINAMATH_CALUDE_max_m_inequality_l1675_167590

theorem max_m_inequality (m : ℝ) : 
  (∀ a b : ℝ, (a / Real.exp a - b)^2 ≥ m - (a - b + 3)^2) → m ≤ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_max_m_inequality_l1675_167590


namespace NUMINAMATH_CALUDE_tour_group_size_tour_group_size_exists_l1675_167503

/-- Represents the possible solutions for the number of people in the tour group -/
inductive TourGroupSize
  | eight : TourGroupSize
  | thirteen : TourGroupSize

/-- Checks if a given number of adults and children satisfies the ticket price conditions -/
def validTicketCombination (adults : ℕ) (children : ℕ) : Prop :=
  8 * adults + 3 * children = 44

/-- Calculates the total number of people in the tour group -/
def groupSize (adults : ℕ) (children : ℕ) : ℕ :=
  adults + children

/-- Theorem stating that the only valid tour group sizes are 8 or 13 -/
theorem tour_group_size :
  ∀ (adults children : ℕ),
    validTicketCombination adults children →
    (groupSize adults children = 8 ∨ groupSize adults children = 13) :=
by sorry

/-- Theorem stating that both 8 and 13 are possible tour group sizes -/
theorem tour_group_size_exists :
  (∃ (adults children : ℕ), validTicketCombination adults children ∧ groupSize adults children = 8) ∧
  (∃ (adults children : ℕ), validTicketCombination adults children ∧ groupSize adults children = 13) :=
by sorry

end NUMINAMATH_CALUDE_tour_group_size_tour_group_size_exists_l1675_167503


namespace NUMINAMATH_CALUDE_inequality_sum_l1675_167519

theorem inequality_sum (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : x < a) (h2 : y < b) : x + y < a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_sum_l1675_167519


namespace NUMINAMATH_CALUDE_stock_price_increase_probability_l1675_167564

/-- Probability of stock price increase given interest rate conditions -/
theorem stock_price_increase_probability
  (p_increase_when_lowered : ℝ)
  (p_increase_when_unchanged : ℝ)
  (p_increase_when_raised : ℝ)
  (p_rate_reduction : ℝ)
  (p_rate_unchanged : ℝ)
  (h1 : p_increase_when_lowered = 0.7)
  (h2 : p_increase_when_unchanged = 0.2)
  (h3 : p_increase_when_raised = 0.1)
  (h4 : p_rate_reduction = 0.6)
  (h5 : p_rate_unchanged = 0.3)
  (h6 : p_rate_reduction + p_rate_unchanged + (1 - p_rate_reduction - p_rate_unchanged) = 1) :
  p_rate_reduction * p_increase_when_lowered +
  p_rate_unchanged * p_increase_when_unchanged +
  (1 - p_rate_reduction - p_rate_unchanged) * p_increase_when_raised = 0.49 := by
  sorry


end NUMINAMATH_CALUDE_stock_price_increase_probability_l1675_167564


namespace NUMINAMATH_CALUDE_intersection_difference_l1675_167504

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 3 * x + 5

-- Define the intersection points
def intersection_points : Set ℝ := {x : ℝ | parabola1 x = parabola2 x}

-- Theorem statement
theorem intersection_difference : 
  ∃ (p r : ℝ), p ∈ intersection_points ∧ r ∈ intersection_points ∧ 
  r ≥ p ∧ ∀ x ∈ intersection_points, (x = p ∨ x = r) ∧ 
  r - p = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_difference_l1675_167504


namespace NUMINAMATH_CALUDE_arithmetic_proof_l1675_167558

theorem arithmetic_proof : (1) - 2^3 / (-1/5) - 1/2 * (-4)^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l1675_167558


namespace NUMINAMATH_CALUDE_parabola_shift_left_2_l1675_167524

/-- Represents a parabola in the form y = (x - h)^2 + k, where (h, k) is the vertex -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The equation of a parabola given x -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  (x - p.h)^2 + p.k

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (shift : ℝ) : Parabola :=
  { h := p.h + shift, k := p.k }

theorem parabola_shift_left_2 :
  let original := Parabola.mk 0 0
  let shifted := shift_parabola original (-2)
  ∀ x, parabola_equation shifted x = (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_left_2_l1675_167524


namespace NUMINAMATH_CALUDE_jesses_friends_bananas_l1675_167570

/-- The total number of bananas given the number of friends and bananas per friend -/
def total_bananas (num_friends : ℝ) (bananas_per_friend : ℝ) : ℝ :=
  num_friends * bananas_per_friend

/-- Theorem: Jesse's friends have 63.0 bananas in total -/
theorem jesses_friends_bananas :
  total_bananas 3.0 21.0 = 63.0 := by
  sorry

end NUMINAMATH_CALUDE_jesses_friends_bananas_l1675_167570


namespace NUMINAMATH_CALUDE_gcd_30_problem_l1675_167510

theorem gcd_30_problem (n : ℕ) : 
  70 ≤ n ∧ n ≤ 90 → Nat.gcd 30 n = 10 → n = 70 ∨ n = 80 := by sorry

end NUMINAMATH_CALUDE_gcd_30_problem_l1675_167510


namespace NUMINAMATH_CALUDE_abrahams_a_students_l1675_167515

theorem abrahams_a_students (total_students : ℕ) (total_a_students : ℕ) (abraham_students : ℕ) :
  total_students = 40 →
  total_a_students = 25 →
  abraham_students = 10 →
  (abraham_students : ℚ) / total_students * total_a_students = (abraham_students : ℕ) →
  ∃ (abraham_a_students : ℕ), 
    (abraham_a_students : ℚ) / abraham_students = (total_a_students : ℚ) / total_students ∧
    abraham_a_students = 6 :=
by sorry

end NUMINAMATH_CALUDE_abrahams_a_students_l1675_167515


namespace NUMINAMATH_CALUDE_function_shift_l1675_167555

theorem function_shift (f : ℝ → ℝ) :
  (∀ x : ℝ, f (x + 1) = x^2 - 2*x - 3) →
  (∀ x : ℝ, f x = x^2 - 4*x) := by
  sorry

end NUMINAMATH_CALUDE_function_shift_l1675_167555


namespace NUMINAMATH_CALUDE_girl_speed_l1675_167586

/-- Given a girl traveling a distance of 96 meters in 16 seconds,
    prove that her speed is 6 meters per second. -/
theorem girl_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 96) 
    (h2 : time = 16) 
    (h3 : speed = distance / time) : 
  speed = 6 := by
  sorry

end NUMINAMATH_CALUDE_girl_speed_l1675_167586


namespace NUMINAMATH_CALUDE_longest_all_green_interval_is_20_seconds_l1675_167585

/-- Represents a traffic light with its timing properties -/
structure TrafficLight where
  greenDuration : ℝ
  yellowDuration : ℝ
  redDuration : ℝ
  cycleStart : ℝ

/-- Calculates the longest interval during which all lights are green -/
def longestAllGreenInterval (lights : List TrafficLight) : ℝ :=
  sorry

/-- The main theorem stating the longest interval of all green lights -/
theorem longest_all_green_interval_is_20_seconds :
  let lights : List TrafficLight := List.range 8 |>.map (fun i =>
    { greenDuration := 90  -- 1.5 minutes in seconds
      yellowDuration := 3
      redDuration := 90    -- 1.5 minutes in seconds
      cycleStart := i * 10 -- Each light starts 10 seconds after the previous
    })
  longestAllGreenInterval lights = 20 := by
  sorry

end NUMINAMATH_CALUDE_longest_all_green_interval_is_20_seconds_l1675_167585


namespace NUMINAMATH_CALUDE_coloring_exists_l1675_167582

def M : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2010}

theorem coloring_exists :
  ∃ (f : ℕ → Fin 5),
    ∀ (a d : ℕ),
      a ∈ M →
      d > 0 →
      (∀ k, k ∈ Finset.range 9 → (a + k * d) ∈ M) →
      ∃ (i j : Fin 9), i ≠ j ∧ f (a + i * d) ≠ f (a + j * d) :=
by sorry

end NUMINAMATH_CALUDE_coloring_exists_l1675_167582


namespace NUMINAMATH_CALUDE_three_colors_sufficient_and_necessary_l1675_167528

/-- A function that returns the minimum number of colors needed to uniquely identify n keys on a single keychain. -/
def min_colors (n : ℕ) : ℕ :=
  if n ≤ 2 then n else 3

/-- Theorem stating that for n ≥ 3 keys on a single keychain, 3 colors are sufficient and necessary to uniquely identify each key. -/
theorem three_colors_sufficient_and_necessary (n : ℕ) (h : n ≥ 3) :
  min_colors n = 3 := by sorry

end NUMINAMATH_CALUDE_three_colors_sufficient_and_necessary_l1675_167528


namespace NUMINAMATH_CALUDE_sprocket_production_rate_l1675_167596

theorem sprocket_production_rate : ∀ (a g : ℝ),
  -- Machine G produces 10% more sprockets per hour than Machine A
  g = 1.1 * a →
  -- Machine A takes 10 hours longer to produce 660 sprockets
  660 / a = 660 / g + 10 →
  -- The production rate of Machine A
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_sprocket_production_rate_l1675_167596


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1675_167560

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1675_167560


namespace NUMINAMATH_CALUDE_martha_clothes_count_l1675_167554

def jacket_ratio : ℕ := 2
def tshirt_ratio : ℕ := 3
def jackets_bought : ℕ := 4
def tshirts_bought : ℕ := 9

def total_clothes : ℕ := 
  (jackets_bought + jackets_bought / jacket_ratio) + 
  (tshirts_bought + tshirts_bought / tshirt_ratio)

theorem martha_clothes_count : total_clothes = 18 := by
  sorry

end NUMINAMATH_CALUDE_martha_clothes_count_l1675_167554


namespace NUMINAMATH_CALUDE_problem_statement_l1675_167592

theorem problem_statement (x₁ x₂ x₃ a : ℝ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁)
  (h_nonzero : x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₃ ≠ 0)
  (h_eq1 : x₁ * x₂ - a * x₁ + a^2 = 0)
  (h_eq2 : x₂ * x₃ - a * x₂ + a^2 = 0) :
  (x₃ * x₁ - a * x₃ + a^2 = 0) ∧
  (x₁ * x₂ * x₃ + a^3 = 0) ∧
  (1 / (x₁ - x₂) + 1 / (x₂ - x₃) + 1 / (x₃ - x₁) = 1 / a) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1675_167592


namespace NUMINAMATH_CALUDE_cycle_reappearance_l1675_167589

theorem cycle_reappearance (letter_seq_length digit_seq_length : ℕ) 
  (h1 : letter_seq_length = 9)
  (h2 : digit_seq_length = 4) : 
  Nat.lcm letter_seq_length digit_seq_length = 36 := by
  sorry

end NUMINAMATH_CALUDE_cycle_reappearance_l1675_167589


namespace NUMINAMATH_CALUDE_base_7_2534_equals_956_l1675_167506

def base_7_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_7_2534_equals_956 :
  base_7_to_10 [4, 3, 5, 2] = 956 := by
  sorry

end NUMINAMATH_CALUDE_base_7_2534_equals_956_l1675_167506


namespace NUMINAMATH_CALUDE_february_first_day_l1675_167531

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

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

-- Define a function to advance a given number of days
def advanceDays (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | n + 1 => nextDay (advanceDays start n)

-- Theorem statement
theorem february_first_day (january_first : DayOfWeek) 
  (h : january_first = DayOfWeek.Monday) : 
  advanceDays january_first 31 = DayOfWeek.Thursday := by
  sorry


end NUMINAMATH_CALUDE_february_first_day_l1675_167531


namespace NUMINAMATH_CALUDE_root_product_expression_l1675_167507

theorem root_product_expression (p q : ℝ) 
  (hα : ∃ α : ℝ, α^2 + p*α + 2 = 0)
  (hβ : ∃ β : ℝ, β^2 + p*β + 2 = 0)
  (hγ : ∃ γ : ℝ, γ^2 + q*γ - 3 = 0)
  (hδ : ∃ δ : ℝ, δ^2 + q*δ - 3 = 0)
  (hαβ_distinct : ∀ α β : ℝ, α^2 + p*α + 2 = 0 → β^2 + p*β + 2 = 0 → α ≠ β)
  (hγδ_distinct : ∀ γ δ : ℝ, γ^2 + q*γ - 3 = 0 → δ^2 + q*δ - 3 = 0 → γ ≠ δ) :
  ∃ (α β γ δ : ℝ), (α - γ)*(β - γ)*(α + δ)*(β + δ) = 3*(q^2 - p^2) + 15 :=
by sorry

end NUMINAMATH_CALUDE_root_product_expression_l1675_167507


namespace NUMINAMATH_CALUDE_work_completion_proof_l1675_167548

/-- The number of days A takes to complete the work alone -/
def a_days : ℝ := 15

/-- The number of days B takes to complete the work alone -/
def b_days : ℝ := 20

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.41666666666666663

/-- The number of days A and B worked together -/
def days_worked_together : ℝ := 5

theorem work_completion_proof :
  let work_rate_a := 1 / a_days
  let work_rate_b := 1 / b_days
  let combined_rate := work_rate_a + work_rate_b
  combined_rate * days_worked_together = 1 - work_left :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l1675_167548


namespace NUMINAMATH_CALUDE_parabola_circle_equation_l1675_167517

/-- The equation of a circle with center at the focus of a parabola and diameter
    equal to the line segment formed by the intersection of the parabola with a
    line perpendicular to the x-axis passing through the focus. -/
theorem parabola_circle_equation (x y : ℝ) : 
  let parabola := {(x, y) | y^2 = 4*x}
  let focus := (1, 0)
  let perpendicular_line := {(x, y) | x = 1}
  let intersection := parabola ∩ perpendicular_line
  true → (x - 1)^2 + y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_circle_equation_l1675_167517


namespace NUMINAMATH_CALUDE_julia_tag_monday_l1675_167508

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := 14

/-- The additional number of kids Julia played tag with on Monday compared to Tuesday -/
def additional_monday_kids : ℕ := 8

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := tuesday_kids + additional_monday_kids

theorem julia_tag_monday : monday_kids = 22 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_monday_l1675_167508


namespace NUMINAMATH_CALUDE_xiaomin_final_score_l1675_167567

/-- Calculates the final score for the "Book-loving Youth" selection -/
def final_score (honor_score : ℝ) (speech_score : ℝ) : ℝ :=
  0.4 * honor_score + 0.6 * speech_score

/-- Theorem: Xiaomin's final score in the "Book-loving Youth" selection is 86 points -/
theorem xiaomin_final_score :
  let honor_score := 80
  let speech_score := 90
  final_score honor_score speech_score = 86 :=
by
  sorry

end NUMINAMATH_CALUDE_xiaomin_final_score_l1675_167567


namespace NUMINAMATH_CALUDE_sqrt_225_range_l1675_167535

theorem sqrt_225_range : 15 < Real.sqrt 225 ∧ Real.sqrt 225 < 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_225_range_l1675_167535


namespace NUMINAMATH_CALUDE_sin_75_cos_15_minus_1_l1675_167583

theorem sin_75_cos_15_minus_1 : 
  2 * Real.sin (75 * π / 180) * Real.cos (15 * π / 180) - 1 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_cos_15_minus_1_l1675_167583


namespace NUMINAMATH_CALUDE_all_points_on_single_line_l1675_167536

/-- A point in a plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line. -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if three points are collinear. -/
def collinear (p q r : Point) : Prop :=
  ∃ l : Line, pointOnLine p l ∧ pointOnLine q l ∧ pointOnLine r l

/-- The main theorem. -/
theorem all_points_on_single_line (k : ℕ) (points : Fin k → Point)
    (h : ∀ i j : Fin k, i ≠ j → ∃ m : Fin k, m ≠ i ∧ m ≠ j ∧ collinear (points i) (points j) (points m)) :
    ∃ l : Line, ∀ i : Fin k, pointOnLine (points i) l := by
  sorry

end NUMINAMATH_CALUDE_all_points_on_single_line_l1675_167536


namespace NUMINAMATH_CALUDE_integral_proofs_l1675_167578

theorem integral_proofs :
  ∀ x : ℝ,
    (deriv (λ y => Real.arctan (Real.log y)) x = 1 / (x * (1 + Real.log x ^ 2))) ∧
    (deriv (λ y => Real.arctan (Real.exp y)) x = Real.exp x / (1 + Real.exp (2 * x))) ∧
    (deriv (λ y => Real.arctan (Real.sin y)) x = Real.cos x / (1 + Real.sin x ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_integral_proofs_l1675_167578


namespace NUMINAMATH_CALUDE_school_student_count_miyoung_school_students_l1675_167532

theorem school_student_count (grades classes_per_grade : ℕ) 
  (rank_from_front rank_from_back : ℕ) : ℕ :=
  let students_per_class := rank_from_front + rank_from_back - 1
  let students_per_grade := classes_per_grade * students_per_class
  let total_students := grades * students_per_grade
  total_students

theorem miyoung_school_students : 
  school_student_count 3 12 12 12 = 828 := by
  sorry

end NUMINAMATH_CALUDE_school_student_count_miyoung_school_students_l1675_167532


namespace NUMINAMATH_CALUDE_min_length_GH_l1675_167559

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the vertices A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define a point P on the ellipse above the x-axis
def P (x y : ℝ) : Prop := ellipse_C x y ∧ y > 0

-- Define the line y = 3
def line_y_3 (x y : ℝ) : Prop := y = 3

-- Define the intersection points G and H
def G (x y : ℝ) : Prop := ∃ (k : ℝ), y = k * (x + 2) ∧ line_y_3 x y
def H (x y : ℝ) : Prop := ∃ (k : ℝ), y = -1/(4*k) * (x - 2) ∧ line_y_3 x y

-- Theorem statement
theorem min_length_GH :
  ∀ (x_p y_p x_g y_g x_h y_h : ℝ),
    P x_p y_p →
    G x_g y_g →
    H x_h y_h →
    ∀ (l : ℝ), l = |x_g - x_h| →
    ∃ (min_l : ℝ), min_l = 8 ∧ l ≥ min_l :=
sorry

end NUMINAMATH_CALUDE_min_length_GH_l1675_167559


namespace NUMINAMATH_CALUDE_inverse_function_solution_l1675_167591

/-- Given a function f(x) = 1 / (2ax + 3b) where a and b are nonzero constants,
    prove that the solution to f⁻¹(x) = -1 is x = 1 / (-2a + 3b) -/
theorem inverse_function_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f : ℝ → ℝ := λ x => 1 / (2 * a * x + 3 * b)
  ∃! x, f x = -1 ∧ x = 1 / (-2 * a + 3 * b) := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_solution_l1675_167591


namespace NUMINAMATH_CALUDE_prob_two_pairs_eq_nine_twentytwo_l1675_167513

/-- Represents the number of socks of each color --/
def socks_per_color : ℕ := 3

/-- Represents the number of colors --/
def num_colors : ℕ := 4

/-- Represents the total number of socks --/
def total_socks : ℕ := socks_per_color * num_colors

/-- Represents the number of socks drawn --/
def socks_drawn : ℕ := 5

/-- Calculates the probability of drawing exactly two pairs of socks with different colors --/
def prob_two_pairs : ℚ :=
  (num_colors.choose 2 * (num_colors - 2).choose 1 * socks_per_color.choose 2 * socks_per_color.choose 2 * socks_per_color.choose 1) /
  (total_socks.choose socks_drawn)

theorem prob_two_pairs_eq_nine_twentytwo : prob_two_pairs = 9 / 22 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_pairs_eq_nine_twentytwo_l1675_167513


namespace NUMINAMATH_CALUDE_problem_statement_l1675_167598

theorem problem_statement : 2006 * ((Real.sqrt 8 - Real.sqrt 2) / Real.sqrt 2) = 2006 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1675_167598


namespace NUMINAMATH_CALUDE_nested_average_equals_two_thirds_l1675_167502

def avg2 (a b : ℚ) : ℚ := (a + b) / 2

def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem nested_average_equals_two_thirds :
  avg3 (avg3 1 2 0) (avg2 0 2) 0 = 2/3 := by sorry

end NUMINAMATH_CALUDE_nested_average_equals_two_thirds_l1675_167502


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l1675_167533

/-- Given 6 numbers with specified averages, prove the average of the remaining 2 numbers -/
theorem average_of_remaining_numbers
  (total_average : Real)
  (first_pair_average : Real)
  (second_pair_average : Real)
  (h1 : total_average = 3.95)
  (h2 : first_pair_average = 3.8)
  (h3 : second_pair_average = 3.85) :
  (6 * total_average - 2 * first_pair_average - 2 * second_pair_average) / 2 = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l1675_167533


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1675_167534

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^2 + 6 * x - 4) - (2 * x^2 + 3 * x - 15) = x^2 + 3 * x + 11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1675_167534


namespace NUMINAMATH_CALUDE_divisible_by_6_up_to_88_characterization_l1675_167544

def divisible_by_6_up_to_88 : Set ℕ :=
  {n : ℕ | 1 < n ∧ n ≤ 88 ∧ n % 6 = 0}

theorem divisible_by_6_up_to_88_characterization :
  divisible_by_6_up_to_88 = {6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84} := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_6_up_to_88_characterization_l1675_167544


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l1675_167573

/-- The number of diagonals in a convex polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex polygon with 25 sides is 275 -/
theorem diagonals_25_sided_polygon :
  numDiagonals 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l1675_167573


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l1675_167587

-- Define the points and line
def M : ℝ × ℝ := (-3, 4)
def N : ℝ × ℝ := (2, 6)
def l (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the reflection property
def is_reflection (M N M' : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), M' = (a, b) ∧
  (b - M.2) / (a - M.1) = -1 ∧
  (M.1 + a) / 2 - (b + M.2) / 2 + 3 = 0

-- Define the property of a line passing through two points
def line_through_points (P Q : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - P.2) * (Q.1 - P.1) = (x - P.1) * (Q.2 - P.2)

-- Theorem statement
theorem reflected_ray_equation 
  (h_reflection : ∃ M' : ℝ × ℝ, is_reflection M N M' l) :
  ∀ x y : ℝ, line_through_points M' N x y ↔ 6 * x - y - 6 = 0 :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l1675_167587


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l1675_167546

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 13) : 
  x^3 + y^3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l1675_167546


namespace NUMINAMATH_CALUDE_five_Y_three_equals_four_l1675_167539

-- Define the Y operation
def Y (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- Theorem statement
theorem five_Y_three_equals_four : Y 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_five_Y_three_equals_four_l1675_167539


namespace NUMINAMATH_CALUDE_total_yellow_balloons_l1675_167529

theorem total_yellow_balloons (fred sam mary : ℕ) 
  (h1 : fred = 5) 
  (h2 : sam = 6) 
  (h3 : mary = 7) : 
  fred + sam + mary = 18 := by
sorry

end NUMINAMATH_CALUDE_total_yellow_balloons_l1675_167529


namespace NUMINAMATH_CALUDE_composite_cubes_surface_area_l1675_167527

/-- Represents a composite shape formed by two cubes -/
structure CompositeCubes where
  large_cube_edge : ℝ
  small_cube_edge : ℝ

/-- Calculate the surface area of the composite shape -/
def surface_area (shape : CompositeCubes) : ℝ :=
  let large_cube_area := 6 * shape.large_cube_edge ^ 2
  let small_cube_area := 6 * shape.small_cube_edge ^ 2
  let covered_area := shape.small_cube_edge ^ 2
  let exposed_small_cube_area := 4 * shape.small_cube_edge ^ 2
  large_cube_area - covered_area + exposed_small_cube_area

/-- Theorem stating that the surface area of the specific composite shape is 49 -/
theorem composite_cubes_surface_area : 
  let shape := CompositeCubes.mk 3 1
  surface_area shape = 49 := by
  sorry

end NUMINAMATH_CALUDE_composite_cubes_surface_area_l1675_167527


namespace NUMINAMATH_CALUDE_jackson_pbj_sandwiches_l1675_167571

/-- Calculates the number of peanut butter and jelly sandwiches Jackson eats during the school year -/
def pbj_sandwiches (weeks : ℕ) (wed_holidays : ℕ) (fri_holidays : ℕ) (ham_cheese_interval : ℕ) (wed_absences : ℕ) (fri_absences : ℕ) : ℕ :=
  let total_wed := weeks
  let total_fri := weeks
  let remaining_wed := total_wed - wed_holidays - wed_absences
  let remaining_fri := total_fri - fri_holidays - fri_absences
  let ham_cheese_weeks := weeks / ham_cheese_interval
  let ham_cheese_wed := ham_cheese_weeks
  let ham_cheese_fri := ham_cheese_weeks * 2
  let pbj_wed := remaining_wed - ham_cheese_wed
  let pbj_fri := remaining_fri - ham_cheese_fri
  pbj_wed + pbj_fri

/-- Theorem stating that Jackson eats 37 peanut butter and jelly sandwiches during the school year -/
theorem jackson_pbj_sandwiches :
  pbj_sandwiches 36 2 3 4 1 2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_jackson_pbj_sandwiches_l1675_167571


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_with_0_and_8_l1675_167576

def is_multiple_of_45 (n : ℕ) : Prop := n % 45 = 0

def contains_only_0_and_8 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 8

theorem smallest_multiple_of_45_with_0_and_8 :
  ∃ (n : ℕ), is_multiple_of_45 n ∧ contains_only_0_and_8 n ∧
  (∀ m : ℕ, m < n → ¬(is_multiple_of_45 m ∧ contains_only_0_and_8 m)) ∧
  n = 8888888880 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_with_0_and_8_l1675_167576


namespace NUMINAMATH_CALUDE_max_prime_factors_l1675_167561

theorem max_prime_factors (a b : ℕ+) 
  (h_gcd : (Finset.card (Nat.primeFactors (Nat.gcd a b))) = 8)
  (h_lcm : (Finset.card (Nat.primeFactors (Nat.lcm a b))) = 30)
  (h_fewer : (Finset.card (Nat.primeFactors a)) < (Finset.card (Nat.primeFactors b))) :
  (Finset.card (Nat.primeFactors a)) ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_max_prime_factors_l1675_167561


namespace NUMINAMATH_CALUDE_set_operations_l1675_167565

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {2,4,5}
def B : Set Nat := {1,2,5}

theorem set_operations :
  (A ∩ B = {2,5}) ∧ (A ∪ (U \ B) = {2,3,4,5,6}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l1675_167565


namespace NUMINAMATH_CALUDE_rectangular_prism_painted_faces_l1675_167579

theorem rectangular_prism_painted_faces (a : ℕ) : 
  2 < a → a < 5 → (a - 2) * 3 * 4 = 4 * 3 + 4 * 4 → a = 4 := by sorry

end NUMINAMATH_CALUDE_rectangular_prism_painted_faces_l1675_167579


namespace NUMINAMATH_CALUDE_parabola_a_value_l1675_167547

/-- A parabola with equation y = ax^2 + bx + c, vertex at (3, -2), and passing through (0, -50) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ
  vertex_y : ℝ
  point_x : ℝ
  point_y : ℝ
  vertex_condition : vertex_y = a * vertex_x^2 + b * vertex_x + c
  point_condition : point_y = a * point_x^2 + b * point_x + c

/-- The theorem stating that the value of 'a' for the given parabola is -16/3 -/
theorem parabola_a_value (p : Parabola) 
  (h1 : p.vertex_x = 3) 
  (h2 : p.vertex_y = -2) 
  (h3 : p.point_x = 0) 
  (h4 : p.point_y = -50) : 
  p.a = -16/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_a_value_l1675_167547


namespace NUMINAMATH_CALUDE_intersection_sum_l1675_167552

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 6 = (y + 1)^2

-- Define the intersection points
def intersection_points : Prop := ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
  (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
  (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
  (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
  (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄)

-- Theorem statement
theorem intersection_sum : intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
  (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
  (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
  (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
  (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄) ∧
  x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l1675_167552


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_l1675_167511

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2*x*y) :
  x + 4*y ≥ 9/2 ∧ (x + 4*y = 9/2 ↔ x = 3/2 ∧ y = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_l1675_167511


namespace NUMINAMATH_CALUDE_rectangle_perimeter_area_inequality_l1675_167530

theorem rectangle_perimeter_area_inequality (l S : ℝ) (hl : l > 0) (hS : S > 0) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ l = 2 * (a + b) ∧ S = a * b) → l^2 ≥ 16 * S :=
by sorry

#check rectangle_perimeter_area_inequality

end NUMINAMATH_CALUDE_rectangle_perimeter_area_inequality_l1675_167530


namespace NUMINAMATH_CALUDE_roses_kept_l1675_167545

theorem roses_kept (total : ℕ) (to_mother : ℕ) (to_grandmother : ℕ) (to_sister : ℕ) 
  (h1 : total = 20)
  (h2 : to_mother = 6)
  (h3 : to_grandmother = 9)
  (h4 : to_sister = 4) : 
  total - (to_mother + to_grandmother + to_sister) = 1 := by
  sorry

end NUMINAMATH_CALUDE_roses_kept_l1675_167545


namespace NUMINAMATH_CALUDE_tan_150_degrees_l1675_167594

theorem tan_150_degrees : Real.tan (150 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l1675_167594


namespace NUMINAMATH_CALUDE_range_of_m_main_theorem_l1675_167512

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 + m*p.1 + 2}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 1 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ 2}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (A m ∩ B).Nonempty → m ∈ Set.Iic (-1) := by
  sorry

-- Define the range of m
def range_m : Set ℝ := {m : ℝ | ∃ x : ℝ, (A m ∩ B).Nonempty}

-- State the main theorem
theorem main_theorem : range_m = Set.Iic (-1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_main_theorem_l1675_167512


namespace NUMINAMATH_CALUDE_m_range_isosceles_perimeter_l1675_167569

-- Define the triangle ABC
structure Triangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ

-- Define the specific triangle from the problem
def triangleABC (m : ℝ) : Triangle where
  AB := 17
  BC := 8
  AC := 2 * m - 1

-- Theorem for the range of m
theorem m_range (m : ℝ) : 
  (∃ t : Triangle, t = triangleABC m ∧ t.AB + t.BC > t.AC ∧ t.AB + t.AC > t.BC ∧ t.AC + t.BC > t.AB) 
  ↔ (5 < m ∧ m < 13) := by sorry

-- Theorem for the perimeter when isosceles
theorem isosceles_perimeter (m : ℝ) :
  (∃ t : Triangle, t = triangleABC m ∧ (t.AB = t.AC ∨ t.AB = t.BC ∨ t.AC = t.BC)) →
  (∃ t : Triangle, t = triangleABC m ∧ t.AB + t.BC + t.AC = 42) := by sorry

end NUMINAMATH_CALUDE_m_range_isosceles_perimeter_l1675_167569


namespace NUMINAMATH_CALUDE_third_square_is_G_l1675_167574

-- Define the type for squares
inductive Square | A | B | C | D | E | F | G | H

-- Define the placement order
def placement_order : List Square := [Square.F, Square.H, Square.G, Square.D, Square.A, Square.B, Square.C, Square.E]

-- Define the property of being fully visible
def is_fully_visible (s : Square) : Prop := s = Square.E

-- Define the property of being partially visible
def is_partially_visible (s : Square) : Prop := s ≠ Square.E

-- Define the property of being the last placed square
def is_last_placed (s : Square) : Prop := s = Square.E

-- Theorem statement
theorem third_square_is_G :
  (∀ s : Square, s ∈ placement_order) →
  (List.length placement_order = 8) →
  (∀ s : Square, s ≠ Square.E → is_partially_visible s) →
  is_fully_visible Square.E →
  is_last_placed Square.E →
  placement_order[2] = Square.G :=
by sorry

end NUMINAMATH_CALUDE_third_square_is_G_l1675_167574


namespace NUMINAMATH_CALUDE_sal_and_phil_combined_money_l1675_167584

/-- Given that Kim has 40% more money than Sal, Sal has 20% less money than Phil,
    and Kim has $1.12, prove that Sal and Phil have a combined total of $1.80. -/
theorem sal_and_phil_combined_money :
  ∀ (kim sal phil : ℝ),
  kim = 1.4 * sal →
  sal = 0.8 * phil →
  kim = 1.12 →
  sal + phil = 1.80 :=
by
  sorry

end NUMINAMATH_CALUDE_sal_and_phil_combined_money_l1675_167584


namespace NUMINAMATH_CALUDE_unique_prime_in_set_l1675_167525

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def six_digit_number (B : ℕ) : ℕ := 303200 + B

theorem unique_prime_in_set :
  ∃! B : ℕ, B ≤ 9 ∧ is_prime (six_digit_number B) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_in_set_l1675_167525


namespace NUMINAMATH_CALUDE_vote_count_theorem_l1675_167538

theorem vote_count_theorem (votes_A votes_B : ℕ) : 
  (votes_B = (20 * votes_A) / 21) →  -- B's votes are 20/21 of A's
  (votes_A > votes_B) →  -- A wins
  (votes_B + 4 > votes_A - 4) →  -- If B gains 4 votes, B would win
  (votes_A < 168) →  -- derived from the inequality in the solution
  (∀ (x : ℕ), x < votes_A → x ≠ votes_A ∨ (20 * x) / 21 ≠ votes_B) →  -- A's vote count is minimal
  ((votes_A = 147 ∧ votes_B = 140) ∨ (votes_A = 126 ∧ votes_B = 120)) :=
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_vote_count_theorem_l1675_167538


namespace NUMINAMATH_CALUDE_multiple_births_quintuplets_l1675_167581

theorem multiple_births_quintuplets (total_babies : ℕ) 
  (triplets_to_quintuplets : ℕ → ℕ) 
  (twins_to_triplets : ℕ → ℕ) 
  (quadruplets_to_quintuplets : ℕ → ℕ) 
  (h1 : total_babies = 1540)
  (h2 : ∀ q, triplets_to_quintuplets q = 6 * q)
  (h3 : ∀ t, twins_to_triplets t = 2 * t)
  (h4 : ∀ q, quadruplets_to_quintuplets q = 3 * q)
  (h5 : ∀ q, 2 * (twins_to_triplets (triplets_to_quintuplets q)) + 
             3 * (triplets_to_quintuplets q) + 
             4 * (quadruplets_to_quintuplets q) + 
             5 * q = total_babies) : 
  ∃ q : ℚ, q = 7700 / 59 ∧ 5 * q = (quintuplets_babies : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_multiple_births_quintuplets_l1675_167581


namespace NUMINAMATH_CALUDE_karen_beats_tom_l1675_167557

theorem karen_beats_tom (karen_speed : ℝ) (tom_speed : ℝ) (karen_delay : ℝ) (winning_margin : ℝ) :
  karen_speed = 60 →
  tom_speed = 45 →
  karen_delay = 4 / 60 →
  winning_margin = 4 →
  (tom_speed * (karen_delay + (winning_margin + tom_speed * karen_delay) / (karen_speed - tom_speed))) = 21 :=
by sorry

end NUMINAMATH_CALUDE_karen_beats_tom_l1675_167557


namespace NUMINAMATH_CALUDE_cube_root_of_negative_27_l1675_167542

theorem cube_root_of_negative_27 :
  let S : Set ℂ := {z : ℂ | z^3 = -27}
  S = {-3, (3/2 : ℂ) + (3*Complex.I*Real.sqrt 3)/2, (3/2 : ℂ) - (3*Complex.I*Real.sqrt 3)/2} := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_27_l1675_167542


namespace NUMINAMATH_CALUDE_magical_stack_size_with_157_fixed_l1675_167543

/-- A stack of cards is magical if it satisfies certain conditions --/
structure MagicalStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (pile_a : Finset ℕ := Finset.range n)
  (pile_b : Finset ℕ := Finset.range n)
  (card_157_position : ℕ)
  (card_157_retains_position : card_157_position = 157)

/-- The number of cards in a magical stack where card 157 retains its position --/
def magical_stack_size (stack : MagicalStack) : ℕ := stack.total_cards

/-- Theorem: The number of cards in a magical stack where card 157 retains its position is 470 --/
theorem magical_stack_size_with_157_fixed (stack : MagicalStack) :
  magical_stack_size stack = 470 := by sorry

end NUMINAMATH_CALUDE_magical_stack_size_with_157_fixed_l1675_167543


namespace NUMINAMATH_CALUDE_square_paper_side_length_l1675_167597

/-- The length of a cube's edge in centimeters -/
def cube_edge : ℝ := 12

/-- The number of square paper pieces covering the cube -/
def num_squares : ℕ := 54

/-- The surface area of a cube given its edge length -/
def cube_surface_area (edge : ℝ) : ℝ := 6 * edge^2

/-- The theorem stating that the side length of each square paper is 4 cm -/
theorem square_paper_side_length :
  ∃ (side : ℝ),
    side > 0 ∧
    side^2 * num_squares = cube_surface_area cube_edge ∧
    side = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_paper_side_length_l1675_167597


namespace NUMINAMATH_CALUDE_z_percentage_of_1000_l1675_167537

theorem z_percentage_of_1000 (x y z : ℝ) : 
  x = (3/5) * 4864 →
  y = (2/3) * 9720 →
  z = (1/4) * 800 →
  (z / 1000) * 100 = 20 :=
by sorry

end NUMINAMATH_CALUDE_z_percentage_of_1000_l1675_167537


namespace NUMINAMATH_CALUDE_cats_vasyas_equality_l1675_167523

variable {α : Type*}
variable (C V : Set α)

theorem cats_vasyas_equality : C ∩ V = V ∩ C := by
  sorry

end NUMINAMATH_CALUDE_cats_vasyas_equality_l1675_167523


namespace NUMINAMATH_CALUDE_second_number_value_l1675_167593

def average_of_three (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem second_number_value (x y : ℚ) 
  (h1 : average_of_three 2 y x = 5) 
  (h2 : x = -63) : 
  y = 76 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l1675_167593


namespace NUMINAMATH_CALUDE_perpendicular_sum_equals_perimeter_l1675_167556

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define perpendicular points on angle bisectors
structure PerpendicularPoints (T : Triangle) :=
  (A1 A2 B1 B2 C1 C2 : ℝ × ℝ)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem perpendicular_sum_equals_perimeter (T : Triangle) (P : PerpendicularPoints T) :
  2 * (distance P.A1 P.A2 + distance P.B1 P.B2 + distance P.C1 P.C2) =
  distance T.A T.B + distance T.B T.C + distance T.C T.A := by sorry

end NUMINAMATH_CALUDE_perpendicular_sum_equals_perimeter_l1675_167556


namespace NUMINAMATH_CALUDE_smallest_angle_is_22_5_degrees_l1675_167595

def smallest_positive_angle (y : ℝ) : Prop :=
  6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 3/2 ∧
  y > 0 ∧
  ∀ z, z > 0 → 6 * Real.sin z * (Real.cos z)^3 - 6 * (Real.sin z)^3 * Real.cos z = 3/2 → y ≤ z

theorem smallest_angle_is_22_5_degrees :
  ∃ y, smallest_positive_angle y ∧ y = 22.5 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_is_22_5_degrees_l1675_167595


namespace NUMINAMATH_CALUDE_sandy_net_spent_l1675_167572

def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def jacket_return : ℚ := 7.43

theorem sandy_net_spent (shorts_cost shirt_cost jacket_return : ℚ) :
  shorts_cost = 13.99 →
  shirt_cost = 12.14 →
  jacket_return = 7.43 →
  shorts_cost + shirt_cost - jacket_return = 18.70 :=
by sorry

end NUMINAMATH_CALUDE_sandy_net_spent_l1675_167572


namespace NUMINAMATH_CALUDE_divisibility_puzzle_l1675_167509

theorem divisibility_puzzle (a : ℤ) :
  (∃! n : Fin 4, ¬ ((n = 0 → a % 2 = 0) ∧
                    (n = 1 → a % 4 = 0) ∧
                    (n = 2 → a % 12 = 0) ∧
                    (n = 3 → a % 24 = 0))) →
  ¬ (a % 24 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_puzzle_l1675_167509


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1675_167551

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1675_167551


namespace NUMINAMATH_CALUDE_factorial_divisibility_l1675_167599

/-- The number of ones in the binary representation of a natural number -/
def binary_ones (n : ℕ) : ℕ := sorry

theorem factorial_divisibility (n : ℕ) (h : binary_ones n = 1995) :
  ∃ k : ℕ, n! = k * 2^(n - 1995) :=
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l1675_167599


namespace NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_five_fourth_power_l1675_167550

theorem nearest_integer_to_three_plus_sqrt_five_fourth_power :
  ∃ (n : ℤ), ∀ (m : ℤ), |((3 : ℝ) + Real.sqrt 5)^4 - n| ≤ |((3 : ℝ) + Real.sqrt 5)^4 - m| ∧ n = 752 :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_five_fourth_power_l1675_167550


namespace NUMINAMATH_CALUDE_brother_age_in_five_years_l1675_167588

/-- Given the ages of Nick and his siblings, prove the brother's age in 5 years -/
theorem brother_age_in_five_years
  (nick_age : ℕ)
  (sister_age_diff : ℕ)
  (h_nick_age : nick_age = 13)
  (h_sister_age_diff : sister_age_diff = 6)
  (brother_age : ℕ)
  (h_brother_age : brother_age = (nick_age + (nick_age + sister_age_diff)) / 2) :
  brother_age + 5 = 21 := by
sorry


end NUMINAMATH_CALUDE_brother_age_in_five_years_l1675_167588


namespace NUMINAMATH_CALUDE_bicycle_tire_swap_theorem_l1675_167575

/-- Represents the characteristics and behavior of a bicycle with swappable tires -/
structure Bicycle where
  front_tire_life : ℝ
  rear_tire_life : ℝ

/-- Calculates the maximum distance a bicycle can travel with one tire swap -/
def max_distance_with_swap (b : Bicycle) : ℝ := sorry

/-- Calculates the optimal distance at which to swap tires -/
def optimal_swap_distance (b : Bicycle) : ℝ := sorry

/-- Theorem stating the maximum distance and optimal swap point for a specific bicycle -/
theorem bicycle_tire_swap_theorem (b : Bicycle) 
  (h1 : b.front_tire_life = 11000)
  (h2 : b.rear_tire_life = 9000) :
  max_distance_with_swap b = 9900 ∧ optimal_swap_distance b = 4950 := by sorry

end NUMINAMATH_CALUDE_bicycle_tire_swap_theorem_l1675_167575


namespace NUMINAMATH_CALUDE_pencil_cost_solution_l1675_167553

/-- The cost of Daniel's purchase -/
def purchase_problem (magazine_cost pencil_cost coupon_discount total_spent : ℚ) : Prop :=
  magazine_cost + pencil_cost - coupon_discount = total_spent

theorem pencil_cost_solution :
  ∃ (pencil_cost : ℚ),
    purchase_problem 0.85 pencil_cost 0.35 1 ∧ pencil_cost = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_solution_l1675_167553


namespace NUMINAMATH_CALUDE_cycle_selling_price_l1675_167514

/-- Calculates the final selling price of a cycle given initial cost, profit percentage, discount percentage, and sales tax percentage. -/
def finalSellingPrice (costPrice : ℚ) (profitPercentage : ℚ) (discountPercentage : ℚ) (salesTaxPercentage : ℚ) : ℚ :=
  let markedPrice := costPrice * (1 + profitPercentage / 100)
  let discountedPrice := markedPrice * (1 - discountPercentage / 100)
  discountedPrice * (1 + salesTaxPercentage / 100)

/-- Theorem stating that the final selling price of the cycle is 936.32 given the specified conditions. -/
theorem cycle_selling_price :
  finalSellingPrice 800 10 5 12 = 936.32 := by
  sorry


end NUMINAMATH_CALUDE_cycle_selling_price_l1675_167514


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l1675_167540

theorem vector_magnitude_problem (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2 = 5) →
  ((a.1 - b.1)^2 + (a.2 - b.2)^2 = 20) →
  (b.1^2 + b.2^2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l1675_167540


namespace NUMINAMATH_CALUDE_remainder_sum_l1675_167520

theorem remainder_sum (x y : ℤ) (hx : x % 90 = 75) (hy : y % 120 = 115) :
  (x + y) % 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1675_167520


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l1675_167541

/-- The surface area of a cube with the same volume as a rectangular prism of dimensions 10 inches by 5 inches by 20 inches is 600 square inches. -/
theorem cube_surface_area_equal_volume (prism_length prism_width prism_height : ℝ)
  (h1 : prism_length = 10)
  (h2 : prism_width = 5)
  (h3 : prism_height = 20) :
  (6 : ℝ) * ((prism_length * prism_width * prism_height) ^ (1/3 : ℝ))^2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l1675_167541


namespace NUMINAMATH_CALUDE_num_routes_eq_expected_l1675_167563

/-- Represents the number of southern cities -/
def num_southern_cities : ℕ := 4

/-- Represents the number of northern cities -/
def num_northern_cities : ℕ := 5

/-- Calculates the number of different routes for a traveler -/
def num_routes : ℕ := (Nat.factorial (num_southern_cities - 1)) * (num_northern_cities ^ num_southern_cities)

/-- Theorem stating that the number of routes is equal to 3! × 5^4 -/
theorem num_routes_eq_expected : num_routes = 3750 := by
  sorry

end NUMINAMATH_CALUDE_num_routes_eq_expected_l1675_167563


namespace NUMINAMATH_CALUDE_point_B_coordinates_l1675_167518

-- Define the point type
structure Point := (x : ℝ) (y : ℝ)

-- Define the problem statement
theorem point_B_coordinates (A B : Point) (h1 : A.x = 1 ∧ A.y = -1) 
  (h2 : (B.x - A.x)^2 + (B.y - A.y)^2 = 3^2) 
  (h3 : B.x = A.x) : 
  (B = Point.mk 1 (-4) ∨ B = Point.mk 1 2) := by
  sorry


end NUMINAMATH_CALUDE_point_B_coordinates_l1675_167518


namespace NUMINAMATH_CALUDE_dubblefud_red_balls_l1675_167577

/-- The game of dubblefud with red, blue, and green balls -/
def dubblefud (r b g : ℕ) : Prop :=
  2^r * 4^b * 5^g = 16000 ∧ b = g

theorem dubblefud_red_balls :
  ∃ (r b g : ℕ), dubblefud r b g ∧ r = 6 :=
sorry

end NUMINAMATH_CALUDE_dubblefud_red_balls_l1675_167577
