import Mathlib

namespace NUMINAMATH_CALUDE_road_trip_total_hours_l194_19454

/-- Calculates the total hours driven on a road trip -/
def total_hours_driven (days : ℕ) (jade_hours_per_day : ℕ) (krista_hours_per_day : ℕ) : ℕ :=
  days * (jade_hours_per_day + krista_hours_per_day)

/-- Proves that the total hours driven by Jade and Krista over 3 days equals 42 hours -/
theorem road_trip_total_hours : total_hours_driven 3 8 6 = 42 := by
  sorry

#eval total_hours_driven 3 8 6

end NUMINAMATH_CALUDE_road_trip_total_hours_l194_19454


namespace NUMINAMATH_CALUDE_total_cards_l194_19411

/-- The number of cards each person has -/
structure Cards where
  janet : ℕ
  brenda : ℕ
  mara : ℕ

/-- The conditions of the problem -/
def problem_conditions (c : Cards) : Prop :=
  c.janet = c.brenda + 9 ∧
  c.mara = 2 * c.janet ∧
  c.mara = 150 - 40

/-- The theorem to prove -/
theorem total_cards (c : Cards) (h : problem_conditions c) : 
  c.janet + c.brenda + c.mara = 211 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_l194_19411


namespace NUMINAMATH_CALUDE_age_inconsistency_l194_19472

/-- Given the ages of Sandy, Molly, and Noah, this theorem proves that the given conditions lead to a contradiction. -/
theorem age_inconsistency (S M N : ℕ) : 
  (M = S + 20) →  -- Sandy is younger than Molly by 20 years
  (S : ℚ) / M = 7 / 9 →  -- The ratio of Sandy's age to Molly's age is 7:9
  S + M + N = 120 →  -- The sum of their ages is 120
  (N - M : ℚ) = (1 / 2 : ℚ) * (M - S) →  -- The age difference between Noah and Molly is half that between Sandy and Molly
  False :=
by
  sorry

#eval 70 + 90  -- This evaluates to 160, which is already greater than 120


end NUMINAMATH_CALUDE_age_inconsistency_l194_19472


namespace NUMINAMATH_CALUDE_pen_sale_profit_percentage_l194_19485

/-- Calculate the profit percentage for a store owner's pen sale --/
theorem pen_sale_profit_percentage 
  (purchase_quantity : ℕ) 
  (marked_price_quantity : ℕ) 
  (discount_percentage : ℝ) : ℝ :=
by
  -- Assume purchase_quantity = 200
  -- Assume marked_price_quantity = 180
  -- Assume discount_percentage = 2
  
  -- Define cost price
  let cost_price := marked_price_quantity

  -- Define selling price per item
  let selling_price_per_item := 1 - (1 * discount_percentage / 100)

  -- Calculate total revenue
  let total_revenue := purchase_quantity * selling_price_per_item

  -- Calculate profit
  let profit := total_revenue - cost_price

  -- Calculate profit percentage
  let profit_percentage := (profit / cost_price) * 100

  -- Prove that profit_percentage ≈ 8.89
  sorry

-- The statement of the theorem
#check pen_sale_profit_percentage

end NUMINAMATH_CALUDE_pen_sale_profit_percentage_l194_19485


namespace NUMINAMATH_CALUDE_a2_value_l194_19464

def sequence_sum (n : ℕ) (k : ℕ) : ℚ := -1/2 * n^2 + k*n

theorem a2_value (k : ℕ) (h1 : k > 0) 
  (h2 : ∃ (n : ℕ), ∀ (m : ℕ), sequence_sum m k ≤ sequence_sum n k)
  (h3 : ∃ (n : ℕ), sequence_sum n k = 8) :
  sequence_sum 2 k - sequence_sum 1 k = 5/2 := by
sorry

end NUMINAMATH_CALUDE_a2_value_l194_19464


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l194_19440

/-- The parabolas y = (x - 2)^2 and x + 6 = (y + 1)^2 intersect at points that lie on a circle with radius squared 5/2 -/
theorem intersection_points_on_circle :
  ∃ (c : ℝ × ℝ) (r : ℝ),
    (∀ (p : ℝ × ℝ),
      (p.2 = (p.1 - 2)^2 ∧ p.1 + 6 = (p.2 + 1)^2) →
      (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    r^2 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l194_19440


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l194_19468

theorem inequality_holds_iff (m : ℝ) :
  (∀ x : ℝ, (x^2 + m*x - 1) / (2*x^2 - 2*x + 3) < 1) ↔ m > -6 ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l194_19468


namespace NUMINAMATH_CALUDE_middle_group_frequency_l194_19445

/-- Represents a frequency distribution histogram with 5 rectangles. -/
structure Histogram where
  rectangles : Fin 5 → ℝ
  total_sample : ℝ
  middle_equals_sum : rectangles 2 = (rectangles 0) + (rectangles 1) + (rectangles 3) + (rectangles 4)
  sample_size : total_sample = 100

/-- The frequency of the middle group in the histogram is 50. -/
theorem middle_group_frequency (h : Histogram) : h.rectangles 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_middle_group_frequency_l194_19445


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l194_19491

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^2 - 19 * X + 53 = (X - 3) * q + 23 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l194_19491


namespace NUMINAMATH_CALUDE_inscribing_square_area_l194_19495

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 + 2 * y^2 - 8 * x - 12 * y + 24 = 0

/-- The square inscribing the circle -/
structure InscribingSquare where
  side : ℝ
  center_x : ℝ
  center_y : ℝ
  inscribes_circle : ∀ (x y : ℝ), circle_equation x y →
    (|x - center_x| ≤ side / 2) ∧ (|y - center_y| ≤ side / 2)
  parallel_to_axes : True  -- This condition is implicit in the structure

/-- The theorem stating that the area of the inscribing square is 4 -/
theorem inscribing_square_area :
  ∀ (s : InscribingSquare), s.side^2 = 4 := by sorry

end NUMINAMATH_CALUDE_inscribing_square_area_l194_19495


namespace NUMINAMATH_CALUDE_ordering_abc_l194_19416

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem ordering_abc : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l194_19416


namespace NUMINAMATH_CALUDE_medicine_supply_duration_l194_19431

/-- Represents the duration in days that a supply of pills will last -/
def duration_in_days (num_pills : ℕ) (pill_fraction : ℚ) (days_between_doses : ℕ) : ℚ :=
  (num_pills : ℚ) * (days_between_doses : ℚ) / pill_fraction

/-- Converts days to months, assuming 30 days per month -/
def days_to_months (days : ℚ) : ℚ :=
  days / 30

theorem medicine_supply_duration :
  let num_pills : ℕ := 60
  let pill_fraction : ℚ := 3/4
  let days_between_doses : ℕ := 3
  let duration_days := duration_in_days num_pills pill_fraction days_between_doses
  let duration_months := days_to_months duration_days
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/3 ∧ |duration_months - 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_medicine_supply_duration_l194_19431


namespace NUMINAMATH_CALUDE_exist_four_lines_eight_regions_l194_19450

/-- A line in the coordinate plane defined by y = kx + b --/
structure Line where
  k : ℕ
  b : ℕ
  k_in_range : k ∈ Finset.range 9 \ {0}
  b_in_range : b ∈ Finset.range 9 \ {0}

/-- The set of four lines --/
def FourLines : Type := Fin 4 → Line

/-- All coefficients and constants are distinct --/
def all_distinct (lines : FourLines) : Prop :=
  ∀ i j, i ≠ j → lines i ≠ lines j

/-- The number of regions formed by the lines --/
def num_regions (lines : FourLines) : ℕ := sorry

/-- Theorem: There exist 4 lines that divide the plane into 8 regions --/
theorem exist_four_lines_eight_regions :
  ∃ (lines : FourLines), all_distinct lines ∧ num_regions lines = 8 := by sorry

end NUMINAMATH_CALUDE_exist_four_lines_eight_regions_l194_19450


namespace NUMINAMATH_CALUDE_egg_production_increase_l194_19497

theorem egg_production_increase (last_year_production this_year_production : ℕ) 
  (h1 : last_year_production = 1416)
  (h2 : this_year_production = 4636) :
  this_year_production - last_year_production = 3220 := by
  sorry

end NUMINAMATH_CALUDE_egg_production_increase_l194_19497


namespace NUMINAMATH_CALUDE_triangle_properties_l194_19451

/-- Given a triangle ABC with the following properties:
    1. (1 - tan A)(1 - tan B) = 2
    2. b = 2√2
    3. c = 4
    Prove that:
    1. Angle C = π/4
    2. Area of triangle ABC = 2√3 + 2 -/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  (1 - Real.tan A) * (1 - Real.tan B) = 2 →
  b = 2 * Real.sqrt 2 →
  c = 4 →
  C = π / 4 ∧
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l194_19451


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l194_19492

theorem complex_fraction_sum (a b : ℝ) :
  (3 + b * Complex.I) / (1 - Complex.I) = a + b * Complex.I →
  a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l194_19492


namespace NUMINAMATH_CALUDE_quadratic_function_property_l194_19490

theorem quadratic_function_property (a b c : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = a * x^2 + b * x + c) 
  (h_cond : f 0 = f 4 ∧ f 0 > f 1) :
  a > 0 ∧ 4 * a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l194_19490


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l194_19439

-- Inequality 1: (x+1)^2 + 3(x+1) - 4 > 0
theorem inequality_one (x : ℝ) : 
  (x + 1)^2 + 3*(x + 1) - 4 > 0 ↔ x < -5 ∨ x > 0 := by sorry

-- Inequality 2: x^4 - 2x^2 + 1 > x^2 - 1
theorem inequality_two (x : ℝ) : 
  x^4 - 2*x^2 + 1 > x^2 - 1 ↔ x < -Real.sqrt 2 ∨ (-1 < x ∧ x < 1) ∨ x > Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l194_19439


namespace NUMINAMATH_CALUDE_consecutive_number_sums_contradiction_l194_19405

theorem consecutive_number_sums_contradiction (a : Fin 15 → ℤ) :
  (∀ i : Fin 13, a i + a (i + 1) + a (i + 2) > 0) →
  (∀ i : Fin 12, a i + a (i + 1) + a (i + 2) + a (i + 3) < 0) →
  False :=
by sorry

end NUMINAMATH_CALUDE_consecutive_number_sums_contradiction_l194_19405


namespace NUMINAMATH_CALUDE_gnomes_taken_is_40_percent_l194_19414

/-- The percentage of gnomes taken by the forest owner in Ravenswood forest --/
def gnomes_taken_percentage (westerville_gnomes : ℕ) (ravenswood_multiplier : ℕ) (remaining_gnomes : ℕ) : ℚ :=
  100 - (remaining_gnomes : ℚ) / ((westerville_gnomes * ravenswood_multiplier) : ℚ) * 100

/-- Theorem stating that the percentage of gnomes taken is 40% --/
theorem gnomes_taken_is_40_percent :
  gnomes_taken_percentage 20 4 48 = 40 := by
  sorry

end NUMINAMATH_CALUDE_gnomes_taken_is_40_percent_l194_19414


namespace NUMINAMATH_CALUDE_stratified_sampling_survey_l194_19433

theorem stratified_sampling_survey (total_counties : ℕ) (jiujiang_counties : ℕ) (jiujiang_samples : ℕ) : 
  total_counties = 20 → jiujiang_counties = 8 → jiujiang_samples = 2 →
  ∃ (total_samples : ℕ), total_samples = 5 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_survey_l194_19433


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_range_l194_19403

theorem quadratic_always_nonnegative_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*m*x + m + 2 ≥ 0) ↔ m ∈ Set.Icc (-1) 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_range_l194_19403


namespace NUMINAMATH_CALUDE_unsold_bars_l194_19432

/-- Proves the number of unsold chocolate bars given total bars, price per bar, and total revenue --/
theorem unsold_bars (total_bars : ℕ) (price_per_bar : ℕ) (total_revenue : ℕ) : 
  total_bars = 13 → price_per_bar = 2 → total_revenue = 18 → 
  total_bars - (total_revenue / price_per_bar) = 4 := by
  sorry

end NUMINAMATH_CALUDE_unsold_bars_l194_19432


namespace NUMINAMATH_CALUDE_lawrence_county_camp_attendance_l194_19401

/-- The number of kids from Lawrence county who went to camp -/
def kids_at_camp (total : ℕ) (stayed_home : ℕ) : ℕ :=
  total - stayed_home

/-- Proof that 893,835 kids from Lawrence county went to camp -/
theorem lawrence_county_camp_attendance :
  kids_at_camp 1538832 644997 = 893835 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_camp_attendance_l194_19401


namespace NUMINAMATH_CALUDE_equation_solution_l194_19461

/-- Given positive real numbers a, b, c ≤ 1, the equation 
    min{√((ab+1)/(abc)), √((bc+1)/(abc)), √((ac+1)/(abc))} = √((1-a)/a) + √((1-b)/b) + √((1-c)/c)
    is satisfied if and only if (a, b, c) = (1/(-t^2 + t + 1), t, 1 - t) for 1/2 ≤ t < 1 or its permutations. -/
theorem equation_solution (a b c : ℝ) (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (min (Real.sqrt ((a*b+1)/(a*b*c))) (min (Real.sqrt ((b*c+1)/(a*b*c))) (Real.sqrt ((a*c+1)/(a*b*c)))) =
   Real.sqrt ((1-a)/a) + Real.sqrt ((1-b)/b) + Real.sqrt ((1-c)/c)) ↔
  (∃ t : ℝ, (1/2 ≤ t ∧ t < 1) ∧
   ((a = 1/(-t^2 + t + 1) ∧ b = t ∧ c = 1 - t) ∨
    (a = t ∧ b = 1 - t ∧ c = 1/(-t^2 + t + 1)) ∨
    (a = 1 - t ∧ b = 1/(-t^2 + t + 1) ∧ c = t))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l194_19461


namespace NUMINAMATH_CALUDE_max_principals_is_five_l194_19404

/-- Represents the duration of a principal's term in years -/
def term_length : ℕ := 3

/-- Represents the total period in years we're considering -/
def total_period : ℕ := 10

/-- Calculates the maximum number of principals that can serve during the total period -/
def max_principals : ℕ := 
  (total_period / term_length) + 
  (if total_period % term_length > 0 then 2 else 1)

/-- Theorem stating the maximum number of principals during the given period -/
theorem max_principals_is_five : max_principals = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_principals_is_five_l194_19404


namespace NUMINAMATH_CALUDE_same_volume_prisms_l194_19471

def edge_lengths : List ℕ := [12, 18, 20, 24, 30, 33, 70, 24, 154]

def is_valid_prism (a b c : ℕ) : Bool :=
  a ∈ edge_lengths ∧ b ∈ edge_lengths ∧ c ∈ edge_lengths

def prism_volume (a b c : ℕ) : ℕ := a * b * c

theorem same_volume_prisms :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℕ),
    is_valid_prism a₁ b₁ c₁ ∧
    is_valid_prism a₂ b₂ c₂ ∧
    is_valid_prism a₃ b₃ c₃ ∧
    prism_volume a₁ b₁ c₁ = prism_volume a₂ b₂ c₂ ∧
    prism_volume a₂ b₂ c₂ = prism_volume a₃ b₃ c₃ ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂) ∧
    (a₂, b₂, c₂) ≠ (a₃, b₃, c₃) ∧
    (a₁, b₁, c₁) ≠ (a₃, b₃, c₃) :=
by
  sorry

#check same_volume_prisms

end NUMINAMATH_CALUDE_same_volume_prisms_l194_19471


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_half_l194_19477

/-- First line parameterization --/
def line1 (s : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (2 + s, 4 - k*s, -1 + k*s)

/-- Second line parameterization --/
def line2 (t : ℝ) : ℝ × ℝ × ℝ := (2*t, 2 + t, 3 - t)

/-- Direction vector of the first line --/
def dir1 (k : ℝ) : ℝ × ℝ × ℝ := (1, -k, k)

/-- Direction vector of the second line --/
def dir2 : ℝ × ℝ × ℝ := (2, 1, -1)

/-- Two lines are coplanar if and only if k = -1/2 --/
theorem lines_coplanar_iff_k_eq_neg_half :
  (∃ (a b : ℝ), a • dir1 k + b • dir2 = (0, 0, 0)) ↔ k = -1/2 := by sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_half_l194_19477


namespace NUMINAMATH_CALUDE_friend_walking_rates_l194_19415

theorem friend_walking_rates (trail_length : ℝ) (p_distance : ℝ) 
  (h1 : trail_length = 36)
  (h2 : p_distance = 20)
  (h3 : p_distance < trail_length) :
  let q_distance := trail_length - p_distance
  let rate_ratio := p_distance / q_distance
  (rate_ratio - 1) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_friend_walking_rates_l194_19415


namespace NUMINAMATH_CALUDE_bob_anne_distance_difference_l194_19493

/-- Represents the dimensions of a rectangular block in Geometrytown --/
structure BlockDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle --/
def rectanglePerimeter (d : BlockDimensions) : ℕ :=
  2 * (d.length + d.width)

/-- Represents the street width in Geometrytown --/
def streetWidth : ℕ := 30

/-- Calculates Bob's running distance around the block --/
def bobDistance (d : BlockDimensions) : ℕ :=
  rectanglePerimeter { length := d.length + 2 * streetWidth, width := d.width + 2 * streetWidth }

/-- Calculates Anne's running distance around the block --/
def anneDistance (d : BlockDimensions) : ℕ :=
  rectanglePerimeter d

/-- The main theorem stating the difference between Bob's and Anne's running distances --/
theorem bob_anne_distance_difference (d : BlockDimensions) 
    (h1 : d.length = 300) 
    (h2 : d.width = 500) : 
    bobDistance d - anneDistance d = 240 := by
  sorry

end NUMINAMATH_CALUDE_bob_anne_distance_difference_l194_19493


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l194_19418

theorem fraction_equality_solution :
  ∃! y : ℚ, (2 + y) / (6 + y) = (3 + y) / (4 + y) :=
by
  -- The unique solution is y = -10/3
  use -10/3
  sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l194_19418


namespace NUMINAMATH_CALUDE_game_result_l194_19467

def game_operation (n : ℕ) : ℕ :=
  if n % 2 = 1 then n + 3 else n / 2

def reaches_one_in (n : ℕ) (steps : ℕ) : Prop :=
  ∃ (seq : Fin steps.succ → ℕ), 
    seq 0 = n ∧ 
    seq steps = 1 ∧ 
    ∀ i : Fin steps, seq (i.succ) = game_operation (seq i)

theorem game_result :
  {n : ℕ | reaches_one_in n 5} = {1, 8, 16, 10, 13} := by sorry

end NUMINAMATH_CALUDE_game_result_l194_19467


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l194_19488

theorem binomial_coefficient_ratio : ∀ a₀ a₁ a₂ a₃ a₄ a₅ : ℤ,
  (∀ x : ℤ, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61 / 60 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l194_19488


namespace NUMINAMATH_CALUDE_stone_distribution_fractions_l194_19473

/-- Number of indistinguishable stones -/
def n : ℕ := 12

/-- Number of distinguishable boxes -/
def k : ℕ := 4

/-- Total number of ways to distribute n stones among k boxes -/
def total_distributions : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Number of ways to distribute stones with even number in each box -/
def even_distributions : ℕ := Nat.choose ((n / 2) + k - 1) (k - 1)

/-- Number of ways to distribute stones with odd number in each box -/
def odd_distributions : ℕ := Nat.choose ((n - k) / 2 + k - 1) (k - 1)

theorem stone_distribution_fractions :
  (even_distributions : ℚ) / total_distributions = 12 / 65 ∧
  (odd_distributions : ℚ) / total_distributions = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_stone_distribution_fractions_l194_19473


namespace NUMINAMATH_CALUDE_max_cyclic_sum_l194_19476

theorem max_cyclic_sum (a b c d e : ℕ) :
  a ∈ ({1, 2, 3, 4, 5} : Finset ℕ) →
  b ∈ ({1, 2, 3, 4, 5} : Finset ℕ) →
  c ∈ ({1, 2, 3, 4, 5} : Finset ℕ) →
  d ∈ ({1, 2, 3, 4, 5} : Finset ℕ) →
  e ∈ ({1, 2, 3, 4, 5} : Finset ℕ) →
  a ≠ b → a ≠ c → a ≠ d → a ≠ e →
  b ≠ c → b ≠ d → b ≠ e →
  c ≠ d → c ≠ e →
  d ≠ e →
  (∀ x : ℕ, x ∈ ({1, 2, 3, 4, 5} : Finset ℕ) → a * b + b * c + c * d + d * e + e * a ≤ x) →
  a * b + b * c + c * d + d * e + e * a = 42 :=
sorry

end NUMINAMATH_CALUDE_max_cyclic_sum_l194_19476


namespace NUMINAMATH_CALUDE_project_B_highest_score_l194_19452

structure Project where
  name : String
  innovation : ℝ
  practicality : ℝ

def totalScore (p : Project) : ℝ :=
  0.6 * p.innovation + 0.4 * p.practicality

def projectA : Project := ⟨"A", 90, 90⟩
def projectB : Project := ⟨"B", 95, 90⟩
def projectC : Project := ⟨"C", 90, 95⟩
def projectD : Project := ⟨"D", 90, 85⟩

def projects : List Project := [projectA, projectB, projectC, projectD]

theorem project_B_highest_score :
  ∀ p ∈ projects, p ≠ projectB → totalScore p ≤ totalScore projectB :=
sorry

end NUMINAMATH_CALUDE_project_B_highest_score_l194_19452


namespace NUMINAMATH_CALUDE_intersection_equals_half_open_interval_l194_19424

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {x : ℝ | x^2 < 1}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- State the theorem
theorem intersection_equals_half_open_interval :
  M_intersect_N = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_half_open_interval_l194_19424


namespace NUMINAMATH_CALUDE_number_relationship_l194_19417

/-- Given two real numbers satisfying certain conditions, prove they are approximately equal to specific values. -/
theorem number_relationship (x y : ℝ) 
  (h1 : 0.25 * x = 1.3 * 0.35 * y) 
  (h2 : x - y = 155) : 
  ∃ (εx εy : ℝ), εx < 1 ∧ εy < 1 ∧ |x - 344| < εx ∧ |y - 189| < εy :=
sorry

end NUMINAMATH_CALUDE_number_relationship_l194_19417


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l194_19496

/-- Given a rhombus with diagonals of length 10 and 24, its perimeter is 52. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 52 := by
sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l194_19496


namespace NUMINAMATH_CALUDE_tangent_line_determines_b_l194_19428

/-- A curve of the form y = x³ + ax + b -/
def curve (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- The derivative of the curve -/
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_line_determines_b (a b : ℝ) :
  curve a b 1 = 3 →
  curve_derivative a 1 = 2 →
  b = 3 := by
  sorry

#check tangent_line_determines_b

end NUMINAMATH_CALUDE_tangent_line_determines_b_l194_19428


namespace NUMINAMATH_CALUDE_triangle_array_properties_l194_19448

-- Define what it means to be a triangle array
def is_triangle_array (a b c : ℝ) : Prop :=
  0 < a ∧ a ≤ b ∧ b ≤ c ∧ a + b > c

-- Theorem statement
theorem triangle_array_properties 
  (p q r : ℝ) 
  (h : is_triangle_array p q r) : 
  (is_triangle_array (Real.sqrt p) (Real.sqrt q) (Real.sqrt r)) ∧ 
  (∃ p q r : ℝ, is_triangle_array p q r ∧ ¬is_triangle_array (p^2) (q^2) (r^2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_array_properties_l194_19448


namespace NUMINAMATH_CALUDE_probability_reach_top_correct_l194_19447

/-- The probability of reaching the top floor using only open doors in a building with n floors and two staircases, where half the doors are randomly locked. -/
def probability_reach_top (n : ℕ) : ℚ :=
  (2 ^ (n - 1)) / (Nat.choose (2 * (n - 1)) (n - 1))

/-- Theorem stating the probability of reaching the top floor using only open doors. -/
theorem probability_reach_top_correct (n : ℕ) (h : n > 1) :
  probability_reach_top n = (2 ^ (n - 1)) / (Nat.choose (2 * (n - 1)) (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_probability_reach_top_correct_l194_19447


namespace NUMINAMATH_CALUDE_modular_inverse_of_two_mod_187_l194_19481

theorem modular_inverse_of_two_mod_187 : ∃ x : ℤ, 0 ≤ x ∧ x < 187 ∧ (2 * x) % 187 = 1 :=
by
  use 94
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_two_mod_187_l194_19481


namespace NUMINAMATH_CALUDE_largest_integer_solution_l194_19463

theorem largest_integer_solution (x : ℤ) : x ≤ 2 ↔ x / 3 + 4 / 5 < 5 / 3 := by sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l194_19463


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l194_19466

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (3 - 2*I) / I
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l194_19466


namespace NUMINAMATH_CALUDE_inequality_theorem_l194_19438

theorem inequality_theorem (p : ℝ) (h1 : p ≥ 0) (h2 : p < 2.232) :
  ∀ q : ℝ, q > 0 → (4 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 5 * p^2 * q := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l194_19438


namespace NUMINAMATH_CALUDE_illegal_parking_percentage_l194_19487

theorem illegal_parking_percentage (total_cars : ℕ) (towed_cars : ℕ) (illegal_cars : ℕ) :
  towed_cars = (2 : ℕ) * total_cars / 100 →
  (80 : ℕ) * illegal_cars / 100 = illegal_cars - towed_cars →
  illegal_cars * 100 / total_cars = 10 := by
  sorry

end NUMINAMATH_CALUDE_illegal_parking_percentage_l194_19487


namespace NUMINAMATH_CALUDE_root_difference_squared_l194_19426

theorem root_difference_squared (a : ℝ) (r s : ℝ) : 
  r^2 - (a+1)*r + a = 0 → 
  s^2 - (a+1)*s + a = 0 → 
  (r-s)^2 = a^2 - 2*a + 1 := by
sorry

end NUMINAMATH_CALUDE_root_difference_squared_l194_19426


namespace NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l194_19408

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 527 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l194_19408


namespace NUMINAMATH_CALUDE_coin_flip_probability_l194_19446

/-- The probability of getting exactly k successes in n independent trials -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The number of coin flips -/
def n : ℕ := 3

/-- The number of times we want the coin to land tails up -/
def k : ℕ := 2

/-- The probability of the coin landing tails up on a single flip -/
def p : ℝ := 0.5

theorem coin_flip_probability : binomial_probability n k p = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l194_19446


namespace NUMINAMATH_CALUDE_triangle_area_l194_19474

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  c^2 = (a - b)^2 + 6 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l194_19474


namespace NUMINAMATH_CALUDE_ratio_problem_l194_19409

theorem ratio_problem (x y : ℚ) (h : (3*x - 2*y) / (2*x + y) = 3/4) : x / y = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l194_19409


namespace NUMINAMATH_CALUDE_power_of_two_equality_l194_19456

theorem power_of_two_equality (x : ℕ) : (1 / 8 : ℝ) * (2 ^ 36) = 2 ^ x → x = 33 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l194_19456


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l194_19499

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_eq : a + b + c = 8)
  (sum_prod_eq : a * b + a * c + b * c = 13)
  (prod_eq : a * b * c = -22) :
  a^4 + b^4 + c^4 = 1378 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l194_19499


namespace NUMINAMATH_CALUDE_median_inequality_l194_19462

-- Define a right triangle with medians
structure RightTriangle where
  c : ℝ  -- length of hypotenuse
  sa : ℝ  -- length of median to one leg
  sb : ℝ  -- length of median to the other leg
  c_pos : c > 0  -- hypotenuse length is positive

-- State the theorem
theorem median_inequality (t : RightTriangle) : 
  (3/2) * t.c < t.sa + t.sb ∧ t.sa + t.sb ≤ (Real.sqrt 10 / 2) * t.c := by
  sorry

end NUMINAMATH_CALUDE_median_inequality_l194_19462


namespace NUMINAMATH_CALUDE_equipment_production_l194_19435

theorem equipment_production (total : ℕ) (sample_size : ℕ) (sample_a : ℕ) 
  (h_total : total = 4800)
  (h_sample_size : sample_size = 80)
  (h_sample_a : sample_a = 50) :
  total - (total * sample_a / sample_size) = 1800 := by
sorry

end NUMINAMATH_CALUDE_equipment_production_l194_19435


namespace NUMINAMATH_CALUDE_painted_cube_problem_l194_19429

theorem painted_cube_problem (n : ℕ) : 
  n > 0 →  -- Ensure n is positive
  (2 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 6 → 
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_problem_l194_19429


namespace NUMINAMATH_CALUDE_luncheon_invitees_l194_19489

/-- The number of people who didn't show up to the luncheon -/
def no_shows : ℕ := 50

/-- The number of people each table can hold -/
def people_per_table : ℕ := 3

/-- The number of tables needed for the people who showed up -/
def tables_used : ℕ := 6

/-- The total number of people originally invited to the luncheon -/
def total_invited : ℕ := no_shows + people_per_table * tables_used + 1

/-- Theorem stating that the number of people originally invited to the luncheon is 101 -/
theorem luncheon_invitees : total_invited = 101 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_invitees_l194_19489


namespace NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_y_is_six_l194_19402

theorem smallest_integer_y (y : ℤ) : (10 - 5*y < -15) ↔ (y ≥ 6) := by
  sorry

theorem smallest_integer_y_is_six : ∃ (y : ℤ), (10 - 5*y < -15) ∧ (∀ (z : ℤ), (10 - 5*z < -15) → z ≥ y) ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_y_is_six_l194_19402


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_7580_l194_19486

/-- Calculates the cost of white washing a trapezoidal room with given dimensions and conditions -/
def whitewashingCost (length width height1 height2 : ℕ) (doorCount windowCount : ℕ) 
  (doorLength doorWidth windowLength windowWidth : ℕ) (decorationArea : ℕ) (ratePerSqFt : ℕ) : ℕ :=
  let totalWallArea := 2 * (length * height1 + width * height2)
  let doorArea := doorCount * doorLength * doorWidth
  let windowArea := windowCount * windowLength * windowWidth
  let adjustedArea := totalWallArea - doorArea - windowArea - decorationArea
  adjustedArea * ratePerSqFt

/-- Theorem stating that the cost of white washing the given trapezoidal room is 7580 -/
theorem whitewashing_cost_is_7580 : 
  whitewashingCost 25 15 12 8 2 3 6 3 4 3 10 10 = 7580 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_is_7580_l194_19486


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_51200_l194_19478

/-- The number of factors of 51200 that are perfect squares -/
def perfect_square_factors_of_51200 : ℕ :=
  (Finset.range 6).card * (Finset.range 2).card

/-- Theorem stating that the number of factors of 51200 that are perfect squares is 12 -/
theorem count_perfect_square_factors_51200 :
  perfect_square_factors_of_51200 = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_51200_l194_19478


namespace NUMINAMATH_CALUDE_pairball_playing_time_l194_19475

theorem pairball_playing_time (n : ℕ) (total_time : ℕ) (h1 : n = 7) (h2 : total_time = 105) :
  let players_per_game : ℕ := 2
  let total_child_minutes : ℕ := players_per_game * total_time
  let time_per_child : ℕ := total_child_minutes / n
  time_per_child = 30 := by sorry

end NUMINAMATH_CALUDE_pairball_playing_time_l194_19475


namespace NUMINAMATH_CALUDE_tim_watched_24_hours_l194_19437

/-- Calculates the total hours of TV watched given the number of episodes and duration per episode for two shows. -/
def total_hours_watched (short_episodes : ℕ) (short_duration : ℚ) (long_episodes : ℕ) (long_duration : ℚ) : ℚ :=
  short_episodes * short_duration + long_episodes * long_duration

/-- Proves that Tim watched 24 hours of TV given the specified conditions. -/
theorem tim_watched_24_hours :
  let short_episodes : ℕ := 24
  let short_duration : ℚ := 1/2
  let long_episodes : ℕ := 12
  let long_duration : ℚ := 1
  total_hours_watched short_episodes short_duration long_episodes long_duration = 24 := by
  sorry

#eval total_hours_watched 24 (1/2) 12 1

end NUMINAMATH_CALUDE_tim_watched_24_hours_l194_19437


namespace NUMINAMATH_CALUDE_solve_equation_l194_19420

theorem solve_equation (y : ℝ) (h : (2 * y) / 3 = 12) : y = 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l194_19420


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l194_19483

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) :
  1 / x + 1 / y = 8 / 75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l194_19483


namespace NUMINAMATH_CALUDE_zeros_arithmetic_sequence_implies_a_value_l194_19480

/-- A cubic polynomial function -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + x + a

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 + 4*x + 1

/-- Theorem: If the zeros of f form an arithmetic sequence, then a = -23/54 -/
theorem zeros_arithmetic_sequence_implies_a_value (a : ℝ) : 
  (∃ r s t : ℝ, (f a r = 0 ∧ f a s = 0 ∧ f a t = 0) ∧ 
   (s - r = t - s) ∧ (r < s ∧ s < t)) → 
  a = -23/54 := by
  sorry

end NUMINAMATH_CALUDE_zeros_arithmetic_sequence_implies_a_value_l194_19480


namespace NUMINAMATH_CALUDE_students_not_in_same_row_or_column_l194_19421

/-- Represents a student's position in a classroom --/
structure Position where
  row : ℕ
  column : ℕ

/-- Defines the seating arrangement for students A and B --/
def seating_arrangement : (Position × Position) :=
  (⟨3, 6⟩, ⟨6, 3⟩)

/-- Theorem stating that students A and B are not in the same row or column --/
theorem students_not_in_same_row_or_column :
  let (student_a, student_b) := seating_arrangement
  (student_a.row ≠ student_b.row) ∧ (student_a.column ≠ student_b.column) := by
  sorry

#check students_not_in_same_row_or_column

end NUMINAMATH_CALUDE_students_not_in_same_row_or_column_l194_19421


namespace NUMINAMATH_CALUDE_book_pages_count_l194_19459

/-- The number of pages Liam read in a week -/
def total_pages : ℕ :=
  let first_three_days := 3 * 40
  let next_three_days := 3 * 50
  let seventh_day_first_session := 15
  let seventh_day_second_session := 2 * seventh_day_first_session
  first_three_days + next_three_days + seventh_day_first_session + seventh_day_second_session

/-- Theorem stating that the total number of pages in the book is 315 -/
theorem book_pages_count : total_pages = 315 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l194_19459


namespace NUMINAMATH_CALUDE_min_value_theorem_l194_19465

/-- Given a function y = a^x + b where b > 0 and the graph passes through point (1,3),
    the minimum value of 4/(a-1) + 1/b is 9/2 -/
theorem min_value_theorem (a b : ℝ) (h1 : b > 0) (h2 : a^1 + b = 3) :
  (∀ x y : ℝ, x > 1 ∧ y > 0 ∧ x^1 + y = 3 → 4/(x-1) + 1/y ≥ 9/2) ∧
  (∃ x y : ℝ, x > 1 ∧ y > 0 ∧ x^1 + y = 3 ∧ 4/(x-1) + 1/y = 9/2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l194_19465


namespace NUMINAMATH_CALUDE_total_volume_is_716_l194_19484

/-- The volume of a cube with side length s -/
def cube_volume (s : ℝ) : ℝ := s ^ 3

/-- The number of cubes Carl has -/
def carl_cubes : ℕ := 8

/-- The side length of Carl's cubes -/
def carl_side_length : ℝ := 3

/-- The number of cubes Kate has -/
def kate_cubes : ℕ := 4

/-- The side length of Kate's cubes -/
def kate_side_length : ℝ := 5

/-- The total volume of all cubes -/
def total_volume : ℝ :=
  (carl_cubes : ℝ) * cube_volume carl_side_length +
  (kate_cubes : ℝ) * cube_volume kate_side_length

theorem total_volume_is_716 : total_volume = 716 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_is_716_l194_19484


namespace NUMINAMATH_CALUDE_bridge_length_problem_l194_19410

/-- The length of a bridge crossed by a man walking at a given speed in a given time -/
def bridge_length (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A man walking at 10 km/hr crosses a bridge in 18 minutes. The bridge length is 3 km. -/
theorem bridge_length_problem :
  let walking_speed : ℝ := 10  -- km/hr
  let crossing_time : ℝ := 18 / 60  -- 18 minutes converted to hours
  bridge_length walking_speed crossing_time = 3 := by
sorry


end NUMINAMATH_CALUDE_bridge_length_problem_l194_19410


namespace NUMINAMATH_CALUDE_age_equation_solution_l194_19498

/-- Given a person's age and a number of years, this function represents the equation in the problem. -/
def ageEquation (A : ℕ) (x : ℕ) : Prop :=
  3 * (A + x) - 3 * (A - x) = A

/-- The theorem states that for an age of 30, the equation is satisfied when x is 5. -/
theorem age_equation_solution :
  ageEquation 30 5 := by
  sorry

end NUMINAMATH_CALUDE_age_equation_solution_l194_19498


namespace NUMINAMATH_CALUDE_square_sum_plus_double_sum_squares_l194_19482

theorem square_sum_plus_double_sum_squares : (5 + 7)^2 + (5^2 + 7^2) * 2 = 292 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_plus_double_sum_squares_l194_19482


namespace NUMINAMATH_CALUDE_triangle_inequality_range_l194_19407

/-- A right-angled triangle with sides a, b, and hypotenuse c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  pythagorean : a^2 + b^2 = c^2

/-- The theorem stating the range of t for which the given inequality holds -/
theorem triangle_inequality_range (tri : RightTriangle) :
  (∀ t : ℝ, 1 / tri.a^2 + 4 / tri.b^2 + t / tri.c^2 ≥ 0) ↔ 
  (∀ t : ℝ, t ≥ -9 ∧ t ∈ Set.Ici (-9)) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_range_l194_19407


namespace NUMINAMATH_CALUDE_complex_equation_solution_l194_19441

theorem complex_equation_solution (z : ℂ) (h : (3 + z) * Complex.I = 1) : z = -3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l194_19441


namespace NUMINAMATH_CALUDE_test_score_calculation_l194_19400

theorem test_score_calculation (total_marks : ℕ) (percentage : ℚ) : 
  total_marks = 50 → percentage = 80 / 100 → (percentage * total_marks : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_test_score_calculation_l194_19400


namespace NUMINAMATH_CALUDE_unique_cubic_zero_a_range_l194_19479

/-- A cubic function with a unique positive zero point -/
structure UniqueCubicZero where
  a : ℝ
  f : ℝ → ℝ := λ x ↦ a * x^3 - 3 * x^2 + 1
  x₀ : ℝ
  x₀_pos : x₀ > 0
  x₀_zero : f x₀ = 0
  unique_zero : ∀ x, f x = 0 → x = x₀

/-- The range of 'a' for a cubic function with a unique positive zero point -/
theorem unique_cubic_zero_a_range (c : UniqueCubicZero) : c.a < -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_cubic_zero_a_range_l194_19479


namespace NUMINAMATH_CALUDE_average_birds_per_site_l194_19458

-- Define the data for each day
def monday_sites : ℕ := 5
def monday_avg : ℕ := 7
def tuesday_sites : ℕ := 5
def tuesday_avg : ℕ := 5
def wednesday_sites : ℕ := 10
def wednesday_avg : ℕ := 8

-- Define the total number of sites
def total_sites : ℕ := monday_sites + tuesday_sites + wednesday_sites

-- Define the total number of birds
def total_birds : ℕ := monday_sites * monday_avg + tuesday_sites * tuesday_avg + wednesday_sites * wednesday_avg

-- Theorem to prove
theorem average_birds_per_site :
  total_birds / total_sites = 7 := by
  sorry

end NUMINAMATH_CALUDE_average_birds_per_site_l194_19458


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l194_19455

theorem fraction_equation_solution : 
  ∃ x : ℚ, (3 / (x - 2) + 5 / (x + 2) = 8 / (x^2 - 4)) ∧ (x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l194_19455


namespace NUMINAMATH_CALUDE_chemistry_class_b_count_l194_19460

/-- Represents the number of students who earn each grade in a chemistry class. -/
structure GradeDistribution where
  a : ℝ  -- Number of students earning A
  b : ℝ  -- Number of students earning B
  c : ℝ  -- Number of students earning C
  d : ℝ  -- Number of students earning D

/-- The grade distribution in a chemistry class of 50 students satisfies given probability ratios. -/
def chemistryClass (g : GradeDistribution) : Prop :=
  g.a = 0.5 * g.b ∧
  g.c = 1.2 * g.b ∧
  g.d = 0.3 * g.b ∧
  g.a + g.b + g.c + g.d = 50

/-- The number of students earning a B in the chemistry class is 50/3. -/
theorem chemistry_class_b_count :
  ∀ g : GradeDistribution, chemistryClass g → g.b = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_class_b_count_l194_19460


namespace NUMINAMATH_CALUDE_teacher_arrangement_count_l194_19425

/-- The number of female teachers -/
def num_female : ℕ := 2

/-- The number of male teachers -/
def num_male : ℕ := 4

/-- The number of female teachers per group -/
def female_per_group : ℕ := 1

/-- The number of male teachers per group -/
def male_per_group : ℕ := 2

/-- The total number of groups -/
def num_groups : ℕ := 2

theorem teacher_arrangement_count :
  (num_female.choose female_per_group) * (num_male.choose male_per_group) = 12 := by
  sorry

end NUMINAMATH_CALUDE_teacher_arrangement_count_l194_19425


namespace NUMINAMATH_CALUDE_coach_cost_l194_19443

/-- Proves that the cost of the coach before discount is $2500 given the problem conditions -/
theorem coach_cost (sectional_cost other_cost total_paid : ℝ) 
  (h1 : sectional_cost = 3500)
  (h2 : other_cost = 2000)
  (h3 : total_paid = 7200)
  (discount : ℝ) (h4 : discount = 0.1)
  : ∃ (coach_cost : ℝ), 
    coach_cost = 2500 ∧ 
    (1 - discount) * (coach_cost + sectional_cost + other_cost) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_coach_cost_l194_19443


namespace NUMINAMATH_CALUDE_average_weight_problem_l194_19412

theorem average_weight_problem (total_boys : Nat) (group_a_boys : Nat) (group_b_boys : Nat)
  (group_b_avg_weight : ℝ) (total_avg_weight : ℝ) :
  total_boys = 34 →
  group_a_boys = 26 →
  group_b_boys = 8 →
  group_b_avg_weight = 45.15 →
  total_avg_weight = 49.05 →
  let group_a_avg_weight := (total_boys * total_avg_weight - group_b_boys * group_b_avg_weight) / group_a_boys
  group_a_avg_weight = 50.25 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l194_19412


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l194_19434

/-- The quadratic equation x^2 - 6mx + 9m has exactly one real root if and only if m = 1 (for positive m) -/
theorem unique_root_quadratic (m : ℝ) (h : m > 0) : 
  (∃! x : ℝ, x^2 - 6*m*x + 9*m = 0) ↔ m = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l194_19434


namespace NUMINAMATH_CALUDE_anion_and_salt_identification_l194_19470

/-- Represents an anion with an O-O bond -/
structure AnionWithOOBond where
  has_oo_bond : Bool

/-- Represents a salt formed during anodic oxidation of bisulfate -/
structure SaltFromBisulfateOxidation where
  is_sulfate_based : Bool

/-- Theorem stating that an anion with an O-O bond is a peroxide ion and 
    the salt formed from bisulfate oxidation is sulfate-based -/
theorem anion_and_salt_identification 
  (anion : AnionWithOOBond) 
  (salt : SaltFromBisulfateOxidation) : 
  (anion.has_oo_bond → (∃ x : String, x = "O₂²⁻")) ∧ 
  (salt.is_sulfate_based → (∃ y : String, y = "K₂SO₄")) := by
  sorry

end NUMINAMATH_CALUDE_anion_and_salt_identification_l194_19470


namespace NUMINAMATH_CALUDE_special_function_value_l194_19494

/-- A binary function on positive integers satisfying certain properties -/
def special_function (f : ℕ+ → ℕ+ → ℕ+) : Prop :=
  (∀ x, f x x = x) ∧
  (∀ x y, f x y = f y x) ∧
  (∀ x y, (x + y) * (f x y) = y * (f x (x + y)))

/-- Theorem stating that f(12, 16) = 48 for any function satisfying the special properties -/
theorem special_function_value (f : ℕ+ → ℕ+ → ℕ+) (h : special_function f) : 
  f 12 16 = 48 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l194_19494


namespace NUMINAMATH_CALUDE_tan_half_alpha_l194_19469

theorem tan_half_alpha (α : ℝ) (h1 : π < α) (h2 : α < 3*π/2) 
  (h3 : Real.sin (3*π/2 + α) = 4/5) : Real.tan (α/2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_alpha_l194_19469


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l194_19457

/-- A system of linear equations in two variables x and y -/
structure LinearSystem where
  a : ℝ
  eq1 : ℝ → ℝ → ℝ := λ x y => x + 2 * y - a
  eq2 : ℝ → ℝ → ℝ := λ x y => 2 * x - y - 1

/-- The system has a solution when both equations equal zero -/
def HasSolution (s : LinearSystem) (x y : ℝ) : Prop :=
  s.eq1 x y = 0 ∧ s.eq2 x y = 0

theorem solution_part1 (s : LinearSystem) :
  HasSolution s 1 1 → s.a = 3 := by sorry

theorem solution_part2 (s : LinearSystem) :
  s.a = -2 → HasSolution s 0 (-1) ∧
  (∀ x y : ℝ, HasSolution s x y → x = 0 ∧ y = -1) := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l194_19457


namespace NUMINAMATH_CALUDE_bottle_production_l194_19442

/-- Given that 6 identical machines produce 24 bottles per minute at a constant rate,
    prove that 10 such machines will produce 160 bottles in 4 minutes. -/
theorem bottle_production 
  (rate : ℕ) -- Production rate per machine per minute
  (h1 : 6 * rate = 24) -- 6 machines produce 24 bottles per minute
  : 10 * rate * 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_bottle_production_l194_19442


namespace NUMINAMATH_CALUDE_ed_weight_l194_19422

/-- Given the weights of Al, Ben, Carl, and Ed, prove that Ed weighs 146 pounds -/
theorem ed_weight (al ben carl ed : ℕ) : 
  al = ben + 25 →
  ben = carl - 16 →
  ed = al - 38 →
  carl = 175 →
  ed = 146 := by
  sorry

end NUMINAMATH_CALUDE_ed_weight_l194_19422


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l194_19453

theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) :
  (∀ n, a n = a 0 + n * d) →  -- arithmetic sequence definition
  (a 19 = 205) →             -- given condition a_20 = 205 (index starts at 0)
  (a 0 = 91) :=              -- prove a_1 = 91 (a_1 is a 0 in 0-indexed notation)
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l194_19453


namespace NUMINAMATH_CALUDE_divisibility_probability_l194_19419

def is_divisible (r k : ℤ) : Prop := ∃ m : ℤ, r = k * m

def count_divisible_pairs : ℕ := 30

def total_pairs : ℕ := 88

theorem divisibility_probability :
  (count_divisible_pairs : ℚ) / (total_pairs : ℚ) = 15 / 44 := by sorry

end NUMINAMATH_CALUDE_divisibility_probability_l194_19419


namespace NUMINAMATH_CALUDE_line_l_standard_equation_l194_19406

/-- A line in 2D space defined by parametric equations. -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given parametric line. -/
def line_l : ParametricLine where
  x := fun t => 1 + t
  y := fun t => -1 + t

/-- The standard form of a line equation: ax + by + c = 0 -/
structure StandardLineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The proposed standard equation of the line. -/
def proposed_equation : StandardLineEquation where
  a := 1
  b := -1
  c := -2

theorem line_l_standard_equation :
  ∀ t : ℝ, proposed_equation.a * (line_l.x t) + proposed_equation.b * (line_l.y t) + proposed_equation.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_l_standard_equation_l194_19406


namespace NUMINAMATH_CALUDE_box_volume_l194_19423

/-- The volume of a rectangular box formed from a cardboard sheet -/
theorem box_volume (initial_length initial_width corner_side : ℝ) 
  (h1 : initial_length = 13)
  (h2 : initial_width = 9)
  (h3 : corner_side = 2) : 
  (initial_length - 2 * corner_side) * (initial_width - 2 * corner_side) * corner_side = 90 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l194_19423


namespace NUMINAMATH_CALUDE_unique_solution_aabb_equation_l194_19444

theorem unique_solution_aabb_equation :
  ∃! (a b n : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1000 * a + 100 * a + 10 * b + b = n^4 - 6 * n^3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_aabb_equation_l194_19444


namespace NUMINAMATH_CALUDE_supplementary_angles_problem_l194_19413

theorem supplementary_angles_problem (A B : Real) : 
  A + B = 180 →  -- angles A and B are supplementary
  A = 7 * B →    -- measure of angle A is 7 times angle B
  A = 157.5 :=   -- prove that measure of angle A is 157.5°
by
  sorry

end NUMINAMATH_CALUDE_supplementary_angles_problem_l194_19413


namespace NUMINAMATH_CALUDE_vector_equation_solution_l194_19436

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)
variable (x y : ℝ)

theorem vector_equation_solution (h_not_collinear : ¬ ∃ (k : ℝ), b = k • a) 
  (h_eq : (2*x - y) • a + 4 • b = 5 • a + (x - 2*y) • b) : 
  x + y = 1 := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l194_19436


namespace NUMINAMATH_CALUDE_annual_savings_l194_19430

/-- Given monthly income and expenses, calculate annual savings --/
theorem annual_savings (monthly_income monthly_expenses : ℕ) : 
  monthly_income = 5000 → 
  monthly_expenses = 4600 → 
  (monthly_income - monthly_expenses) * 12 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_annual_savings_l194_19430


namespace NUMINAMATH_CALUDE_least_square_value_l194_19449

theorem least_square_value (a x y : ℕ+) 
  (h1 : 15 * a + 165 = x^2)
  (h2 : 16 * a - 155 = y^2) :
  min (x^2) (y^2) ≥ 231361 := by
  sorry

end NUMINAMATH_CALUDE_least_square_value_l194_19449


namespace NUMINAMATH_CALUDE_min_value_of_f_fourth_composition_l194_19427

/-- The function f(x) = x^2 + 6x + 7 -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 7

/-- The statement that the minimum value of f(f(f(f(x)))) over all real x is 23 -/
theorem min_value_of_f_fourth_composition :
  ∀ x : ℝ, f (f (f (f x))) ≥ 23 ∧ ∃ y : ℝ, f (f (f (f y))) = 23 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_fourth_composition_l194_19427
