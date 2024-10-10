import Mathlib

namespace pie_crust_flour_redistribution_l3390_339086

theorem pie_crust_flour_redistribution 
  (initial_crusts : ℕ) 
  (initial_flour_per_crust : ℚ) 
  (new_crusts : ℕ) 
  (total_flour : ℚ) 
  (h1 : initial_crusts = 40)
  (h2 : initial_flour_per_crust = 1 / 8)
  (h3 : new_crusts = 25)
  (h4 : total_flour = initial_crusts * initial_flour_per_crust)
  : (total_flour / new_crusts : ℚ) = 1 / 5 := by
  sorry

end pie_crust_flour_redistribution_l3390_339086


namespace triangle_inequality_l3390_339074

/-- Given a triangle with semi-perimeter s, circumradius R, and inradius r,
    prove the inequality relating these quantities. -/
theorem triangle_inequality (s R r : ℝ) (hs : s > 0) (hR : R > 0) (hr : r > 0) :
  2 * Real.sqrt (r * (r + 4 * R)) < 2 * s ∧ 
  2 * s ≤ Real.sqrt (4 * (r + 2 * R)^2 + 2 * R^2) :=
by sorry

end triangle_inequality_l3390_339074


namespace range_of_a_l3390_339031

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 2 3 → (a * x^2 + 2 * y^2) / (x * y) - 1 > 0) → 
  a ∈ Set.Ioi (-1) :=
by sorry

end range_of_a_l3390_339031


namespace marble_distribution_l3390_339005

theorem marble_distribution (n : ℕ) (hn : n = 450) :
  (Finset.filter (fun m : ℕ => m > 1 ∧ n / m > 1) (Finset.range (n + 1))).card = 16 := by
  sorry

end marble_distribution_l3390_339005


namespace smallest_a_value_l3390_339022

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) :
  ∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x)) → a' ≥ 17 := by
sorry

end smallest_a_value_l3390_339022


namespace sqrt_two_minus_one_abs_plus_pi_minus_one_pow_zero_l3390_339038

theorem sqrt_two_minus_one_abs_plus_pi_minus_one_pow_zero (π : ℝ) : 
  |Real.sqrt 2 - 1| + (π - 1)^0 = Real.sqrt 2 := by
  sorry

end sqrt_two_minus_one_abs_plus_pi_minus_one_pow_zero_l3390_339038


namespace lindas_trip_length_l3390_339092

theorem lindas_trip_length :
  ∀ (total_length : ℚ),
  (1 / 4 : ℚ) * total_length + 30 + (1 / 6 : ℚ) * total_length = total_length →
  total_length = 360 / 7 := by
  sorry

end lindas_trip_length_l3390_339092


namespace pure_imaginary_fraction_l3390_339072

theorem pure_imaginary_fraction (a : ℝ) : 
  let z : ℂ := (a - Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = -1 := by
sorry

end pure_imaginary_fraction_l3390_339072


namespace students_with_dogs_l3390_339069

theorem students_with_dogs 
  (total_students : ℕ) 
  (girls_percentage : ℚ) 
  (boys_percentage : ℚ) 
  (girls_with_dogs_percentage : ℚ) 
  (boys_with_dogs_percentage : ℚ) :
  total_students = 100 →
  girls_percentage = 1/2 →
  boys_percentage = 1/2 →
  girls_with_dogs_percentage = 1/5 →
  boys_with_dogs_percentage = 1/10 →
  (girls_percentage * total_students * girls_with_dogs_percentage +
   boys_percentage * total_students * boys_with_dogs_percentage : ℚ) = 15 := by
sorry

end students_with_dogs_l3390_339069


namespace power_difference_equals_eight_l3390_339015

theorem power_difference_equals_eight : 4^2 - 2^3 = 8 := by
  sorry

end power_difference_equals_eight_l3390_339015


namespace f_strictly_increasing_l3390_339089

def f (x : ℝ) : ℝ := x^3 + x^2 - 5*x - 5

theorem f_strictly_increasing :
  (∀ x y, x < y ∧ ((x < -5/3 ∧ y < -5/3) ∨ (x > 1 ∧ y > 1)) → f x < f y) :=
sorry

end f_strictly_increasing_l3390_339089


namespace sin_squared_sum_three_angles_l3390_339068

theorem sin_squared_sum_three_angles (α : ℝ) : 
  (Real.sin (α - Real.pi / 3))^2 + (Real.sin α)^2 + (Real.sin (α + Real.pi / 3))^2 = 3/2 := by
  sorry

end sin_squared_sum_three_angles_l3390_339068


namespace hotel_pricing_l3390_339016

/-- The hotel pricing problem -/
theorem hotel_pricing
  (night_rate : ℝ)
  (night_hours : ℝ)
  (morning_hours : ℝ)
  (initial_money : ℝ)
  (remaining_money : ℝ)
  (h1 : night_rate = 1.5)
  (h2 : night_hours = 6)
  (h3 : morning_hours = 4)
  (h4 : initial_money = 80)
  (h5 : remaining_money = 63)
  : ∃ (morning_rate : ℝ), 
    night_rate * night_hours + morning_rate * morning_hours = initial_money - remaining_money ∧
    morning_rate = 2 := by
  sorry

end hotel_pricing_l3390_339016


namespace seashell_collection_l3390_339017

theorem seashell_collection (joan_daily : ℕ) (jessica_daily : ℕ) (days : ℕ) : 
  joan_daily = 6 → jessica_daily = 8 → days = 7 → 
  (joan_daily + jessica_daily) * days = 98 := by
  sorry

end seashell_collection_l3390_339017


namespace min_value_of_f_l3390_339043

def f (x : ℝ) := 2 * x^3 - 6 * x^2 + 3

theorem min_value_of_f :
  (∀ x ∈ Set.Icc (-2) 2, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-2) 2, f x = 3) →
  (∃ x₀ ∈ Set.Icc (-2) 2, ∀ x ∈ Set.Icc (-2) 2, f x₀ ≤ f x) ∧
  (∀ x ∈ Set.Icc (-2) 2, f x ≥ -37) ∧
  (∃ x ∈ Set.Icc (-2) 2, f x = -37) :=
by sorry

end min_value_of_f_l3390_339043


namespace f_monotone_range_of_a_l3390_339029

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 5*x + 6

-- Define the property of being monotonically increasing on an interval
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem f_monotone_range_of_a :
  {a : ℝ | MonotonicallyIncreasing (f a) 1 3} = {a | a ≤ -3 ∨ a ≥ -3} :=
by sorry

end f_monotone_range_of_a_l3390_339029


namespace evelyn_family_without_daughters_l3390_339091

/-- Represents the family structure of Evelyn and her descendants -/
structure EvelynFamily where
  daughters : ℕ
  granddaughters : ℕ
  daughters_with_daughters : ℕ
  daughters_per_mother : ℕ

/-- The actual family structure of Evelyn -/
def evelyn_family : EvelynFamily :=
  { daughters := 8,
    granddaughters := 36 - 8,
    daughters_with_daughters := (36 - 8) / 7,
    daughters_per_mother := 7 }

/-- The number of Evelyn's daughters and granddaughters who have no daughters -/
def women_without_daughters (f : EvelynFamily) : ℕ :=
  (f.daughters - f.daughters_with_daughters) + f.granddaughters

theorem evelyn_family_without_daughters :
  women_without_daughters evelyn_family = 32 := by
  sorry

end evelyn_family_without_daughters_l3390_339091


namespace even_quadratic_sum_l3390_339036

/-- A quadratic function f(x) = ax^2 + bx defined on [-1, 2] -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

/-- The property of f being an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The theorem stating that if f is even, then a + b = 1/3 -/
theorem even_quadratic_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f a b x = f a b (-x)) →
  a + b = 1/3 :=
by sorry

end even_quadratic_sum_l3390_339036


namespace smallest_total_is_47_l3390_339062

/-- Represents the number of students in each grade --/
structure StudentCounts where
  ninth : ℕ
  seventh : ℕ
  sixth : ℕ

/-- Checks if the given student counts satisfy the required ratios --/
def satisfiesRatios (counts : StudentCounts) : Prop :=
  3 * counts.seventh = 2 * counts.ninth ∧
  7 * counts.sixth = 4 * counts.ninth

/-- The smallest possible total number of students --/
def smallestTotal : ℕ := 47

/-- Theorem stating that the smallest possible total number of students is 47 --/
theorem smallest_total_is_47 :
  ∃ (counts : StudentCounts),
    satisfiesRatios counts ∧
    counts.ninth + counts.seventh + counts.sixth = smallestTotal ∧
    (∀ (other : StudentCounts),
      satisfiesRatios other →
      other.ninth + other.seventh + other.sixth ≥ smallestTotal) :=
  sorry

end smallest_total_is_47_l3390_339062


namespace math_test_paper_probability_l3390_339056

theorem math_test_paper_probability :
  let total_papers : ℕ := 12
  let math_papers : ℕ := 4
  let probability := math_papers / total_papers
  probability = (1 : ℚ) / 3 := by
  sorry

end math_test_paper_probability_l3390_339056


namespace basketball_game_scores_l3390_339097

/-- Represents the quarterly scores of a team -/
structure QuarterlyScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if the scores form an arithmetic sequence -/
def is_arithmetic_sequence (s : QuarterlyScores) : Prop :=
  ∃ d : ℕ, d > 0 ∧ 
    s.q2 = s.q1 + d ∧
    s.q3 = s.q2 + d ∧
    s.q4 = s.q3 + d

/-- Checks if the scores form a geometric sequence -/
def is_geometric_sequence (s : QuarterlyScores) : Prop :=
  ∃ r : ℚ, r > 1 ∧
    s.q2 = s.q1 * r ∧
    s.q3 = s.q2 * r ∧
    s.q4 = s.q3 * r

/-- The main theorem -/
theorem basketball_game_scores 
  (tigers lions : QuarterlyScores)
  (h1 : tigers.q1 = lions.q1)  -- Tied at the end of first quarter
  (h2 : is_arithmetic_sequence tigers)
  (h3 : is_geometric_sequence lions)
  (h4 : (tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4) + 2 = 
        (lions.q1 + lions.q2 + lions.q3 + lions.q4))  -- Lions won by 2 points
  (h5 : tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4 ≤ 100)
  (h6 : lions.q1 + lions.q2 + lions.q3 + lions.q4 ≤ 100)
  : tigers.q1 + tigers.q2 + lions.q1 + lions.q2 = 19 :=
by
  sorry

end basketball_game_scores_l3390_339097


namespace photo_ratio_theorem_l3390_339042

/-- Represents the number of photos in various scenarios --/
structure PhotoCounts where
  initial : ℕ  -- Initial number of photos in the gallery
  firstDay : ℕ  -- Number of photos taken on the first day
  secondDay : ℕ  -- Number of photos taken on the second day
  final : ℕ  -- Final number of photos in the gallery

/-- Theorem stating the ratio of first day photos to initial gallery photos --/
theorem photo_ratio_theorem (p : PhotoCounts) 
  (h1 : p.initial = 400)
  (h2 : p.secondDay = p.firstDay + 120)
  (h3 : p.final = 920)
  (h4 : p.final = p.initial + p.firstDay + p.secondDay) :
  p.firstDay * 2 = p.initial := by
  sorry

#check photo_ratio_theorem

end photo_ratio_theorem_l3390_339042


namespace kenny_monday_jumping_jacks_l3390_339088

/-- Represents the number of jumping jacks Kenny did on each day of the week. -/
structure WeeklyJumpingJacks where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total number of jumping jacks for the week. -/
def totalJumpingJacks (week : WeeklyJumpingJacks) : ℕ :=
  week.sunday + week.monday + week.tuesday + week.wednesday + week.thursday + week.friday + week.saturday

/-- Theorem stating that Kenny must have done 20 jumping jacks on Monday. -/
theorem kenny_monday_jumping_jacks :
  ∃ (this_week : WeeklyJumpingJacks),
    this_week.sunday = 34 ∧
    this_week.tuesday = 0 ∧
    this_week.wednesday = 123 ∧
    this_week.thursday = 64 ∧
    this_week.friday = 23 ∧
    this_week.saturday = 61 ∧
    totalJumpingJacks this_week = 325 ∧
    this_week.monday = 20 := by
  sorry

#check kenny_monday_jumping_jacks

end kenny_monday_jumping_jacks_l3390_339088


namespace equidistant_point_coordinates_l3390_339030

/-- A point with coordinates (4-a, 2a+1) that has equal distances to both coordinate axes -/
structure EquidistantPoint where
  a : ℝ
  equal_distance : |4 - a| = |2*a + 1|

theorem equidistant_point_coordinates (P : EquidistantPoint) :
  (P.a = 1 ∧ (4 - P.a, 2*P.a + 1) = (3, 3)) ∨
  (P.a = -5 ∧ (4 - P.a, 2*P.a + 1) = (9, -9)) := by
  sorry

end equidistant_point_coordinates_l3390_339030


namespace kingdom_animal_percentage_l3390_339003

/-- Represents the number of cats in the kingdom -/
def num_cats : ℕ := 25

/-- Represents the number of hogs in the kingdom -/
def num_hogs : ℕ := 75

/-- The relationship between hogs and cats -/
axiom hogs_cats_relation : num_hogs = 3 * num_cats

/-- The percentage we're looking for -/
def percentage : ℚ := 50

theorem kingdom_animal_percentage :
  (percentage / 100) * (num_cats - 5 : ℚ) = 10 :=
sorry

end kingdom_animal_percentage_l3390_339003


namespace quadratic_sum_l3390_339071

/-- Given a quadratic x^2 - 20x + 36 that can be written as (x + b)^2 + c,
    prove that b + c = -74 -/
theorem quadratic_sum (b c : ℝ) : 
  (∀ x, x^2 - 20*x + 36 = (x + b)^2 + c) → b + c = -74 := by
  sorry

end quadratic_sum_l3390_339071


namespace total_frogs_in_pond_l3390_339045

def frogs_on_lilypads : ℕ := 5
def frogs_on_logs : ℕ := 3
def dozen : ℕ := 12
def baby_frogs_dozens : ℕ := 2

theorem total_frogs_in_pond : 
  frogs_on_lilypads + frogs_on_logs + baby_frogs_dozens * dozen = 32 := by
  sorry

end total_frogs_in_pond_l3390_339045


namespace perpendicular_lines_l3390_339040

theorem perpendicular_lines (b : ℚ) : 
  (∀ x y : ℚ, 2 * x + 3 * y + 4 = 0 → ∃ m₁ : ℚ, y = m₁ * x + (-4/3)) →
  (∀ x y : ℚ, b * x + 3 * y + 4 = 0 → ∃ m₂ : ℚ, y = m₂ * x + (-4/3)) →
  (∃ m₁ m₂ : ℚ, m₁ * m₂ = -1) →
  b = -9/2 :=
by sorry

end perpendicular_lines_l3390_339040


namespace swan_percentage_among_non_ducks_l3390_339095

theorem swan_percentage_among_non_ducks (geese swan heron duck : ℚ) :
  geese = 1/5 →
  swan = 3/10 →
  heron = 1/4 →
  duck = 1/4 →
  geese + swan + heron + duck = 1 →
  swan / (geese + swan + heron) = 2/5 :=
sorry

end swan_percentage_among_non_ducks_l3390_339095


namespace class_representative_count_l3390_339047

theorem class_representative_count (male_students female_students : ℕ) :
  male_students = 26 → female_students = 24 →
  male_students + female_students = 50 :=
by sorry

end class_representative_count_l3390_339047


namespace optimal_production_time_l3390_339049

def shaping_time : ℕ := 15
def firing_time : ℕ := 30
def total_items : ℕ := 75
def total_workers : ℕ := 13

def production_time (shaping_workers : ℕ) (firing_workers : ℕ) : ℕ :=
  let shaping_rounds := (total_items + shaping_workers - 1) / shaping_workers
  let firing_rounds := (total_items + firing_workers - 1) / firing_workers
  max (shaping_rounds * shaping_time) (firing_rounds * firing_time)

theorem optimal_production_time :
  ∃ (shaping_workers firing_workers : ℕ),
    shaping_workers + firing_workers = total_workers ∧
    ∀ (s f : ℕ), s + f = total_workers →
      production_time shaping_workers firing_workers ≤ production_time s f ∧
      production_time shaping_workers firing_workers = 325 :=
by sorry

end optimal_production_time_l3390_339049


namespace cone_lateral_surface_area_l3390_339066

theorem cone_lateral_surface_area 
  (r : ℝ) (V : ℝ) (h : ℝ) (l : ℝ) (S : ℝ) :
  r = 3 →
  V = 12 * Real.pi →
  V = (1/3) * Real.pi * r^2 * h →
  l^2 = r^2 + h^2 →
  S = Real.pi * r * l →
  S = 15 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l3390_339066


namespace right_triangle_area_and_perimeter_l3390_339065

theorem right_triangle_area_and_perimeter :
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  c = 13 →
  a = 5 →
  b > a →
  (1/2 * a * b = 30 ∧ a + b + c = 30) :=
by
  sorry

end right_triangle_area_and_perimeter_l3390_339065


namespace mall_sales_optimal_profit_l3390_339018

/-- Represents the selling prices and profit calculation for products A and B --/
structure ProductSales where
  cost_price : ℝ
  price_a : ℝ
  price_b : ℝ
  sales_a : ℝ → ℝ
  sales_b : ℝ → ℝ
  profit : ℝ → ℝ

/-- The theorem statement based on the given problem --/
theorem mall_sales_optimal_profit (s : ProductSales) : 
  s.cost_price = 20 ∧ 
  20 * s.price_a + 10 * s.price_b = 840 ∧ 
  10 * s.price_a + 15 * s.price_b = 660 ∧
  s.sales_a 0 = 40 ∧
  (∀ m, s.sales_a m = s.sales_a 0 + 10 * m) ∧
  (∀ m, s.price_a - m ≥ s.price_b) ∧
  (∀ m, s.profit m = (s.price_a - m - s.cost_price) * s.sales_a m + (s.price_b - s.cost_price) * s.sales_b m) →
  s.price_a = 30 ∧ 
  s.price_b = 24 ∧ 
  (∃ m, s.sales_a m = s.sales_b m ∧ 
       s.profit m = 810 ∧ 
       ∀ n, s.profit n ≤ s.profit m) := by
  sorry

end mall_sales_optimal_profit_l3390_339018


namespace right_triangle_sides_l3390_339093

theorem right_triangle_sides : ∃! (a b c : ℕ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨ 
   (a = 2 ∧ b = 3 ∧ c = 4) ∨ 
   (a = 4 ∧ b = 5 ∧ c = 6) ∨ 
   (a = 5 ∧ b = 6 ∧ c = 7)) ∧ 
  a^2 + b^2 = c^2 := by
  sorry

end right_triangle_sides_l3390_339093


namespace composition_properties_l3390_339080

variable {X Y V : Type*}
variable (f : X → Y) (g : Y → V)

theorem composition_properties :
  ((∀ x₁ x₂ : X, g (f x₁) = g (f x₂) → x₁ = x₂) → (∀ x₁ x₂ : X, f x₁ = f x₂ → x₁ = x₂)) ∧
  ((∀ v : V, ∃ x : X, g (f x) = v) → (∀ v : V, ∃ y : Y, g y = v)) := by
  sorry

end composition_properties_l3390_339080


namespace partial_fraction_decomposition_l3390_339054

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 4) (h3 : x ≠ -2) :
  (x^2 + 4*x + 11) / ((x - 1)*(x - 4)*(x + 2)) = 
  (-16/9) / (x - 1) + (35/18) / (x - 4) + (1/6) / (x + 2) := by
sorry

end partial_fraction_decomposition_l3390_339054


namespace math_is_90_average_l3390_339014

/-- Represents the scores in three subjects -/
structure Scores where
  physics : ℝ
  chemistry : ℝ
  mathematics : ℝ

/-- Represents the conditions given in the problem -/
def satisfiesConditions (s : Scores) : Prop :=
  s.physics = 80 ∧
  (s.physics + s.chemistry + s.mathematics) / 3 = 80 ∧
  (s.physics + s.chemistry) / 2 = 70 ∧
  ∃ x, (s.physics + x) / 2 = 90 ∧ (x = s.chemistry ∨ x = s.mathematics)

/-- Theorem stating that mathematics is the subject averaging 90 with physics -/
theorem math_is_90_average (s : Scores) (h : satisfiesConditions s) :
  (s.physics + s.mathematics) / 2 = 90 := by
  sorry

end math_is_90_average_l3390_339014


namespace empty_solution_set_range_l3390_339090

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) ↔ -1 < a ∧ a < 0 := by
  sorry

end empty_solution_set_range_l3390_339090


namespace min_product_of_squares_plus_one_l3390_339094

/-- The polynomial P(x) = x^4 + ax^3 + bx^2 + cx + d -/
def P (a b c d x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem min_product_of_squares_plus_one (a b c d : ℝ) (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : b - d ≥ 5)
  (h₂ : P a b c d x₁ = 0)
  (h₃ : P a b c d x₂ = 0)
  (h₄ : P a b c d x₃ = 0)
  (h₅ : P a b c d x₄ = 0) :
  (x₁^2 + 1) * (x₂^2 + 1) * (x₃^2 + 1) * (x₄^2 + 1) ≥ 16 :=
sorry

end min_product_of_squares_plus_one_l3390_339094


namespace eagle_count_theorem_l3390_339026

/-- The total number of unique types of eagles across all sections of the mountain -/
def total_unique_eagles (lower middle upper overlapping : ℕ) : ℕ :=
  lower + middle + upper - overlapping

/-- Theorem stating that the total number of unique types of eagles is 32 -/
theorem eagle_count_theorem (lower middle upper overlapping : ℕ) 
  (h1 : lower = 12)
  (h2 : middle = 8)
  (h3 : upper = 16)
  (h4 : overlapping = 4) :
  total_unique_eagles lower middle upper overlapping = 32 := by
  sorry

end eagle_count_theorem_l3390_339026


namespace geometric_series_equation_solution_l3390_339079

theorem geometric_series_equation_solution (x : ℝ) : 
  (|x| < 0.5) →
  (∑' n, (2*x)^n = 3.4 - 1.2*x) →
  x = 1/3 :=
by
  sorry

end geometric_series_equation_solution_l3390_339079


namespace swimmer_speed_ratio_l3390_339024

theorem swimmer_speed_ratio :
  ∀ (v₁ v₂ : ℝ),
    v₁ > v₂ →
    v₁ > 0 →
    v₂ > 0 →
    (v₁ + v₂) * 3 = 12 →
    (v₁ - v₂) * 6 = 12 →
    v₁ / v₂ = 3 :=
by
  sorry

end swimmer_speed_ratio_l3390_339024


namespace inscribed_squares_ratio_l3390_339098

theorem inscribed_squares_ratio (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧
  (8 - x) / x = 8 / 6 ∧
  (6 - y) / y = 8 / 10 →
  x / y = 36 / 35 := by sorry

end inscribed_squares_ratio_l3390_339098


namespace sum_of_digits_is_23_l3390_339064

/-- A structure representing a four-digit number with unique digits -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  d1_pos : d1 > 0
  d2_pos : d2 > 0
  d3_pos : d3 > 0
  d4_pos : d4 > 0
  d1_lt_10 : d1 < 10
  d2_lt_10 : d2 < 10
  d3_lt_10 : d3 < 10
  d4_lt_10 : d4 < 10
  unique : d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

/-- Theorem stating that for a four-digit number with product of digits 810 and unique digits, the sum of digits is 23 -/
theorem sum_of_digits_is_23 (n : FourDigitNumber) (h : n.d1 * n.d2 * n.d3 * n.d4 = 810) :
  n.d1 + n.d2 + n.d3 + n.d4 = 23 := by
  sorry

end sum_of_digits_is_23_l3390_339064


namespace base_n_representation_of_b_l3390_339006

theorem base_n_representation_of_b (n : ℕ) (a b : ℤ) (x y : ℚ) : 
  n > 9 →
  x^2 - a*x + b = 0 →
  y^2 - a*y + b = 0 →
  (x = n ∨ y = n) →
  2*x - y = 6 →
  a = 2*n + 7 →
  b = 14 :=
by sorry

end base_n_representation_of_b_l3390_339006


namespace polynomial_substitution_l3390_339052

theorem polynomial_substitution (x y : ℝ) :
  y = x + 1 →
  3 * x^3 + 7 * x^2 + 9 * x + 6 = 3 * y^3 - 2 * y^2 + 4 * y + 1 := by
  sorry

end polynomial_substitution_l3390_339052


namespace root_sum_fraction_values_l3390_339002

theorem root_sum_fraction_values (α β γ : ℝ) : 
  (α^3 - α^2 - 2*α + 1 = 0) →
  (β^3 - β^2 - 2*β + 1 = 0) →
  (γ^3 - γ^2 - 2*γ + 1 = 0) →
  (α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0) →
  (α/β + β/γ + γ/α = 3 ∨ α/β + β/γ + γ/α = -4) := by
sorry

end root_sum_fraction_values_l3390_339002


namespace king_paths_count_l3390_339077

/-- The number of paths for a king on a 7x7 chessboard -/
def numPaths : Fin 7 → Fin 7 → ℕ
| ⟨i, hi⟩, ⟨j, hj⟩ =>
  if i = 3 ∧ j = 3 then 0  -- Central cell (4,4) is forbidden
  else if i = 0 ∨ j = 0 then 1  -- First row and column
  else 
    have hi' : i - 1 < 7 := by sorry
    have hj' : j - 1 < 7 := by sorry
    numPaths ⟨i - 1, hi'⟩ ⟨j, hj⟩ + 
    numPaths ⟨i, hi⟩ ⟨j - 1, hj'⟩ + 
    numPaths ⟨i - 1, hi'⟩ ⟨j - 1, hj'⟩

/-- The theorem stating the number of paths for the king -/
theorem king_paths_count : numPaths ⟨6, by simp⟩ ⟨6, by simp⟩ = 5020 := by
  sorry

end king_paths_count_l3390_339077


namespace age_difference_l3390_339051

theorem age_difference (alice_age bob_age : ℕ) : 
  alice_age + 5 = 19 →
  alice_age + 6 = 2 * (bob_age + 6) →
  alice_age - bob_age = 10 := by
sorry

end age_difference_l3390_339051


namespace symmetry_of_point_l3390_339046

def point_symmetric_to_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), -(p.2))

theorem symmetry_of_point :
  let A : ℝ × ℝ := (-1, 2)
  let A' : ℝ × ℝ := point_symmetric_to_origin A
  A' = (1, -2) := by sorry

end symmetry_of_point_l3390_339046


namespace element_in_set_l3390_339012

theorem element_in_set : 
  let M : Set ℕ := {0, 1, 2}
  let a : ℕ := 0
  a ∈ M :=
by sorry

end element_in_set_l3390_339012


namespace biff_wifi_cost_l3390_339011

/-- Proves the hourly cost of WiFi for Biff to break even on a 3-hour bus trip -/
theorem biff_wifi_cost (ticket : ℝ) (snacks : ℝ) (headphones : ℝ) (hourly_rate : ℝ) 
  (trip_duration : ℝ) :
  ticket = 11 →
  snacks = 3 →
  headphones = 16 →
  hourly_rate = 12 →
  trip_duration = 3 →
  ∃ (wifi_cost : ℝ),
    wifi_cost = 2 ∧
    trip_duration * hourly_rate = ticket + snacks + headphones + trip_duration * wifi_cost :=
by sorry

end biff_wifi_cost_l3390_339011


namespace xiao_ming_final_score_l3390_339058

/-- Calculates the final score of a speech contest given individual scores and weights -/
def final_score (speech_image : ℝ) (content : ℝ) (effectiveness : ℝ) 
  (weight_image : ℝ) (weight_content : ℝ) (weight_effectiveness : ℝ) : ℝ :=
  speech_image * weight_image + content * weight_content + effectiveness * weight_effectiveness

/-- Xiao Ming's speech contest scores and weights -/
def xiao_ming_scores : ℝ × ℝ × ℝ := (9, 8, 8)
def xiao_ming_weights : ℝ × ℝ × ℝ := (0.3, 0.4, 0.3)

theorem xiao_ming_final_score :
  final_score xiao_ming_scores.1 xiao_ming_scores.2.1 xiao_ming_scores.2.2
              xiao_ming_weights.1 xiao_ming_weights.2.1 xiao_ming_weights.2.2 = 8.3 := by
  sorry

end xiao_ming_final_score_l3390_339058


namespace sin_721_degrees_equals_sin_1_degree_l3390_339083

theorem sin_721_degrees_equals_sin_1_degree :
  Real.sin (721 * π / 180) = Real.sin (π / 180) := by
  sorry

end sin_721_degrees_equals_sin_1_degree_l3390_339083


namespace inequality_proof_l3390_339021

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (1 / x + 1 / y ≥ 4 / (x + y)) ∧
  (1 / x + 1 / y + 1 / z ≥ 2 / (x + y) + 2 / (y + z) + 2 / (z + x)) := by
  sorry

end inequality_proof_l3390_339021


namespace complex_subtraction_l3390_339099

theorem complex_subtraction (a b : ℂ) (h1 : a = 4 - 2*I) (h2 : b = 3 + 2*I) :
  a - 2*b = -2 - 6*I := by
  sorry

end complex_subtraction_l3390_339099


namespace absolute_value_inequality_l3390_339027

theorem absolute_value_inequality (x : ℝ) : 
  (|5 - x| < 6) ↔ (-1 < x ∧ x < 11) := by sorry

end absolute_value_inequality_l3390_339027


namespace range_of_f_l3390_339035

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -8 ≤ y ∧ y ≤ 8 :=
by sorry

end range_of_f_l3390_339035


namespace specific_trapezoid_dimensions_l3390_339055

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedIsoscelesTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The angle at the base of the trapezoid -/
  baseAngle : ℝ
  /-- The length of the shorter base -/
  shorterBase : ℝ
  /-- The length of the longer base -/
  longerBase : ℝ
  /-- The length of the legs (equal for isosceles trapezoid) -/
  legLength : ℝ

/-- The theorem about the specific trapezoid -/
theorem specific_trapezoid_dimensions (t : CircumscribedIsoscelesTrapezoid) 
  (h_area : t.area = 8)
  (h_angle : t.baseAngle = π / 6) :
  t.shorterBase = 4 - 2 * Real.sqrt 3 ∧ 
  t.longerBase = 4 + 2 * Real.sqrt 3 ∧ 
  t.legLength = 4 := by
  sorry

end specific_trapezoid_dimensions_l3390_339055


namespace min_value_theorem_l3390_339019

theorem min_value_theorem (y₁ y₂ y₃ : ℝ) (h_pos₁ : y₁ > 0) (h_pos₂ : y₂ > 0) (h_pos₃ : y₃ > 0)
  (h_sum : 2 * y₁ + 3 * y₂ + 4 * y₃ = 120) :
  y₁^2 + 4 * y₂^2 + 9 * y₃^2 ≥ 14400 / 29 ∧
  (∃ (y₁' y₂' y₃' : ℝ), y₁'^2 + 4 * y₂'^2 + 9 * y₃'^2 = 14400 / 29 ∧
    2 * y₁' + 3 * y₂' + 4 * y₃' = 120 ∧ y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0) := by
  sorry

end min_value_theorem_l3390_339019


namespace treats_per_day_l3390_339033

def treat_cost : ℚ := 1 / 10
def total_cost : ℚ := 6
def days_in_month : ℕ := 30

theorem treats_per_day :
  (total_cost / treat_cost) / days_in_month = 2 := by sorry

end treats_per_day_l3390_339033


namespace complex_power_difference_l3390_339073

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) :
  (1 + 2*i)^24 - (1 - 2*i)^24 = 0 :=
by sorry

end complex_power_difference_l3390_339073


namespace min_omega_value_l3390_339000

open Real

theorem min_omega_value (ω φ : ℝ) (f : ℝ → ℝ) : 
  ω > 0 → 
  abs φ < π / 2 →
  (∀ x, f x = sin (ω * x + φ)) →
  f 0 = 1 / 2 →
  (∀ x, f x ≤ f (π / 12)) →
  (∀ ω' > 0, (∀ x, sin (ω' * x + φ) ≤ sin (ω' * π / 12 + φ)) → ω' ≥ ω) →
  ω = 4 := by
sorry

end min_omega_value_l3390_339000


namespace num_triangles_equals_closest_integer_l3390_339057

/-- The number of distinct triangles in a regular n-gon -/
def num_triangles (n : ℕ) : ℕ := sorry

/-- The integer closest to n^2/12 -/
def closest_integer (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of distinct triangles in a regular n-gon
    is equal to the integer closest to n^2/12 -/
theorem num_triangles_equals_closest_integer (n : ℕ) (h : n ≥ 3) :
  num_triangles n = closest_integer n := by sorry

end num_triangles_equals_closest_integer_l3390_339057


namespace quadratic_inequality_range_l3390_339050

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end quadratic_inequality_range_l3390_339050


namespace intersection_of_A_and_B_l3390_339009

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_of_A_and_B : A ∩ B = {3, 9} := by
  sorry

end intersection_of_A_and_B_l3390_339009


namespace intersection_of_lines_l3390_339096

theorem intersection_of_lines :
  ∃! (x y : ℚ), (8 * x - 5 * y = 10) ∧ (3 * x + 2 * y = 16) ∧ 
  (x = 100 / 31) ∧ (y = 98 / 31) := by
  sorry

end intersection_of_lines_l3390_339096


namespace cross_section_distance_l3390_339059

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Side length of the base hexagon -/
  base_side : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- Distance from the apex of the pyramid -/
  distance_from_apex : ℝ
  /-- Area of the cross section -/
  area : ℝ

/-- Theorem about the distance of a cross section in a right hexagonal pyramid -/
theorem cross_section_distance
  (pyramid : RightHexagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h_parallel : cs1.distance_from_apex < cs2.distance_from_apex)
  (h_areas : cs1.area = 150 * Real.sqrt 3 ∧ cs2.area = 600 * Real.sqrt 3)
  (h_distance : cs2.distance_from_apex - cs1.distance_from_apex = 8) :
  cs2.distance_from_apex = 16 := by
  sorry

end cross_section_distance_l3390_339059


namespace symmetric_circle_equation_l3390_339060

/-- Given a circle C₁ with equation (x+1)² + (y-1)² = 1 and a line L with equation x - y - 1 = 0,
    the circle C₂ symmetric to C₁ about L has equation (x-2)² + (y+2)² = 1 -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∀ X Y : ℝ, (X + 1)^2 + (Y - 1)^2 = 1 → 
    (X - Y - 1 = 0 → (x + 1 = Y ∧ y - 1 = X) → (x - 2)^2 + (y + 2)^2 = 1)) :=
by sorry

end symmetric_circle_equation_l3390_339060


namespace log_base_2_derivative_l3390_339039

open Real

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (λ x => log x / log 2) x = 1 / (x * log 2) := by
  sorry

end log_base_2_derivative_l3390_339039


namespace six_digit_multiple_of_nine_l3390_339044

theorem six_digit_multiple_of_nine :
  ∃ (d : ℕ), d < 10 ∧ (567890 + d) % 9 = 0 :=
by
  -- The proof goes here
  sorry

end six_digit_multiple_of_nine_l3390_339044


namespace square_areas_sum_l3390_339008

theorem square_areas_sum (a b c : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  a^2 + b^2 = c^2 :=
by sorry

end square_areas_sum_l3390_339008


namespace soccer_lineup_combinations_l3390_339013

def num_goalkeepers : ℕ := 3
def num_defenders : ℕ := 5
def num_midfielders : ℕ := 8
def num_forwards : ℕ := 4

theorem soccer_lineup_combinations : 
  num_goalkeepers * num_defenders * num_midfielders * (num_forwards * (num_forwards - 1)) = 1440 :=
by sorry

end soccer_lineup_combinations_l3390_339013


namespace dog_treat_cost_is_six_l3390_339081

/-- The cost of dog treats for a month -/
def dog_treat_cost (treats_per_day : ℕ) (cost_per_treat : ℚ) (days_in_month : ℕ) : ℚ :=
  (treats_per_day : ℚ) * cost_per_treat * (days_in_month : ℚ)

/-- Theorem: The cost of dog treats for a month under given conditions is $6 -/
theorem dog_treat_cost_is_six :
  dog_treat_cost 2 (1/10) 30 = 6 := by
sorry

end dog_treat_cost_is_six_l3390_339081


namespace cube_root_8000_l3390_339084

theorem cube_root_8000 (c d : ℕ+) (h1 : (8000 : ℝ)^(1/3) = c * d^(1/3)) 
  (h2 : ∀ (k : ℕ+), (8000 : ℝ)^(1/3) = c * k^(1/3) → d ≤ k) : 
  c + d = 21 := by
  sorry

end cube_root_8000_l3390_339084


namespace lollipop_sharing_ratio_l3390_339087

theorem lollipop_sharing_ratio : 
  ∀ (total_lollipops : ℕ) (total_cost : ℚ) (shared_cost : ℚ),
  total_lollipops = 12 →
  total_cost = 3 →
  shared_cost = 3/4 →
  (shared_cost / (total_cost / total_lollipops)) / total_lollipops = 1/4 := by
sorry

end lollipop_sharing_ratio_l3390_339087


namespace solution_set_of_inequality_l3390_339041

/-- An even function that is increasing on (-∞, 0] -/
def EvenIncreasingNonPositive (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y)

/-- The theorem statement -/
theorem solution_set_of_inequality 
  (f : ℝ → ℝ) 
  (h : EvenIncreasingNonPositive f) :
  {x : ℝ | f (x - 1) ≥ f 1} = Set.Icc 0 2 := by sorry

end solution_set_of_inequality_l3390_339041


namespace regular_decagon_interior_angle_measure_l3390_339067

/-- The measure of one interior angle of a regular decagon in degrees. -/
def regular_decagon_interior_angle : ℝ := 144

/-- Theorem: The measure of one interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle_measure :
  regular_decagon_interior_angle = 144 := by
  sorry

end regular_decagon_interior_angle_measure_l3390_339067


namespace no_power_of_two_solution_l3390_339034

theorem no_power_of_two_solution : ¬∃ (a b c k : ℕ), 
  a + b + c = 1001 ∧ 27 * a + 14 * b + c = 2^k :=
by sorry

end no_power_of_two_solution_l3390_339034


namespace angle_sum_l3390_339076

theorem angle_sum (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.sin (α - β) = 5/6) (h4 : Real.tan α / Real.tan β = -1/4) :
  α + β = 7 * π / 6 := by
  sorry

end angle_sum_l3390_339076


namespace division_with_remainder_l3390_339025

theorem division_with_remainder (m k : ℤ) (h : m ≠ 0) : 
  ∃ (q r : ℤ), mk + 1 = m * q + r ∧ q = k ∧ r = 1 :=
sorry

end division_with_remainder_l3390_339025


namespace intersection_A_B_l3390_339032

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x | 2 * x^2 - 9 * x + 9 ≤ 0}

theorem intersection_A_B : A ∩ B = {2, 3} := by sorry

end intersection_A_B_l3390_339032


namespace max_tax_revenue_l3390_339082

-- Define the market conditions
def supply (P : ℝ) : ℝ := 6 * P - 312
def demand_slope : ℝ := 4
def tax_rate : ℝ := 30
def consumer_price : ℝ := 118

-- Define the demand function
def demand (P : ℝ) : ℝ := 688 - demand_slope * P

-- Define the tax revenue function
def tax_revenue (t : ℝ) : ℝ := (288 - 2.4 * t) * t

-- Theorem statement
theorem max_tax_revenue :
  ∃ (t : ℝ), ∀ (t' : ℝ), tax_revenue t ≥ tax_revenue t' ∧ tax_revenue t = 8640 :=
sorry

end max_tax_revenue_l3390_339082


namespace trajectory_difference_latitude_l3390_339078

/-- The latitude at which the difference in trajectory lengths equals the height difference -/
theorem trajectory_difference_latitude (R h : ℝ) (θ : ℝ) 
  (h_pos : h > 0) 
  (r₁_def : R * Real.cos θ = R * Real.cos θ)
  (r₂_def : (R + h) * Real.cos θ = (R + h) * Real.cos θ)
  (s_def : 2 * Real.pi * (R + h) * Real.cos θ - 2 * Real.pi * R * Real.cos θ = h) :
  θ = Real.arccos (1 / (2 * Real.pi)) := by
  sorry

end trajectory_difference_latitude_l3390_339078


namespace hat_cloak_color_probability_l3390_339010

/-- The number of possible hat colors for sixth-graders -/
def num_hat_colors : ℕ := 2

/-- The number of possible cloak colors for seventh-graders -/
def num_cloak_colors : ℕ := 3

/-- The total number of possible color combinations -/
def total_combinations : ℕ := num_hat_colors * num_cloak_colors

/-- The number of combinations where hat and cloak colors are different -/
def different_color_combinations : ℕ := num_hat_colors * (num_cloak_colors - 1)

/-- The probability of hat and cloak colors being different -/
def prob_different_colors : ℚ := different_color_combinations / total_combinations

theorem hat_cloak_color_probability :
  prob_different_colors = 2 / 3 := by sorry

end hat_cloak_color_probability_l3390_339010


namespace fraction_problem_l3390_339048

theorem fraction_problem (x : ℝ) : 
  (0.3 * x = 63.0000000000001) → 
  (∃ f : ℝ, f = 0.4 * x + 12 ∧ f = 96) :=
by sorry

end fraction_problem_l3390_339048


namespace smallest_in_consecutive_odd_integers_l3390_339028

/-- A set of consecutive odd integers -/
def ConsecutiveOddIntegers := Set ℤ

/-- The median of a set of integers -/
def median (s : Set ℤ) : ℤ := sorry

/-- The smallest element in a set of integers -/
def smallest (s : Set ℤ) : ℤ := sorry

/-- The largest element in a set of integers -/
def largest (s : Set ℤ) : ℤ := sorry

theorem smallest_in_consecutive_odd_integers 
  (S : ConsecutiveOddIntegers) 
  (h_median : median S = 152) 
  (h_largest : largest S = 163) : 
  smallest S = 138 := by sorry

end smallest_in_consecutive_odd_integers_l3390_339028


namespace f_even_implies_a_eq_two_l3390_339053

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x * exp(x) / (exp(a*x) - 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

/-- If f(x) = x * exp(x) / (exp(a*x) - 1) is an even function, then a = 2 -/
theorem f_even_implies_a_eq_two (a : ℝ) :
  IsEven (f a) → a = 2 := by
  sorry

end f_even_implies_a_eq_two_l3390_339053


namespace opponent_total_score_l3390_339037

def volleyball_problem (team_scores : List Nat) : Prop :=
  let n := team_scores.length
  n = 6 ∧
  team_scores = [2, 3, 5, 7, 11, 13] ∧
  (∃ lost_scores : List Nat,
    lost_scores.length = 3 ∧
    lost_scores ⊆ team_scores ∧
    (∀ score ∈ lost_scores, ∃ opp_score, opp_score = score + 2)) ∧
  (∃ won_scores : List Nat,
    won_scores.length = 3 ∧
    won_scores ⊆ team_scores ∧
    (∀ score ∈ won_scores, ∃ opp_score, 3 * opp_score = score))

theorem opponent_total_score (team_scores : List Nat) 
  (h : volleyball_problem team_scores) : 
  (List.sum (team_scores.map (λ score => 
    if score ∈ [2, 3, 5] then score + 2 
    else score / 3))) = 25 := by
  sorry

end opponent_total_score_l3390_339037


namespace triangle_inequality_l3390_339085

theorem triangle_inequality (a b c a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a₁ ≥ 0 → b₁ ≥ 0 → c₁ ≥ 0 →
  a₂ ≥ 0 → b₂ ≥ 0 → c₂ ≥ 0 →
  a + b > c → b + c > a → c + a > b →
  a * a₁ * a₂ + b * b₁ * b₂ + c * c₁ * c₂ ≥ a * b * c :=
by sorry

end triangle_inequality_l3390_339085


namespace pete_calculation_l3390_339070

theorem pete_calculation (x y z : ℕ+) : 
  (x + y) * z = 14 ∧ 
  x * y + z = 14 → 
  ∃ (s : Finset ℕ+), s.card = 4 ∧ ∀ a : ℕ+, a ∈ s ↔ 
    ∃ (b c : ℕ+), ((a + b) * c = 14 ∧ a * b + c = 14) := by
  sorry

end pete_calculation_l3390_339070


namespace min_length_AB_l3390_339004

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 2

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Define the chord MN
def chord_MN (M N : ℝ × ℝ) : Prop := 
  circle_C M.1 M.2 ∧ circle_C N.1 N.2

-- Define the perpendicularity condition
def perpendicular_CM_CN (C M N : ℝ × ℝ) : Prop := 
  (M.1 - C.1) * (N.1 - C.1) + (M.2 - C.2) * (N.2 - C.2) = 0

-- Define the midpoint condition
def midpoint_P (P M N : ℝ × ℝ) : Prop := 
  P.1 = (M.1 + N.1) / 2 ∧ P.2 = (M.2 + N.2) / 2

-- Define the angle condition
def angle_APB_geq_pi_div_2 (A P B : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) ≤ 0

-- Main theorem
theorem min_length_AB : 
  ∀ (M N P A B : ℝ × ℝ),
    chord_MN M N →
    perpendicular_CM_CN (2, 4) M N →
    midpoint_P P M N →
    line_l A.1 A.2 →
    line_l B.1 B.2 →
    angle_APB_geq_pi_div_2 A P B →
    (A.1 - B.1)^2 + (A.2 - B.2)^2 ≥ ((6 * Real.sqrt 5) / 5 + 2)^2 :=
sorry

end min_length_AB_l3390_339004


namespace arithmetic_mean_property_l3390_339061

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression (α : Type*) [Add α] [Mul α] where
  a₁ : α  -- First term
  d : α   -- Common difference

variable {α : Type*} [LinearOrderedField α]

/-- The nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression α) (n : ℕ) : α :=
  ap.a₁ + (n - 1 : α) * ap.d

/-- Theorem: In an arithmetic progression, any term (starting from the second)
    is the arithmetic mean of two terms equidistant from it. -/
theorem arithmetic_mean_property (ap : ArithmeticProgression α) (k p : ℕ) 
    (h1 : k ≥ 2) (h2 : p > 0) :
  ap.nthTerm k = (ap.nthTerm (k - p) + ap.nthTerm (k + p)) / 2 := by
  sorry

end arithmetic_mean_property_l3390_339061


namespace train_distance_l3390_339023

/-- Proves that a train traveling at a rate of 1 mile per 1.5 minutes will cover 40 miles in 60 minutes -/
theorem train_distance (rate : ℝ) (time : ℝ) (distance : ℝ) : 
  rate = 1 / 1.5 → time = 60 → distance = rate * time → distance = 40 := by
  sorry

end train_distance_l3390_339023


namespace extreme_points_count_f_nonnegative_range_l3390_339007

/-- The function f(x) defined on (-1, +∞) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * (x^2 - x)

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 1 / (x + 1) + 2 * a * x - a

/-- Theorem about the number of extreme points of f(x) -/
theorem extreme_points_count (a : ℝ) : 
  (a < 0 → ∃! x, x > -1 ∧ f' a x = 0) ∧ 
  (0 ≤ a ∧ a ≤ 8/9 → ∀ x > -1, f' a x ≠ 0) ∧
  (a > 8/9 → ∃ x y, x > -1 ∧ y > -1 ∧ x ≠ y ∧ f' a x = 0 ∧ f' a y = 0) :=
sorry

/-- Theorem about the range of a for which f(x) ≥ 0 when x > 0 -/
theorem f_nonnegative_range : 
  {a : ℝ | ∀ x > 0, f a x ≥ 0} = Set.Icc 0 1 :=
sorry

end extreme_points_count_f_nonnegative_range_l3390_339007


namespace quadratic_inequality_condition_l3390_339001

theorem quadratic_inequality_condition (x : ℝ) : 
  2 * x^2 - 5 * x - 3 ≥ 0 ↔ x ≤ -1/2 ∨ x ≥ 3 := by
  sorry

end quadratic_inequality_condition_l3390_339001


namespace absolute_value_equation_solution_l3390_339075

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |2*x - 8| = 5 - x ↔ x = 13/3 ∨ x = 3 := by
sorry

end absolute_value_equation_solution_l3390_339075


namespace area_XPQ_is_435_div_48_l3390_339063

/-- Triangle XYZ with points P and Q -/
structure TriangleXYZ where
  /-- Length of side XY -/
  xy : ℝ
  /-- Length of side YZ -/
  yz : ℝ
  /-- Length of side XZ -/
  xz : ℝ
  /-- Distance XP on side XY -/
  xp : ℝ
  /-- Distance XQ on side XZ -/
  xq : ℝ
  /-- xy is positive -/
  xy_pos : 0 < xy
  /-- yz is positive -/
  yz_pos : 0 < yz
  /-- xz is positive -/
  xz_pos : 0 < xz
  /-- xp is positive and less than or equal to xy -/
  xp_bounds : 0 < xp ∧ xp ≤ xy
  /-- xq is positive and less than or equal to xz -/
  xq_bounds : 0 < xq ∧ xq ≤ xz

/-- The area of triangle XPQ in the given configuration -/
def areaXPQ (t : TriangleXYZ) : ℝ := sorry

/-- Theorem stating the area of triangle XPQ is 435/48 for the given configuration -/
theorem area_XPQ_is_435_div_48 (t : TriangleXYZ) 
    (h_xy : t.xy = 8) 
    (h_yz : t.yz = 9) 
    (h_xz : t.xz = 10) 
    (h_xp : t.xp = 3) 
    (h_xq : t.xq = 6) : 
  areaXPQ t = 435 / 48 := by
  sorry

end area_XPQ_is_435_div_48_l3390_339063


namespace sebastian_orchestra_size_l3390_339020

/-- Represents the number of musicians in each section of the orchestra -/
structure OrchestraSection :=
  (percussion : ℕ)
  (brass : ℕ)
  (strings : ℕ)
  (woodwinds : ℕ)
  (keyboardsAndHarp : ℕ)
  (conductor : ℕ)

/-- Calculates the total number of musicians in the orchestra -/
def totalMusicians (o : OrchestraSection) : ℕ :=
  o.percussion + o.brass + o.strings + o.woodwinds + o.keyboardsAndHarp + o.conductor

/-- The specific orchestra composition as described in the problem -/
def sebastiansOrchestra : OrchestraSection :=
  { percussion := 4
  , brass := 13
  , strings := 18
  , woodwinds := 10
  , keyboardsAndHarp := 3
  , conductor := 1 }

/-- Theorem stating that the total number of musicians in Sebastian's orchestra is 49 -/
theorem sebastian_orchestra_size :
  totalMusicians sebastiansOrchestra = 49 := by
  sorry


end sebastian_orchestra_size_l3390_339020
