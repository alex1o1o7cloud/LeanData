import Mathlib

namespace min_value_theorem_l2311_231121

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  2 * a + 3 * b + 6 * c ≥ 27 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 27 ∧ 2 * a₀ + 3 * b₀ + 6 * c₀ = 27 :=
by sorry

end min_value_theorem_l2311_231121


namespace intersection_of_A_and_B_l2311_231148

-- Define set A
def A : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {-1, 0, 1, 2, 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end intersection_of_A_and_B_l2311_231148


namespace cubic_inequality_l2311_231170

theorem cubic_inequality (x : ℝ) (h : x ≥ 0) : 3 * x^3 - 6 * x^2 + 4 ≥ 0 := by
  sorry

end cubic_inequality_l2311_231170


namespace triangle_area_equals_sqrt_semiperimeter_l2311_231108

theorem triangle_area_equals_sqrt_semiperimeter 
  (x y z : ℝ) (a b c s Δ : ℝ) 
  (ha : a = x / y + y / z)
  (hb : b = y / z + z / x)
  (hc : c = z / x + x / y)
  (hs : s = (a + b + c) / 2) :
  Δ = Real.sqrt s := by sorry

end triangle_area_equals_sqrt_semiperimeter_l2311_231108


namespace tangent_line_to_two_parabolas_l2311_231134

/-- Given curves C₁: y = x² and C₂: y = -(x - 2)², prove that the line l: y = -2x + 3 is tangent to both C₁ and C₂ -/
theorem tangent_line_to_two_parabolas :
  let C₁ : ℝ → ℝ := λ x ↦ x^2
  let C₂ : ℝ → ℝ := λ x ↦ -(x - 2)^2
  let l : ℝ → ℝ := λ x ↦ -2*x + 3
  (∃ x₁, (C₁ x₁ = l x₁) ∧ (deriv C₁ x₁ = deriv l x₁)) ∧
  (∃ x₂, (C₂ x₂ = l x₂) ∧ (deriv C₂ x₂ = deriv l x₂)) :=
by sorry

end tangent_line_to_two_parabolas_l2311_231134


namespace complex_simplification_l2311_231104

theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  (1 - 3*i) / (1 - i) = 2 - i := by
  sorry

end complex_simplification_l2311_231104


namespace lcm_of_3_8_9_12_l2311_231176

theorem lcm_of_3_8_9_12 : Nat.lcm 3 (Nat.lcm 8 (Nat.lcm 9 12)) = 72 := by
  sorry

end lcm_of_3_8_9_12_l2311_231176


namespace cubic_factorization_l2311_231127

theorem cubic_factorization (x : ℝ) : x^3 - 2*x^2 + x = x*(x-1)^2 := by
  sorry

end cubic_factorization_l2311_231127


namespace total_wheels_l2311_231112

/-- The number of bikes that can be assembled in the garage -/
def bikes_assemblable : ℕ := 10

/-- The number of wheels required for each bike -/
def wheels_per_bike : ℕ := 2

/-- Theorem: The total number of wheels in the garage is 20 -/
theorem total_wheels : bikes_assemblable * wheels_per_bike = 20 := by
  sorry

end total_wheels_l2311_231112


namespace cleanup_time_is_25_minutes_l2311_231143

/-- Represents the toy cleaning scenario -/
structure ToyCleaningScenario where
  totalToys : ℕ
  momPutRate : ℕ
  miaTakeRate : ℕ
  brotherTossRate : ℕ
  momCycleTime : ℕ
  brotherCycleTime : ℕ

/-- Calculates the time taken to clean up all toys -/
def cleanupTime (scenario : ToyCleaningScenario) : ℚ :=
  sorry

/-- Theorem stating that the cleanup time for the given scenario is 25 minutes -/
theorem cleanup_time_is_25_minutes :
  let scenario : ToyCleaningScenario := {
    totalToys := 40,
    momPutRate := 4,
    miaTakeRate := 3,
    brotherTossRate := 1,
    momCycleTime := 20,
    brotherCycleTime := 40
  }
  cleanupTime scenario = 25 := by
  sorry

end cleanup_time_is_25_minutes_l2311_231143


namespace molecular_weight_C6H8O7_moles_l2311_231128

/-- The molecular weight of a single molecule of C6H8O7 in g/mol -/
def molecular_weight_C6H8O7 : ℝ := 192.124

/-- The total molecular weight in grams -/
def total_weight : ℝ := 960

/-- Theorem stating that the molecular weight of a certain number of moles of C6H8O7 is equal to the total weight -/
theorem molecular_weight_C6H8O7_moles : 
  ∃ (n : ℝ), n * molecular_weight_C6H8O7 = total_weight :=
sorry

end molecular_weight_C6H8O7_moles_l2311_231128


namespace adult_males_in_town_l2311_231197

/-- Represents the population distribution in a small town -/
structure TownPopulation where
  total : ℕ
  ratio_children : ℕ
  ratio_adult_males : ℕ
  ratio_adult_females : ℕ

/-- Calculates the number of adult males in the town -/
def adult_males (town : TownPopulation) : ℕ :=
  let total_ratio := town.ratio_children + town.ratio_adult_males + town.ratio_adult_females
  (town.total / total_ratio) * town.ratio_adult_males

/-- Theorem stating the number of adult males in the specific town -/
theorem adult_males_in_town (town : TownPopulation) 
  (h1 : town.total = 480)
  (h2 : town.ratio_children = 1)
  (h3 : town.ratio_adult_males = 2)
  (h4 : town.ratio_adult_females = 2) :
  adult_males town = 192 := by
  sorry

end adult_males_in_town_l2311_231197


namespace gym_towels_theorem_l2311_231118

/-- Represents the number of guests entering the gym each hour -/
structure GymHours :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)

/-- Calculates the total number of towels used based on gym hours -/
def totalTowels (hours : GymHours) : ℕ :=
  hours.first + hours.second + hours.third + hours.fourth

/-- Theorem: Given the specified conditions, the total number of towels used is 285 -/
theorem gym_towels_theorem (hours : GymHours) 
  (h1 : hours.first = 50)
  (h2 : hours.second = hours.first + hours.first / 5)
  (h3 : hours.third = hours.second + hours.second / 4)
  (h4 : hours.fourth = hours.third + hours.third / 3)
  : totalTowels hours = 285 := by
  sorry

#eval totalTowels { first := 50, second := 60, third := 75, fourth := 100 }

end gym_towels_theorem_l2311_231118


namespace mary_warm_hours_l2311_231193

/-- The number of sticks of wood produced by chopping up a chair. -/
def sticksPerChair : ℕ := 6

/-- The number of sticks of wood produced by chopping up a table. -/
def sticksPerTable : ℕ := 9

/-- The number of sticks of wood produced by chopping up a stool. -/
def sticksPerStool : ℕ := 2

/-- The number of sticks of wood Mary needs to burn per hour to stay warm. -/
def sticksPerHour : ℕ := 5

/-- The number of chairs Mary chops up. -/
def numChairs : ℕ := 18

/-- The number of tables Mary chops up. -/
def numTables : ℕ := 6

/-- The number of stools Mary chops up. -/
def numStools : ℕ := 4

/-- Theorem stating that Mary can keep warm for 34 hours with the firewood from the chopped furniture. -/
theorem mary_warm_hours : 
  (numChairs * sticksPerChair + numTables * sticksPerTable + numStools * sticksPerStool) / sticksPerHour = 34 := by
  sorry


end mary_warm_hours_l2311_231193


namespace digit_sum_multiple_of_nine_l2311_231107

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Theorem: If a number n and 3n have the same digit sum, then n is divisible by 9 -/
theorem digit_sum_multiple_of_nine (n : ℕ) : digitSum n = digitSum (3 * n) → 9 ∣ n := by
  sorry

end digit_sum_multiple_of_nine_l2311_231107


namespace soccer_camp_ratio_l2311_231114

/-- Soccer camp ratio problem -/
theorem soccer_camp_ratio :
  ∀ (total_kids soccer_kids : ℕ),
  total_kids = 2000 →
  soccer_kids * 3 = 750 * 4 →
  soccer_kids * 2 = total_kids :=
by
  sorry

end soccer_camp_ratio_l2311_231114


namespace women_in_sports_club_l2311_231161

/-- The number of women in a sports club -/
def number_of_women (total_members participants : ℕ) : ℕ :=
  let women := 3 * (total_members - participants) / 2
  women

/-- Theorem: The number of women in the sports club is 21 -/
theorem women_in_sports_club :
  number_of_women 36 22 = 21 :=
by
  sorry

end women_in_sports_club_l2311_231161


namespace sheets_per_class_calculation_l2311_231154

/-- The number of sheets of paper used by the school per week -/
def sheets_per_week : ℕ := 9000

/-- The number of school days per week -/
def school_days_per_week : ℕ := 5

/-- The number of classes in the school -/
def num_classes : ℕ := 9

/-- The number of sheets of paper each class uses per day -/
def sheets_per_class_per_day : ℕ := sheets_per_week / school_days_per_week / num_classes

theorem sheets_per_class_calculation :
  sheets_per_class_per_day = 200 := by
  sorry

end sheets_per_class_calculation_l2311_231154


namespace blue_red_face_ratio_l2311_231181

theorem blue_red_face_ratio (n : ℕ) (h : n = 13) : 
  let red_area := 6 * n^2
  let total_area := 6 * n^3
  let blue_area := total_area - red_area
  blue_area / red_area = 12 := by sorry

end blue_red_face_ratio_l2311_231181


namespace water_height_in_cylinder_l2311_231142

/-- The height of water in a cylinder when poured from a cone -/
theorem water_height_in_cylinder (cone_radius cone_height cyl_radius : ℝ) 
  (h_cone_radius : cone_radius = 12)
  (h_cone_height : cone_height = 18)
  (h_cyl_radius : cyl_radius = 24) : 
  (1 / 3 * π * cone_radius^2 * cone_height) / (π * cyl_radius^2) = 1.5 := by
  sorry

#check water_height_in_cylinder

end water_height_in_cylinder_l2311_231142


namespace negative_nine_less_than_negative_two_l2311_231157

theorem negative_nine_less_than_negative_two : -9 < -2 := by
  sorry

end negative_nine_less_than_negative_two_l2311_231157


namespace ralph_square_matchsticks_ralph_uses_eight_matchsticks_per_square_l2311_231196

theorem ralph_square_matchsticks (total_matchsticks : ℕ) (elvis_matchsticks_per_square : ℕ) 
  (elvis_squares : ℕ) (ralph_squares : ℕ) (matchsticks_left : ℕ) : ℕ :=
  let elvis_total_matchsticks := elvis_matchsticks_per_square * elvis_squares
  let total_used_matchsticks := total_matchsticks - matchsticks_left
  let ralph_total_matchsticks := total_used_matchsticks - elvis_total_matchsticks
  ralph_total_matchsticks / ralph_squares

theorem ralph_uses_eight_matchsticks_per_square :
  ralph_square_matchsticks 50 4 5 3 6 = 8 := by
  sorry

end ralph_square_matchsticks_ralph_uses_eight_matchsticks_per_square_l2311_231196


namespace axis_of_symmetry_l2311_231171

-- Define a function f that satisfies the symmetry condition
def f (x : ℝ) : ℝ := sorry

-- State the symmetry condition
axiom symmetry_condition : ∀ x, f x = f (4 - x)

-- Define what it means for a line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Theorem statement
theorem axis_of_symmetry :
  is_axis_of_symmetry 2 :=
sorry

end axis_of_symmetry_l2311_231171


namespace a_not_periodic_l2311_231173

/-- The first digit of a positive integer -/
def firstDigit (n : ℕ) : ℕ :=
  if n < 10 then n else firstDigit (n / 10)

/-- The sequence a_n where a_n is the first digit of n^2 -/
def a (n : ℕ) : ℕ :=
  firstDigit (n * n)

/-- A sequence is periodic if there exists a positive integer p such that
    for all n ≥ some N, a(n+p) = a(n) -/
def isPeriodic (f : ℕ → ℕ) : Prop :=
  ∃ p N : ℕ, p > 0 ∧ ∀ n ≥ N, f (n + p) = f n

/-- The sequence a_n is not periodic -/
theorem a_not_periodic : ¬ isPeriodic a := by
  sorry

end a_not_periodic_l2311_231173


namespace morning_shells_count_l2311_231139

/-- The number of shells Lino picked up in the afternoon -/
def afternoon_shells : ℕ := 324

/-- The total number of shells Lino picked up -/
def total_shells : ℕ := 616

/-- The number of shells Lino picked up in the morning -/
def morning_shells : ℕ := total_shells - afternoon_shells

theorem morning_shells_count : morning_shells = 292 := by
  sorry

end morning_shells_count_l2311_231139


namespace wrexham_orchestra_max_members_l2311_231140

theorem wrexham_orchestra_max_members :
  ∀ m : ℕ,
  (∃ k : ℕ, 30 * m = 31 * k + 7) →
  30 * m < 1200 →
  (∀ n : ℕ, (∃ j : ℕ, 30 * n = 31 * j + 7) → 30 * n < 1200 → 30 * n ≤ 30 * m) →
  30 * m = 720 :=
by sorry

end wrexham_orchestra_max_members_l2311_231140


namespace wuyang_football_school_runners_l2311_231194

theorem wuyang_football_school_runners (x : ℕ) : 
  (x - 4) % 2 = 0 →
  (x - 5) % 3 = 0 →
  x % 5 = 0 →
  ∃ n : ℕ, x = n ^ 2 →
  250 - 10 ≤ x - 3 ∧ x - 3 ≤ 250 + 10 →
  x = 260 := by
sorry

end wuyang_football_school_runners_l2311_231194


namespace projectile_max_height_l2311_231106

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 120 * t + 36

/-- The time at which the maximum height occurs -/
def t_max : ℝ := 3

/-- The maximum height reached by the projectile -/
def h_max : ℝ := 216

theorem projectile_max_height :
  (∀ t, h t ≤ h_max) ∧ h t_max = h_max := by sorry

end projectile_max_height_l2311_231106


namespace die_game_expected_value_l2311_231190

/-- A fair 8-sided die game where you win the rolled amount if it's a multiple of 3 -/
def die_game : ℝ := by sorry

/-- The expected value of the die game -/
theorem die_game_expected_value : die_game = 2.25 := by sorry

end die_game_expected_value_l2311_231190


namespace grade_distribution_l2311_231160

theorem grade_distribution (frac_A frac_B frac_C frac_D : ℝ) 
  (h1 : frac_A = 0.6)
  (h2 : frac_B = 0.25)
  (h3 : frac_C = 0.1)
  (h4 : frac_D = 0.05) :
  frac_A + frac_B + frac_C + frac_D = 1 := by
  sorry

end grade_distribution_l2311_231160


namespace complex_cube_equation_l2311_231105

theorem complex_cube_equation (a b : ℕ+) :
  (↑a + ↑b * Complex.I) ^ 3 = (2 : ℂ) + 11 * Complex.I →
  ↑a + ↑b * Complex.I = (2 : ℂ) + Complex.I := by
  sorry

end complex_cube_equation_l2311_231105


namespace union_of_sets_l2311_231183

theorem union_of_sets (M N : Set ℕ) : 
  M = {0, 1, 3} → 
  N = {x | ∃ a ∈ M, x = 3 * a} → 
  M ∪ N = {0, 1, 3, 9} := by
sorry

end union_of_sets_l2311_231183


namespace triangle_height_l2311_231163

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 3 → area = 9 → area = (base * height) / 2 → height = 6 := by
  sorry

end triangle_height_l2311_231163


namespace stratified_sampling_example_l2311_231198

/-- Represents a population divided into two strata --/
structure Population :=
  (male_count : ℕ)
  (female_count : ℕ)

/-- Represents a sample taken from a population --/
structure Sample :=
  (male_count : ℕ)
  (female_count : ℕ)

/-- Defines a stratified sampling method --/
def is_stratified_sampling (pop : Population) (samp : Sample) : Prop :=
  (pop.male_count : ℚ) / (pop.male_count + pop.female_count) =
  (samp.male_count : ℚ) / (samp.male_count + samp.female_count)

/-- The theorem to be proved --/
theorem stratified_sampling_example :
  let pop := Population.mk 500 400
  let samp := Sample.mk 25 20
  is_stratified_sampling pop samp :=
by sorry

end stratified_sampling_example_l2311_231198


namespace lines_perpendicular_l2311_231102

/-- Two lines with slopes that are roots of x^2 - mx - 1 = 0 are perpendicular --/
theorem lines_perpendicular (m : ℝ) (k₁ k₂ : ℝ) : 
  k₁^2 - m*k₁ - 1 = 0 → k₂^2 - m*k₂ - 1 = 0 → k₁ * k₂ = -1 := by
  sorry

#check lines_perpendicular

end lines_perpendicular_l2311_231102


namespace grape_rate_calculation_l2311_231179

/-- The rate per kg for grapes -/
def grape_rate : ℝ := 68

/-- The amount of grapes purchased in kg -/
def grape_amount : ℝ := 7

/-- The rate per kg for mangoes -/
def mango_rate : ℝ := 48

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℝ := 9

/-- The total amount paid -/
def total_paid : ℝ := 908

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end grape_rate_calculation_l2311_231179


namespace soccer_ball_properties_l2311_231126

/-- A soccer-ball polyhedron has faces that are m-gons or n-gons (m ≠ n),
    and in every vertex, three faces meet: two m-gons and one n-gon. -/
structure SoccerBallPolyhedron where
  m : ℕ
  n : ℕ
  m_ne_n : m ≠ n
  vertex_config : 2 * ((m - 2) * π / m) + ((n - 2) * π / n) = 2 * π

theorem soccer_ball_properties (P : SoccerBallPolyhedron) :
  Even P.m ∧ P.m = 6 ∧ P.n = 5 := by
  sorry

#check soccer_ball_properties

end soccer_ball_properties_l2311_231126


namespace sum_lent_problem_l2311_231145

theorem sum_lent_problem (P : ℝ) : 
  P > 0 →  -- Assuming the sum lent is positive
  (8 * 0.06 * P) = P - 572 → 
  P = 1100 := by
  sorry

end sum_lent_problem_l2311_231145


namespace parallel_vectors_k_value_l2311_231164

/-- Given two vectors are parallel if their coordinates are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 = t * w.1 ∧ v.2 = t * w.2

theorem parallel_vectors_k_value :
  let e₁ : ℝ × ℝ := (1, 0)
  let e₂ : ℝ × ℝ := (0, 1)
  let a : ℝ × ℝ := (e₁.1 - 2 * e₂.1, e₁.2 - 2 * e₂.2)
  ∀ k : ℝ,
    let b : ℝ × ℝ := (k * e₁.1 + e₂.1, k * e₁.2 + e₂.2)
    are_parallel a b → k = -1/2 :=
by sorry

end parallel_vectors_k_value_l2311_231164


namespace quadratic_polynomial_inequality_l2311_231109

/-- A quadratic polynomial with non-negative coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonneg : 0 ≤ a
  b_nonneg : 0 ≤ b
  c_nonneg : 0 ≤ c

/-- The evaluation of a quadratic polynomial at a point -/
def QuadraticPolynomial.eval (P : QuadraticPolynomial) (x : ℝ) : ℝ :=
  P.a * x^2 + P.b * x + P.c

/-- The statement of the theorem -/
theorem quadratic_polynomial_inequality (P : QuadraticPolynomial) (x y : ℝ) :
  (P.eval (x * y))^2 ≤ (P.eval (x^2)) * (P.eval (y^2)) := by sorry

end quadratic_polynomial_inequality_l2311_231109


namespace blue_area_after_transformations_l2311_231122

/-- Represents the fraction of blue area remaining after a single transformation -/
def blue_fraction_after_one_transform : ℚ := 3/4

/-- Represents the number of transformations -/
def num_transformations : ℕ := 3

/-- Represents the fraction of the original area that remains blue after all transformations -/
def final_blue_fraction : ℚ := (blue_fraction_after_one_transform) ^ num_transformations

theorem blue_area_after_transformations :
  final_blue_fraction = 27/64 := by sorry

end blue_area_after_transformations_l2311_231122


namespace point_inside_circle_range_l2311_231123

/-- A point (x, y) is inside a circle if the left side of the circle's equation is less than the right side -/
def is_inside_circle (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*y - 4 < 0

/-- The theorem stating the range of a for which the point (a+1, a-1) is inside the given circle -/
theorem point_inside_circle_range (a : ℝ) :
  is_inside_circle (a+1) (a-1) a ↔ a < 1 := by
  sorry

end point_inside_circle_range_l2311_231123


namespace point_in_first_quadrant_l2311_231125

/-- The point corresponding to (1+3i)(3-i) is located in the first quadrant of the complex plane. -/
theorem point_in_first_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end point_in_first_quadrant_l2311_231125


namespace students_liking_both_desserts_l2311_231131

theorem students_liking_both_desserts 
  (total_students : ℕ) 
  (apple_pie_lovers : ℕ) 
  (chocolate_cake_lovers : ℕ) 
  (neither_dessert_lovers : ℕ) 
  (h1 : total_students = 35)
  (h2 : apple_pie_lovers = 20)
  (h3 : chocolate_cake_lovers = 17)
  (h4 : neither_dessert_lovers = 10) :
  total_students - neither_dessert_lovers + apple_pie_lovers + chocolate_cake_lovers - total_students = 12 := by
sorry

end students_liking_both_desserts_l2311_231131


namespace f_negative_eight_equals_three_l2311_231185

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def has_period_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) = f x + 2 * f 2

theorem f_negative_eight_equals_three
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_period : has_period_property f)
  (h_f_zero : f 0 = 3) :
  f (-8) = 3 := by
sorry

end f_negative_eight_equals_three_l2311_231185


namespace parabola_point_relation_l2311_231120

-- Define the parabola function
def parabola (x c : ℝ) : ℝ := -x^2 + 6*x + c

-- Define the theorem
theorem parabola_point_relation (c y₁ y₂ y₃ : ℝ) :
  parabola 1 c = y₁ →
  parabola 3 c = y₂ →
  parabola 4 c = y₃ →
  y₂ > y₃ ∧ y₃ > y₁ := by
  sorry


end parabola_point_relation_l2311_231120


namespace song_size_calculation_l2311_231178

/-- Given a total number of songs and total memory space occupied,
    calculate the size of each song. -/
def song_size (total_songs : ℕ) (total_memory : ℕ) : ℚ :=
  total_memory / total_songs

theorem song_size_calculation :
  let morning_songs : ℕ := 10
  let later_songs : ℕ := 15
  let night_songs : ℕ := 3
  let total_songs : ℕ := morning_songs + later_songs + night_songs
  let total_memory : ℕ := 140
  song_size total_songs total_memory = 5 := by
  sorry

end song_size_calculation_l2311_231178


namespace perp_planes_necessary_not_sufficient_l2311_231117

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Main theorem
theorem perp_planes_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (h_diff : α ≠ β)
  (h_subset : subset_line_plane m α) :
  (perp_planes α β → perp_line_plane m β) ∧
  ¬(perp_line_plane m β → perp_planes α β) :=
sorry

end perp_planes_necessary_not_sufficient_l2311_231117


namespace train_length_l2311_231101

/-- The length of a train given specific conditions --/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 200 →
  passing_time = 41 →
  (train_speed - jogger_speed) * passing_time - initial_distance = 210 := by
  sorry

end train_length_l2311_231101


namespace rectangular_solid_diagonal_l2311_231153

/-- 
Given a rectangular solid with edge lengths a, b, and c,
if the total surface area is 22 and the total edge length is 24,
then the length of any interior diagonal is √14.
-/
theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 22)
  (h2 : 4 * (a + b + c) = 24) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 14 := by
  sorry

end rectangular_solid_diagonal_l2311_231153


namespace monotonic_at_most_one_solution_l2311_231146

-- Define a monotonic function
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ f x ≥ f y

-- State the theorem
theorem monotonic_at_most_one_solution (f : ℝ → ℝ) (c : ℝ) (h : Monotonic f) :
  ∃! x, f x = c ∨ (∀ x, f x ≠ c) :=
sorry

end monotonic_at_most_one_solution_l2311_231146


namespace tan_alpha_value_l2311_231168

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo π (3 * π / 2))
  (h2 : Real.tan (2 * α) = -Real.cos α / (2 + Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end tan_alpha_value_l2311_231168


namespace wire_ratio_proof_l2311_231186

theorem wire_ratio_proof (total_length shorter_length : ℕ) 
  (h1 : total_length = 140)
  (h2 : shorter_length = 40) :
  shorter_length * 5 = (total_length - shorter_length) * 2 := by
sorry

end wire_ratio_proof_l2311_231186


namespace arccos_one_over_sqrt_two_l2311_231135

theorem arccos_one_over_sqrt_two (π : Real) :
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end arccos_one_over_sqrt_two_l2311_231135


namespace expansion_coefficients_l2311_231144

theorem expansion_coefficients :
  let expr := (1 + X^5 + X^7)^20
  ∃ (p : Polynomial ℤ),
    p = expr ∧
    p.coeff 18 = 0 ∧
    p.coeff 17 = 3420 :=
by sorry

end expansion_coefficients_l2311_231144


namespace jamies_class_girls_l2311_231175

theorem jamies_class_girls (total : ℕ) (girls boys : ℕ) : 
  total = 35 →
  4 * girls = 3 * boys →
  girls + boys = total →
  girls = 15 := by
sorry

end jamies_class_girls_l2311_231175


namespace balls_in_boxes_l2311_231191

theorem balls_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 2) :
  (k : ℕ) ^ n = 64 := by
  sorry

end balls_in_boxes_l2311_231191


namespace brown_dogs_l2311_231152

def kennel (total : ℕ) (long_fur : ℕ) (neither : ℕ) : Prop :=
  total = 45 ∧
  long_fur = 36 ∧
  neither = 8 ∧
  long_fur ≤ total ∧
  neither ≤ total - long_fur

theorem brown_dogs (total long_fur neither : ℕ) 
  (h : kennel total long_fur neither) : ∃ brown : ℕ, brown = 37 :=
sorry

end brown_dogs_l2311_231152


namespace complex_sequence_counterexample_l2311_231155

-- Define the "sequence" relation on complex numbers
def complex_gt (z₁ z₂ : ℂ) : Prop :=
  z₁.re > z₂.re ∨ (z₁.re = z₂.re ∧ z₁.im > z₂.im)

-- Define positive complex numbers
def complex_pos (z : ℂ) : Prop :=
  complex_gt z 0

-- Theorem statement
theorem complex_sequence_counterexample :
  ∃ (z z₁ z₂ : ℂ), complex_pos z ∧ complex_gt z₁ z₂ ∧ ¬(complex_gt (z * z₁) (z * z₂)) := by
  sorry

end complex_sequence_counterexample_l2311_231155


namespace total_rats_l2311_231169

theorem total_rats (elodie hunter kenia : ℕ) : 
  elodie = 30 →
  elodie = hunter + 10 →
  kenia = 3 * (elodie + hunter) →
  elodie + hunter + kenia = 200 := by
sorry

end total_rats_l2311_231169


namespace not_algorithm_quadratic_roots_l2311_231133

/-- Represents a statement that might be an algorithm --/
inductive Statement
  | travel_plan : Statement
  | linear_equation_steps : Statement
  | quadratic_equation_roots : Statement
  | sum_calculation : Statement

/-- Predicate to determine if a statement is an algorithm --/
def is_algorithm (s : Statement) : Prop :=
  match s with
  | Statement.travel_plan => True
  | Statement.linear_equation_steps => True
  | Statement.quadratic_equation_roots => False
  | Statement.sum_calculation => True

theorem not_algorithm_quadratic_roots :
  ¬(is_algorithm Statement.quadratic_equation_roots) ∧
  (is_algorithm Statement.travel_plan) ∧
  (is_algorithm Statement.linear_equation_steps) ∧
  (is_algorithm Statement.sum_calculation) := by
  sorry

end not_algorithm_quadratic_roots_l2311_231133


namespace coordinates_of_point_B_l2311_231151

/-- Given a 2D coordinate system with origin O, point A at (-1, 2), 
    and vector BA = (3, 3), prove that the coordinates of point B are (-4, -1) -/
theorem coordinates_of_point_B (O A B : ℝ × ℝ) : 
  O = (0, 0) → 
  A = (-1, 2) → 
  B - A = (3, 3) →
  B = (-4, -1) := by
sorry

end coordinates_of_point_B_l2311_231151


namespace n_pointed_star_value_l2311_231137

/-- Represents an n-pointed star. -/
structure PointedStar where
  n : ℕ
  segment_length : ℝ
  angle_a : ℝ
  angle_b : ℝ

/-- Theorem stating the properties of the n-pointed star and the value of n. -/
theorem n_pointed_star_value (star : PointedStar) :
  star.segment_length = 2 * star.n ∧
  star.angle_a = star.angle_b - 10 ∧
  star.n > 2 →
  star.n = 36 := by
  sorry

end n_pointed_star_value_l2311_231137


namespace product_closest_to_63_l2311_231165

theorem product_closest_to_63 : 
  let product := 2.1 * (30.3 + 0.13)
  ∀ x ∈ ({55, 60, 63, 65, 70} : Set ℝ), 
    x ≠ 63 → |product - 63| < |product - x| := by
  sorry

end product_closest_to_63_l2311_231165


namespace inequality_solution_set_l2311_231158

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 2 ↔ (m - 1) * x < Real.sqrt (4 * x - x^2)) → 
  m = 2 := by
sorry

end inequality_solution_set_l2311_231158


namespace problem_solution_l2311_231188

theorem problem_solution (x y : ℝ) (h1 : x^(2*y) = 81) (h2 : x = 9) : y = 1 := by
  sorry

end problem_solution_l2311_231188


namespace hot_dogs_remainder_l2311_231195

theorem hot_dogs_remainder : 25197631 % 17 = 10 := by
  sorry

end hot_dogs_remainder_l2311_231195


namespace bird_cost_l2311_231124

/-- The cost of birds at a pet store -/
theorem bird_cost (small_bird_cost large_bird_cost : ℚ) : 
  large_bird_cost = 2 * small_bird_cost →
  5 * large_bird_cost + 3 * small_bird_cost = 
    5 * small_bird_cost + 3 * large_bird_cost + 20 →
  small_bird_cost = 10 ∧ large_bird_cost = 20 := by
sorry

end bird_cost_l2311_231124


namespace eulers_formula_l2311_231110

/-- A closed polyhedron is a structure with a number of edges, faces, and vertices. -/
structure ClosedPolyhedron where
  edges : ℕ
  faces : ℕ
  vertices : ℕ

/-- Euler's formula for polyhedra states that for any closed polyhedron, 
    the number of edges plus 2 is equal to the sum of the number of faces and vertices. -/
theorem eulers_formula (p : ClosedPolyhedron) : p.edges + 2 = p.faces + p.vertices := by
  sorry

end eulers_formula_l2311_231110


namespace petes_journey_distance_l2311_231136

/-- Represents the distance of each segment of Pete's journey in blocks -/
structure JourneySegments where
  toGarage : ℕ
  toPostOffice : ℕ
  toLibrary : ℕ
  toFriend : ℕ

/-- Calculates the total distance of Pete's round trip journey -/
def totalDistance (segments : JourneySegments) : ℕ :=
  2 * (segments.toGarage + segments.toPostOffice + segments.toLibrary + segments.toFriend)

/-- Pete's actual journey segments -/
def petesJourney : JourneySegments :=
  { toGarage := 5
  , toPostOffice := 20
  , toLibrary := 8
  , toFriend := 10 }

/-- Theorem stating that Pete's total journey distance is 86 blocks -/
theorem petes_journey_distance : totalDistance petesJourney = 86 := by
  sorry


end petes_journey_distance_l2311_231136


namespace valid_integers_count_l2311_231138

/-- The number of permutations of 6 distinct elements -/
def total_permutations : ℕ := 720

/-- The number of permutations satisfying the first condition (1 left of 2) -/
def permutations_condition1 : ℕ := total_permutations / 2

/-- The number of permutations satisfying both conditions (1 left of 2 and 3 left of 4) -/
def permutations_both_conditions : ℕ := permutations_condition1 / 2

/-- Theorem stating the number of valid 6-digit integers -/
theorem valid_integers_count : permutations_both_conditions = 180 := by sorry

end valid_integers_count_l2311_231138


namespace print_shop_cost_difference_l2311_231103

/-- Calculates the total cost for color copies at a print shop --/
def calculate_total_cost (base_price : ℝ) (quantity : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let base_cost := base_price * quantity
  let discounted_cost := if quantity > discount_threshold then base_cost * (1 - discount_rate) else base_cost
  discounted_cost * (1 + tax_rate)

/-- Proves that the difference in cost for 40 color copies between Print Shop Y and Print Shop X is $27.40 --/
theorem print_shop_cost_difference : 
  let shop_x_cost := calculate_total_cost 1.20 40 30 0.10 0.05
  let shop_y_cost := calculate_total_cost 1.70 40 50 0.15 0.07
  shop_y_cost - shop_x_cost = 27.40 := by
  sorry

end print_shop_cost_difference_l2311_231103


namespace gcd_of_72_120_168_l2311_231119

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end gcd_of_72_120_168_l2311_231119


namespace subtraction_of_decimals_l2311_231184

theorem subtraction_of_decimals : 25.019 - 3.2663 = 21.7527 := by
  sorry

end subtraction_of_decimals_l2311_231184


namespace sin_double_angle_circle_l2311_231100

theorem sin_double_angle_circle (α : Real) :
  let P : ℝ × ℝ := (1, 2)
  let r : ℝ := Real.sqrt (P.1^2 + P.2^2)
  (P.1^2 + P.2^2 = r^2) →  -- Point P is on the circle
  (P.1 = r * Real.cos α ∧ P.2 = r * Real.sin α) →  -- P is on the terminal side of α
  Real.sin (2 * α) = 4/5 := by
sorry

end sin_double_angle_circle_l2311_231100


namespace sum_of_square_and_two_cubes_sum_of_square_and_three_cubes_l2311_231166

-- Part (a)
theorem sum_of_square_and_two_cubes (k : ℤ) :
  ∃ (a b c : ℤ), 3 * k - 2 = a^2 + b^3 + c^3 := by sorry

-- Part (b)
theorem sum_of_square_and_three_cubes (n : ℤ) :
  ∃ (w x y z : ℤ), n = w^2 + x^3 + y^3 + z^3 := by sorry

end sum_of_square_and_two_cubes_sum_of_square_and_three_cubes_l2311_231166


namespace not_square_sum_divisor_l2311_231159

theorem not_square_sum_divisor (n : ℕ) (d : ℕ) (h : d ∣ 2 * n^2) :
  ¬∃ (x : ℕ), n^2 + d = x^2 := by
sorry

end not_square_sum_divisor_l2311_231159


namespace smallest_permutation_number_is_1089_l2311_231113

/-- A function that returns true if two natural numbers are permutations of each other's digits -/
def is_digit_permutation (a b : ℕ) : Prop := sorry

/-- A function that returns the smallest natural number satisfying the permutation condition when multiplied by 9 -/
noncomputable def smallest_permutation_number : ℕ := sorry

theorem smallest_permutation_number_is_1089 :
  smallest_permutation_number = 1089 ∧
  is_digit_permutation smallest_permutation_number (9 * smallest_permutation_number) ∧
  ∀ n < smallest_permutation_number, ¬is_digit_permutation n (9 * n) :=
by sorry

end smallest_permutation_number_is_1089_l2311_231113


namespace inheritance_problem_l2311_231199

theorem inheritance_problem (x y z w : ℚ) : 
  (y = 0.75 * x) →
  (z = 0.5 * x) →
  (w = 0.25 * x) →
  (y = 45) →
  (z = 2 * w) →
  (x + y + z + w = 150) :=
by sorry

end inheritance_problem_l2311_231199


namespace jerome_money_theorem_l2311_231177

/-- Calculates the amount of money Jerome has left after giving money to Meg and Bianca. -/
def jerome_money_left (initial_money : ℕ) (meg_amount : ℕ) (bianca_multiplier : ℕ) : ℕ :=
  initial_money - meg_amount - (meg_amount * bianca_multiplier)

/-- Proves that Jerome has $54 left after giving money to Meg and Bianca. -/
theorem jerome_money_theorem :
  let initial_money := 43 * 2
  let meg_amount := 8
  let bianca_multiplier := 3
  jerome_money_left initial_money meg_amount bianca_multiplier = 54 := by
  sorry

end jerome_money_theorem_l2311_231177


namespace max_identical_papers_l2311_231116

def heart_stickers : ℕ := 240
def star_stickers : ℕ := 162
def smiley_stickers : ℕ := 90
def sun_stickers : ℕ := 54

def ratio_heart_to_smiley (n : ℕ) : Prop :=
  2 * (n * smiley_stickers) = n * heart_stickers

def ratio_star_to_sun (n : ℕ) : Prop :=
  3 * (n * sun_stickers) = n * star_stickers

def all_stickers_used (n : ℕ) : Prop :=
  n * (heart_stickers / n + star_stickers / n + smiley_stickers / n + sun_stickers / n) =
    heart_stickers + star_stickers + smiley_stickers + sun_stickers

theorem max_identical_papers : 
  ∃ (n : ℕ), n = 18 ∧ 
    ratio_heart_to_smiley n ∧ 
    ratio_star_to_sun n ∧ 
    all_stickers_used n ∧ 
    ∀ (m : ℕ), m > n → 
      ¬(ratio_heart_to_smiley m ∧ ratio_star_to_sun m ∧ all_stickers_used m) :=
by sorry

end max_identical_papers_l2311_231116


namespace inequality_solution_l2311_231162

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -4/3 ∨ x > -3/2 :=
by sorry

end inequality_solution_l2311_231162


namespace james_final_amounts_l2311_231167

def calculate_final_amounts (initial_gold : ℕ) (tax_rate : ℚ) (divorce_loss : ℚ) 
  (investment_percentage : ℚ) (stock_gain : ℕ) (exchange_rates : List ℚ) : ℕ × ℕ × ℕ :=
  sorry

theorem james_final_amounts :
  let initial_gold : ℕ := 60
  let tax_rate : ℚ := 1/10
  let divorce_loss : ℚ := 1/2
  let investment_percentage : ℚ := 1/4
  let stock_gain : ℕ := 1
  let exchange_rates : List ℚ := [5, 7, 3]
  let (silver_bars, remaining_gold, stock_investment) := 
    calculate_final_amounts initial_gold tax_rate divorce_loss investment_percentage stock_gain exchange_rates
  silver_bars = 99 ∧ remaining_gold = 3 ∧ stock_investment = 6 :=
by sorry

end james_final_amounts_l2311_231167


namespace chess_team_girls_l2311_231192

theorem chess_team_girls (total : ℕ) (attended : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  attended = 18 →
  total = boys + girls →
  attended = boys + girls / 3 →
  girls = 18 := by
sorry

end chess_team_girls_l2311_231192


namespace square_area_with_line_area_of_square_ABCD_l2311_231132

/-- A square with a line passing through it -/
structure SquareWithLine where
  /-- The side length of the square -/
  side : ℝ
  /-- The distance from vertex A to the line -/
  dist_A : ℝ
  /-- The distance from vertex C to the line -/
  dist_C : ℝ
  /-- The line passes through the midpoint of AB -/
  midpoint_AB : dist_A = side / 2
  /-- The line intersects BC -/
  intersects_BC : dist_C < side

/-- The theorem stating the area of the square given the conditions -/
theorem square_area_with_line (s : SquareWithLine) (h1 : s.dist_A = 4) (h2 : s.dist_C = 7) : 
  s.side ^ 2 = 185 := by
  sorry

/-- The main theorem proving the area of the square ABCD is 185 -/
theorem area_of_square_ABCD : ∃ s : SquareWithLine, s.side ^ 2 = 185 := by
  sorry

end square_area_with_line_area_of_square_ABCD_l2311_231132


namespace factorial_squared_ge_power_l2311_231172

theorem factorial_squared_ge_power (n : ℕ) (h : n ≥ 1) : (n!)^2 ≥ n^n := by
  sorry

end factorial_squared_ge_power_l2311_231172


namespace number_problem_l2311_231141

theorem number_problem (N : ℚ) : 
  (N / (4/5) = (4/5) * N + 27) → N = 60 := by
  sorry

end number_problem_l2311_231141


namespace range_of_a_given_inequalities_and_unique_solution_l2311_231130

theorem range_of_a_given_inequalities_and_unique_solution :
  ∀ a : ℝ,
  (∃! x : ℤ, (2 * ↑x - 7 < 0 ∧ ↑x - a > 0)) →
  (2 ≤ a ∧ a < 3) :=
by sorry

end range_of_a_given_inequalities_and_unique_solution_l2311_231130


namespace average_fruits_per_basket_l2311_231174

-- Define the number of baskets
def num_baskets : ℕ := 5

-- Define the number of fruits in each basket
def basket_A : ℕ := 15
def basket_B : ℕ := 30
def basket_C : ℕ := 20
def basket_D : ℕ := 25
def basket_E : ℕ := 35

-- Define the total number of fruits
def total_fruits : ℕ := basket_A + basket_B + basket_C + basket_D + basket_E

-- Theorem: The average number of fruits per basket is 25
theorem average_fruits_per_basket : 
  total_fruits / num_baskets = 25 := by
  sorry

end average_fruits_per_basket_l2311_231174


namespace fourth_number_is_28_l2311_231147

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def sequence_property (a b c d : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ is_two_digit d ∧
  (digit_sum a + digit_sum b + digit_sum c + digit_sum d) * 4 = a + b + c + d

theorem fourth_number_is_28 :
  ∃ (d : ℕ), sequence_property 46 19 63 d ∧ d = 28 :=
sorry

end fourth_number_is_28_l2311_231147


namespace trip_time_calculation_l2311_231111

/-- Represents the time for a trip given two different speeds -/
def trip_time (initial_speed initial_time new_speed : ℚ) : ℚ :=
  (initial_speed * initial_time) / new_speed

theorem trip_time_calculation (initial_speed initial_time new_speed : ℚ) :
  initial_speed = 80 →
  initial_time = 16/3 →
  new_speed = 50 →
  trip_time initial_speed initial_time new_speed = 128/15 := by
  sorry

#eval trip_time 80 (16/3) 50

end trip_time_calculation_l2311_231111


namespace min_value_theorem_l2311_231115

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 12*x + 128/x^4 ≥ 256 ∧ ∃ y > 0, y^2 + 12*y + 128/y^4 = 256 := by
  sorry

end min_value_theorem_l2311_231115


namespace dinner_meals_count_l2311_231156

/-- Represents the number of meals in a restaurant scenario -/
structure RestaurantMeals where
  lunch_prepared : ℕ
  lunch_sold : ℕ
  dinner_prepared : ℕ

/-- Calculates the total number of meals available for dinner -/
def meals_for_dinner (r : RestaurantMeals) : ℕ :=
  (r.lunch_prepared - r.lunch_sold) + r.dinner_prepared

/-- Theorem stating the number of meals available for dinner in the given scenario -/
theorem dinner_meals_count (r : RestaurantMeals) 
  (h1 : r.lunch_prepared = 17) 
  (h2 : r.lunch_sold = 12) 
  (h3 : r.dinner_prepared = 5) : 
  meals_for_dinner r = 10 := by
  sorry

end dinner_meals_count_l2311_231156


namespace remainder_172_pow_172_mod_13_l2311_231180

theorem remainder_172_pow_172_mod_13 : 172^172 % 13 = 3 := by
  sorry

end remainder_172_pow_172_mod_13_l2311_231180


namespace binomial_coefficient_modulo_prime_l2311_231187

theorem binomial_coefficient_modulo_prime (p : ℕ) (hp : Nat.Prime p) : 
  (Nat.choose (2 * p) p) ≡ 2 [MOD p] := by
  sorry

end binomial_coefficient_modulo_prime_l2311_231187


namespace initial_brownies_count_l2311_231150

/-- Represents the number of days in a week -/
def week : ℕ := 7

/-- Represents the number of cookies eaten per day -/
def cookiesPerDay : ℕ := 3

/-- Represents the number of brownies eaten per day -/
def browniesPerDay : ℕ := 3

/-- Represents the difference between cookies and brownies after a week -/
def cookieBrownieDifference : ℕ := 36

/-- 
Theorem: If a person eats 3 cookies and 3 brownies per day for a week, 
and ends up with 36 more cookies than brownies, 
then they must have started with 36 brownies.
-/
theorem initial_brownies_count 
  (initialCookies initialBrownies : ℕ) : 
  initialCookies - week * cookiesPerDay = initialBrownies - week * browniesPerDay + cookieBrownieDifference →
  initialBrownies = 36 := by
  sorry

end initial_brownies_count_l2311_231150


namespace variable_value_proof_l2311_231182

theorem variable_value_proof (x a k some_variable : ℝ) :
  (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + some_variable →
  a - some_variable + k = 3 →
  some_variable = -14 := by
  sorry

end variable_value_proof_l2311_231182


namespace correct_sampling_methods_l2311_231189

-- Define the community structure
structure Community where
  high_income : Nat
  middle_income : Nat
  low_income : Nat

-- Define the student group
structure StudentGroup where
  total : Nat

-- Define sampling methods
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic

-- Define the function to determine the correct sampling method for the community survey
def community_sampling_method (c : Community) (sample_size : Nat) : SamplingMethod :=
  sorry

-- Define the function to determine the correct sampling method for the student survey
def student_sampling_method (s : StudentGroup) (sample_size : Nat) : SamplingMethod :=
  sorry

-- Theorem stating the correct sampling methods for both surveys
theorem correct_sampling_methods 
  (community : Community)
  (students : StudentGroup) :
  community_sampling_method {high_income := 100, middle_income := 210, low_income := 90} 100 = SamplingMethod.Stratified ∧
  student_sampling_method {total := 10} 3 = SamplingMethod.SimpleRandom :=
sorry

end correct_sampling_methods_l2311_231189


namespace range_of_m_l2311_231149

def α (x : ℝ) : Prop := x^2 - 3*x - 10 ≤ 0

def β (m x : ℝ) : Prop := m - 3 ≤ x ∧ x ≤ m + 6

theorem range_of_m :
  (∀ x, α x → ∃ m, β m x) →
  ∀ m, (∃ x, β m x) → -1 ≤ m ∧ m ≤ 1 :=
sorry

end range_of_m_l2311_231149


namespace min_value_of_expression_equality_achieved_l2311_231129

theorem min_value_of_expression (x : ℝ) : 
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023 :=
by sorry

theorem equality_achieved : 
  ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = 2023 :=
by sorry

end min_value_of_expression_equality_achieved_l2311_231129
