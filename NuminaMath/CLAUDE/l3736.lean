import Mathlib

namespace unique_quadruple_l3736_373644

theorem unique_quadruple :
  ∃! (a b c d : ℝ), 
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    a^2 + b^2 + c^2 + d^2 = 4 ∧
    (a + b + c + d)^3 = 8 ∧
    a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 :=
by sorry

end unique_quadruple_l3736_373644


namespace puppy_sleep_ratio_l3736_373631

theorem puppy_sleep_ratio (connor_sleep : ℕ) (luke_extra_sleep : ℕ) (puppy_sleep : ℕ) : 
  connor_sleep = 6 →
  luke_extra_sleep = 2 →
  puppy_sleep = 16 →
  (puppy_sleep : ℚ) / (connor_sleep + luke_extra_sleep) = 2 := by
  sorry

end puppy_sleep_ratio_l3736_373631


namespace select_staff_eq_36_l3736_373669

/-- The number of ways to select staff for an event -/
def select_staff : ℕ :=
  let n_volunteers : ℕ := 5
  let n_translators : ℕ := 2
  let n_guides : ℕ := 2
  let n_flexible : ℕ := 1
  let n_abc : ℕ := 3  -- number of volunteers named A, B, or C

  -- Definition: Ways to choose at least one from A, B, C for translators and guides
  let ways_abc : ℕ := n_abc.choose n_translators

  -- Definition: Ways to arrange remaining volunteers
  let ways_arrange : ℕ := (n_volunteers - n_translators - n_guides).factorial

  ways_abc * ways_arrange

/-- Theorem: The number of ways to select staff is 36 -/
theorem select_staff_eq_36 : select_staff = 36 := by
  sorry

end select_staff_eq_36_l3736_373669


namespace function_inequality_l3736_373645

/-- Given a function f(x) = 2^((a-x)^2) where a is a real number,
    if f(1) > f(3) and f(2) > f(3), then |a-1| > |a-2|. -/
theorem function_inequality (a : ℝ) :
  let f : ℝ → ℝ := λ x => 2^((a-x)^2)
  (f 1 > f 3) → (f 2 > f 3) → |a-1| > |a-2| := by
  sorry

end function_inequality_l3736_373645


namespace alcohol_percentage_solution_y_l3736_373603

/-- Proves that the percentage of alcohol in solution y is 30% -/
theorem alcohol_percentage_solution_y :
  let solution_x_volume : ℝ := 200
  let solution_y_volume : ℝ := 600
  let solution_x_percentage : ℝ := 10
  let final_mixture_percentage : ℝ := 25
  let total_volume : ℝ := solution_x_volume + solution_y_volume
  let solution_y_percentage : ℝ := 
    ((final_mixture_percentage / 100) * total_volume - (solution_x_percentage / 100) * solution_x_volume) / 
    solution_y_volume * 100
  solution_y_percentage = 30 := by
sorry

end alcohol_percentage_solution_y_l3736_373603


namespace hyperbola_asymptote_l3736_373682

theorem hyperbola_asymptote (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_imaginary_axis : 2 * b = 2) (h_focal_length : 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 3) :
  ∃ k : ℝ, k = b / a ∧ k = Real.sqrt 2 / 2 := by
  sorry

end hyperbola_asymptote_l3736_373682


namespace parallel_tangents_intersection_l3736_373605

theorem parallel_tangents_intersection (x₀ : ℝ) : 
  (2 * x₀ = -3 * x₀^2) ↔ (x₀ = 0 ∨ x₀ = -2/3) :=
by sorry

end parallel_tangents_intersection_l3736_373605


namespace isosceles_triangle_side_length_l3736_373638

/-- An isosceles triangle with perimeter 16 and one side length 3 has its other side length equal to 6.5 -/
theorem isosceles_triangle_side_length : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a + b + c = 16 →
  (a = b ∧ c = 3) ∨ (a = c ∧ b = 3) ∨ (b = c ∧ a = 3) →
  (a = 6.5 ∧ b = 6.5) ∨ (a = 6.5 ∧ c = 6.5) ∨ (b = 6.5 ∧ c = 6.5) :=
by sorry

end isosceles_triangle_side_length_l3736_373638


namespace louise_teddy_bears_louise_teddy_bears_correct_l3736_373601

theorem louise_teddy_bears (initial_toys : ℕ) (initial_toy_cost : ℕ) 
  (total_money : ℕ) (teddy_bear_cost : ℕ) : ℕ :=
  let remaining_money := total_money - initial_toys * initial_toy_cost
  remaining_money / teddy_bear_cost

theorem louise_teddy_bears_correct 
  (initial_toys : ℕ) (initial_toy_cost : ℕ) 
  (total_money : ℕ) (teddy_bear_cost : ℕ) :
  louise_teddy_bears initial_toys initial_toy_cost total_money teddy_bear_cost = 20 ∧
  initial_toys = 28 ∧
  initial_toy_cost = 10 ∧
  total_money = 580 ∧
  teddy_bear_cost = 15 ∧
  total_money = initial_toys * initial_toy_cost + 
    (louise_teddy_bears initial_toys initial_toy_cost total_money teddy_bear_cost) * teddy_bear_cost :=
by
  sorry

end louise_teddy_bears_louise_teddy_bears_correct_l3736_373601


namespace correct_arrangement_satisfies_conditions_l3736_373656

-- Define the solutions
inductive Solution
| CuSO4
| CuCl2
| BaCl2
| AgNO3
| NH4OH
| HNO3
| HCl
| H2SO4

-- Define the test tubes
def TestTube := Fin 8 → Solution

-- Define the precipitation relation
def precipitates (s1 s2 : Solution) : Prop := sorry

-- Define the solubility relation
def soluble_in (s1 s2 s3 : Solution) : Prop := sorry

-- Define the correct arrangement
def correct_arrangement : TestTube :=
  fun i => match i with
  | 0 => Solution.CuSO4
  | 1 => Solution.CuCl2
  | 2 => Solution.BaCl2
  | 3 => Solution.AgNO3
  | 4 => Solution.NH4OH
  | 5 => Solution.HNO3
  | 6 => Solution.HCl
  | 7 => Solution.H2SO4

-- State the theorem
theorem correct_arrangement_satisfies_conditions (t : TestTube) :
  (t = correct_arrangement) →
  (precipitates (t 0) (t 2)) ∧
  (precipitates (t 0) (t 4)) ∧
  (precipitates (t 0) (t 3)) ∧
  (soluble_in (t 0) (t 3) (t 4)) ∧
  (soluble_in (t 0) (t 4) (t 4)) ∧
  (soluble_in (t 0) (t 4) (t 5)) ∧
  (soluble_in (t 0) (t 4) (t 6)) ∧
  (soluble_in (t 0) (t 4) (t 7)) ∧
  (precipitates (t 1) (t 3)) ∧
  (precipitates (t 1) (t 4)) ∧
  (soluble_in (t 1) (t 3) (t 4)) ∧
  (soluble_in (t 1) (t 4) (t 4)) ∧
  (soluble_in (t 1) (t 4) (t 5)) ∧
  (soluble_in (t 1) (t 4) (t 6)) ∧
  (soluble_in (t 1) (t 4) (t 7)) ∧
  (precipitates (t 2) (t 0)) ∧
  (precipitates (t 2) (t 3)) ∧
  (precipitates (t 2) (t 7)) ∧
  (soluble_in (t 2) (t 3) (t 4)) ∧
  (precipitates (t 3) (t 1)) ∧
  (precipitates (t 3) (t 4)) ∧
  (precipitates (t 3) (t 6)) ∧
  (precipitates (t 3) (t 0)) ∧
  (precipitates (t 3) (t 7)) ∧
  (∀ i, soluble_in (t 3) (t i) (t 4)) :=
by sorry


end correct_arrangement_satisfies_conditions_l3736_373656


namespace library_visitors_average_l3736_373643

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (days_in_month : ℕ) (h1 : sunday_visitors = 570) (h2 : other_day_visitors = 240) 
  (h3 : days_in_month = 30) :
  let sundays := (days_in_month + 6) / 7
  let other_days := days_in_month - sundays
  let total_visitors := sundays * sunday_visitors + other_days * other_day_visitors
  total_visitors / days_in_month = 295 := by
sorry

end library_visitors_average_l3736_373643


namespace count_odd_numbers_300_to_600_l3736_373639

theorem count_odd_numbers_300_to_600 : 
  (Finset.filter (fun n => n % 2 = 1 ∧ n > 300 ∧ n < 600) (Finset.range 600)).card = 149 := by
  sorry

end count_odd_numbers_300_to_600_l3736_373639


namespace two_colonies_growth_time_l3736_373614

/-- Represents the number of days it takes for a bacteria colony to reach its habitat limit -/
def habitatLimitDays : ℕ := 25

/-- Represents the daily growth factor of a bacteria colony -/
def dailyGrowthFactor : ℕ := 2

/-- Theorem stating that two simultaneously growing bacteria colonies 
    will reach the habitat limit in the same number of days as a single colony -/
theorem two_colonies_growth_time (initialSize : ℕ) (habitatLimit : ℕ) :
  initialSize > 0 →
  habitatLimit > 0 →
  habitatLimit = initialSize * dailyGrowthFactor ^ habitatLimitDays →
  (2 * initialSize) * dailyGrowthFactor ^ habitatLimitDays = 2 * habitatLimit :=
by
  sorry

end two_colonies_growth_time_l3736_373614


namespace y_intercept_of_line_l3736_373627

/-- The y-intercept of the line 3x - 4y = 12 is -3 -/
theorem y_intercept_of_line (x y : ℝ) : 3 * x - 4 * y = 12 → x = 0 → y = -3 := by
  sorry

end y_intercept_of_line_l3736_373627


namespace system_solution_l3736_373618

theorem system_solution :
  let f (x y : ℝ) := x^2 - 5*x*y + 6*y^2
  let g (x y : ℝ) := x^2 + y^2
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (f x₁ y₁ = 0 ∧ g x₁ y₁ = 40) ∧
    (f x₂ y₂ = 0 ∧ g x₂ y₂ = 40) ∧
    (f x₃ y₃ = 0 ∧ g x₃ y₃ = 40) ∧
    (f x₄ y₄ = 0 ∧ g x₄ y₄ = 40) ∧
    x₁ = 4 * Real.sqrt 2 ∧ y₁ = 2 * Real.sqrt 2 ∧
    x₂ = -4 * Real.sqrt 2 ∧ y₂ = -2 * Real.sqrt 2 ∧
    x₃ = 6 ∧ y₃ = 2 ∧
    x₄ = -6 ∧ y₄ = -2 :=
by
  sorry

end system_solution_l3736_373618


namespace arithmetic_sequence_common_difference_l3736_373681

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    prove that the common difference is 4 if 2S_3 - 3S_2 = 12 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h1 : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1)))  -- Definition of S_n
  (h2 : 2 * S 3 - 3 * S 2 = 12)  -- Given condition
  : a 2 - a 1 = 4 :=
by sorry

end arithmetic_sequence_common_difference_l3736_373681


namespace set_intersection_theorem_l3736_373629

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem set_intersection_theorem : M ∩ N = {0, 1} := by sorry

end set_intersection_theorem_l3736_373629


namespace midpoint_coordinate_sum_l3736_373663

theorem midpoint_coordinate_sum : 
  let p₁ : ℝ × ℝ := (8, 16)
  let p₂ : ℝ × ℝ := (-2, -8)
  let midpoint := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  (midpoint.1 + midpoint.2 : ℝ) = 7 := by
  sorry

end midpoint_coordinate_sum_l3736_373663


namespace square_intersection_perimeter_ratio_l3736_373675

/-- Given a square with vertices (-a, 0), (a, 0), (a, 2a), (-a, 2a) intersected by the line y = 2x,
    the ratio of the perimeter of one of the resulting congruent quadrilaterals to a is 5 + √5. -/
theorem square_intersection_perimeter_ratio (a : ℝ) (a_pos : a > 0) :
  let square_vertices := [(-a, 0), (a, 0), (a, 2*a), (-a, 2*a)]
  let intersecting_line := (fun x : ℝ => 2*x)
  let quadrilateral_perimeter := 
    (a + 2*a + 2*a + Real.sqrt (a^2 + (2*a)^2))
  quadrilateral_perimeter / a = 5 + Real.sqrt 5 := by
  sorry

end square_intersection_perimeter_ratio_l3736_373675


namespace xy_values_l3736_373652

theorem xy_values (x y : ℝ) 
  (eq1 : x / (x^2 * y^2 - 1) - 1 / x = 4)
  (eq2 : (x^2 * y) / (x^2 * y^2 - 1) + y = 2) :
  x * y = 1 / Real.sqrt 2 ∨ x * y = -(1 / Real.sqrt 2) := by
sorry

end xy_values_l3736_373652


namespace scrap_rate_cost_increase_l3736_373637

/-- The regression equation for cost of cast iron based on scrap rate -/
def cost_equation (x : ℝ) : ℝ := 56 + 8 * x

/-- Theorem stating the relationship between scrap rate increase and cost increase -/
theorem scrap_rate_cost_increase (x : ℝ) :
  cost_equation (x + 1) - cost_equation x = 8 := by
  sorry

end scrap_rate_cost_increase_l3736_373637


namespace subset_condition_disjoint_condition_l3736_373696

-- Define set A
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3*a) < 0}

-- Theorem for the first part of the problem
theorem subset_condition (a : ℝ) : 
  A ⊆ (A ∩ B a) ↔ 4/3 ≤ a ∧ a ≤ 2 :=
sorry

-- Theorem for the second part of the problem
theorem disjoint_condition (a : ℝ) :
  A ∩ B a = ∅ ↔ a ≤ 2/3 ∨ a ≥ 4 :=
sorry

end subset_condition_disjoint_condition_l3736_373696


namespace woman_lawyer_probability_l3736_373660

/-- Represents a study group with given proportions of women and lawyers --/
structure StudyGroup where
  total_members : ℕ
  women_percentage : ℝ
  lawyer_percentage : ℝ
  women_percentage_valid : 0 ≤ women_percentage ∧ women_percentage ≤ 1
  lawyer_percentage_valid : 0 ≤ lawyer_percentage ∧ lawyer_percentage ≤ 1

/-- Calculates the probability of selecting a woman lawyer from the study group --/
def probability_woman_lawyer (group : StudyGroup) : ℝ :=
  group.women_percentage * group.lawyer_percentage

/-- Theorem stating that the probability of selecting a woman lawyer is 0.32 
    given the specified conditions --/
theorem woman_lawyer_probability (group : StudyGroup) 
  (h1 : group.women_percentage = 0.8) 
  (h2 : group.lawyer_percentage = 0.4) : 
  probability_woman_lawyer group = 0.32 := by
  sorry


end woman_lawyer_probability_l3736_373660


namespace lorry_speed_l3736_373686

/-- Calculates the speed of a lorry crossing a bridge -/
theorem lorry_speed (lorry_length bridge_length : ℝ) (crossing_time : ℝ) :
  lorry_length = 200 →
  bridge_length = 200 →
  crossing_time = 17.998560115190784 →
  ∃ (speed : ℝ), abs (speed - 80) < 0.01 ∧ 
  speed = (lorry_length + bridge_length) / crossing_time * 3.6 := by
  sorry

#check lorry_speed

end lorry_speed_l3736_373686


namespace square_b_minus_d_l3736_373678

theorem square_b_minus_d (a b c d : ℤ) 
  (eq1 : a - b - c + d = 13) 
  (eq2 : a + b - c - d = 3) : 
  (b - d)^2 = 25 := by sorry

end square_b_minus_d_l3736_373678


namespace blue_tetrahedron_volume_l3736_373695

/-- Represents a cube with alternating colored corners -/
structure ColoredCube where
  sideLength : ℝ
  alternatingColors : Bool

/-- Calculates the volume of the tetrahedron formed by similarly colored vertices -/
def tetrahedronVolume (cube : ColoredCube) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem blue_tetrahedron_volume (cube : ColoredCube) 
  (h1 : cube.sideLength = 8) 
  (h2 : cube.alternatingColors = true) : 
  tetrahedronVolume cube = 512 / 3 := by
    sorry

end blue_tetrahedron_volume_l3736_373695


namespace pumpkin_weight_difference_l3736_373659

/-- Given three pumpkin weights with specific relationships, 
    prove the difference between the heaviest and lightest is 81 pounds. -/
theorem pumpkin_weight_difference :
  ∀ (brad jessica betty : ℝ),
  brad = 54 →
  jessica = brad / 2 →
  betty = 4 * jessica →
  max brad (max jessica betty) - min brad (min jessica betty) = 81 :=
by
  sorry

end pumpkin_weight_difference_l3736_373659


namespace final_plant_count_l3736_373684

def total_seeds : ℕ := 23
def marigold_seeds : ℕ := 10
def sunflower_seeds : ℕ := 8
def lavender_seeds : ℕ := 5
def seeds_not_grown : ℕ := 5

def marigold_growth_rate : ℚ := 2/5
def sunflower_growth_rate : ℚ := 3/5
def lavender_growth_rate : ℚ := 7/10

def squirrel_eat_rate : ℚ := 1/2
def rabbit_eat_rate : ℚ := 1/4

def pest_control_success_rate : ℚ := 3/4
def pest_control_reduction_rate : ℚ := 1/10

def weed_strangle_rate : ℚ := 1/3

def weeds_pulled : ℕ := 2
def weeds_kept : ℕ := 1

theorem final_plant_count :
  ∃ (grown_marigolds grown_sunflowers grown_lavenders : ℕ),
    grown_marigolds ≤ (marigold_seeds : ℚ) * marigold_growth_rate ∧
    grown_sunflowers ≤ (sunflower_seeds : ℚ) * sunflower_growth_rate ∧
    grown_lavenders ≤ (lavender_seeds : ℚ) * lavender_growth_rate ∧
    ∃ (eaten_marigolds eaten_sunflowers : ℕ),
      eaten_marigolds = ⌊(grown_marigolds : ℚ) * squirrel_eat_rate⌋ ∧
      eaten_sunflowers = ⌊(grown_sunflowers : ℚ) * rabbit_eat_rate⌋ ∧
      ∃ (protected_lavenders : ℕ),
        protected_lavenders ≤ ⌊(grown_lavenders : ℚ) * pest_control_success_rate⌋ ∧
        ∃ (final_marigolds final_sunflowers : ℕ),
          final_marigolds ≤ ⌊(grown_marigolds - eaten_marigolds : ℚ) * (1 - pest_control_reduction_rate)⌋ ∧
          final_sunflowers ≤ ⌊(grown_sunflowers - eaten_sunflowers : ℚ) * (1 - pest_control_reduction_rate)⌋ ∧
          ∃ (total_plants : ℕ),
            total_plants = final_marigolds + final_sunflowers + protected_lavenders ∧
            ∃ (strangled_plants : ℕ),
              strangled_plants = ⌊(total_plants : ℚ) * weed_strangle_rate⌋ ∧
              total_plants - strangled_plants + weeds_kept = 6 :=
by sorry

end final_plant_count_l3736_373684


namespace biased_coin_prob_sum_l3736_373661

/-- The probability of getting heads for a biased coin -/
def h : ℚ :=
  3 / 7

/-- The condition that the probability of 2 heads equals the probability of 3 heads in 6 flips -/
axiom prob_equality : 15 * h^2 * (1 - h)^4 = 20 * h^3 * (1 - h)^3

/-- The probability of getting exactly 4 heads in 6 flips -/
def prob_4_heads : ℚ :=
  15 * h^4 * (1 - h)^2

/-- The numerator and denominator of prob_4_heads in lowest terms -/
def p : ℕ := 19440
def q : ℕ := 117649

theorem biased_coin_prob_sum :
  prob_4_heads = p / q ∧ p + q = 137089 := by sorry

end biased_coin_prob_sum_l3736_373661


namespace dara_half_jane_age_l3736_373664

/-- The problem statement about Dara and Jane's ages -/
theorem dara_half_jane_age :
  let min_age : ℕ := 25  -- Minimum age for employment
  let jane_age : ℕ := 28  -- Jane's current age
  let years_to_min : ℕ := 14  -- Years until Dara reaches minimum age
  let dara_age : ℕ := min_age - years_to_min  -- Dara's current age
  let x : ℕ := 6  -- Years until Dara is half Jane's age
  dara_age + x = (jane_age + x) / 2 := by sorry

end dara_half_jane_age_l3736_373664


namespace subcommittee_count_l3736_373699

/-- The number of people in the main committee -/
def committee_size : ℕ := 7

/-- The size of each sub-committee -/
def subcommittee_size : ℕ := 2

/-- The number of people that can be chosen for the second position in the sub-committee -/
def remaining_choices : ℕ := committee_size - 1

theorem subcommittee_count :
  (committee_size.choose subcommittee_size) / committee_size = remaining_choices :=
sorry

end subcommittee_count_l3736_373699


namespace min_sum_of_prime_factors_l3736_373606

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The sum of n consecutive integers starting from x -/
def consecutiveSum (x n : ℕ) : ℕ :=
  n * (2 * x + n - 1) / 2

theorem min_sum_of_prime_factors (a b c d : ℕ) :
  isPrime a → isPrime b → isPrime c → isPrime d →
  (∃ x : ℕ, a * b * c * d = consecutiveSum x 35) →
  22 ≤ a + b + c + d :=
sorry

end min_sum_of_prime_factors_l3736_373606


namespace line_perp_plane_transitive_l3736_373692

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_transitive 
  (m n : Line) (α : Plane) :
  para m n → perp n α → perp m α :=
sorry

end line_perp_plane_transitive_l3736_373692


namespace sam_puppies_count_l3736_373657

/-- The number of puppies Sam originally had with spots -/
def original_puppies : ℕ := 8

/-- The number of puppies Sam gave to his friends -/
def given_away_puppies : ℕ := 2

/-- The number of puppies Sam has now -/
def remaining_puppies : ℕ := original_puppies - given_away_puppies

theorem sam_puppies_count : remaining_puppies = 6 := by
  sorry

end sam_puppies_count_l3736_373657


namespace work_completion_proof_l3736_373680

/-- The number of days it takes for person B to complete the work alone -/
def person_b_days : ℝ := 45

/-- The fraction of work completed by both persons in 5 days -/
def work_completed_together : ℝ := 0.2777777777777778

/-- The number of days they work together -/
def days_worked_together : ℝ := 5

/-- The number of days it takes for person A to complete the work alone -/
def person_a_days : ℝ := 30

theorem work_completion_proof :
  (days_worked_together * (1 / person_a_days + 1 / person_b_days) = work_completed_together) →
  person_a_days = 30 := by
  sorry

end work_completion_proof_l3736_373680


namespace hatcher_students_l3736_373635

/-- Calculates the total number of students Ms. Hatcher taught -/
def total_students (third_graders : ℕ) : ℕ :=
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  third_graders + fourth_graders + fifth_graders

/-- Theorem stating that Ms. Hatcher taught 70 students -/
theorem hatcher_students : total_students 20 = 70 := by
  sorry

end hatcher_students_l3736_373635


namespace farm_animals_l3736_373674

theorem farm_animals (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 20 + 3 * (cows + chickens)) → 
  (cows = 20 + chickens) :=
by sorry

end farm_animals_l3736_373674


namespace greatest_sum_consecutive_integers_product_less_400_l3736_373655

theorem greatest_sum_consecutive_integers_product_less_400 :
  (∀ n : ℤ, n * (n + 1) < 400 → n + (n + 1) ≤ 39) ∧
  (∃ n : ℤ, n * (n + 1) < 400 ∧ n + (n + 1) = 39) :=
by sorry

end greatest_sum_consecutive_integers_product_less_400_l3736_373655


namespace prism_volume_l3736_373653

/-- 
Given a right rectangular prism with face areas 10, 15, and 6 square inches,
prove that its volume is 30 cubic inches.
-/
theorem prism_volume (l w h : ℝ) 
  (area1 : l * w = 10)
  (area2 : w * h = 15)
  (area3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end prism_volume_l3736_373653


namespace divisibility_of_sum_of_powers_l3736_373697

theorem divisibility_of_sum_of_powers : 
  let n : ℕ := 3^105 + 4^105
  ∃ (a b c d : ℕ), n = 13 * a ∧ n = 49 * b ∧ n = 181 * c ∧ n = 379 * d ∧
  ¬(∃ (e : ℕ), n = 5 * e) ∧ ¬(∃ (f : ℕ), n = 11 * f) := by
  sorry

end divisibility_of_sum_of_powers_l3736_373697


namespace sum_of_roots_l3736_373610

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 9*a^2 + 26*a - 40 = 0)
  (hb : 2*b^3 - 18*b^2 + 22*b - 30 = 0) : 
  a + b = Real.rpow 45 (1/3) + Real.rpow 22.5 (1/3) + 6 := by
sorry

end sum_of_roots_l3736_373610


namespace min_additions_for_54_l3736_373600

/-- A type representing a way to split the number 123456789 into parts -/
def Splitting := List Nat

/-- The original number we're working with -/
def originalNumber : Nat := 123456789

/-- Function to check if a splitting is valid (uses all digits in order) -/
def isValidSplitting (s : Splitting) : Prop :=
  s.foldl (· * 10 + ·) 0 = originalNumber

/-- Function to calculate the sum of a splitting -/
def sumOfSplitting (s : Splitting) : Nat :=
  s.sum

/-- The target sum we want to achieve -/
def targetSum : Nat := 54

/-- Theorem stating that the minimum number of addition signs needed is 7 -/
theorem min_additions_for_54 :
  (∃ (s : Splitting), isValidSplitting s ∧ sumOfSplitting s = targetSum) ∧
  (∀ (s : Splitting), isValidSplitting s ∧ sumOfSplitting s = targetSum → s.length ≥ 8) ∧
  (∃ (s : Splitting), isValidSplitting s ∧ sumOfSplitting s = targetSum ∧ s.length = 8) :=
sorry

end min_additions_for_54_l3736_373600


namespace point_in_fourth_quadrant_l3736_373698

/-- A point is in the fourth quadrant if its x-coordinate is positive and its y-coordinate is negative -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The point P has coordinates (2, -3) -/
def P : ℝ × ℝ := (2, -3)

theorem point_in_fourth_quadrant :
  fourth_quadrant P.1 P.2 := by
  sorry

end point_in_fourth_quadrant_l3736_373698


namespace siblings_weekly_water_consumption_l3736_373677

/-- The number of days in a week -/
def daysInWeek : ℕ := 7

/-- The daily water consumption of the first sibling -/
def sibling1DailyConsumption : ℕ := 8

/-- The daily water consumption of the second sibling -/
def sibling2DailyConsumption : ℕ := 7

/-- The daily water consumption of the third sibling -/
def sibling3DailyConsumption : ℕ := 9

/-- The total weekly water consumption of all siblings -/
def totalWeeklyConsumption : ℕ :=
  (sibling1DailyConsumption + sibling2DailyConsumption + sibling3DailyConsumption) * daysInWeek

theorem siblings_weekly_water_consumption :
  totalWeeklyConsumption = 168 := by
  sorry

end siblings_weekly_water_consumption_l3736_373677


namespace least_prime_factor_11_5_minus_11_4_l3736_373689

theorem least_prime_factor_11_5_minus_11_4 :
  Nat.minFac (11^5 - 11^4) = 2 := by
  sorry

end least_prime_factor_11_5_minus_11_4_l3736_373689


namespace rectangle_length_proof_l3736_373630

theorem rectangle_length_proof (square_perimeter : ℝ) (rectangle_width : ℝ) :
  square_perimeter = 256 →
  rectangle_width = 32 →
  (square_perimeter / 4) ^ 2 = 2 * (rectangle_width * (square_perimeter / 4)) →
  square_perimeter / 4 = 64 :=
by sorry

end rectangle_length_proof_l3736_373630


namespace fraction_product_theorem_l3736_373667

theorem fraction_product_theorem :
  (7 / 4 : ℚ) * (8 / 14 : ℚ) * (9 / 6 : ℚ) * (10 / 25 : ℚ) * 
  (28 / 21 : ℚ) * (15 / 45 : ℚ) * (32 / 16 : ℚ) * (50 / 100 : ℚ) = 4 / 5 := by
  sorry

end fraction_product_theorem_l3736_373667


namespace min_value_expression_l3736_373625

theorem min_value_expression (a x y z : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hx : -a < x ∧ x < a) 
  (hy : -a < y ∧ y < a) 
  (hz : -a < z ∧ z < a) : 
  (∀ x y z, 1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) ≥ 2 / (1 - a^2)^3) ∧ 
  (∃ x y z, 1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) = 2 / (1 - a^2)^3) :=
by sorry

end min_value_expression_l3736_373625


namespace min_square_area_for_rectangles_l3736_373636

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum square side length needed to fit two rectangles -/
def minSquareSide (r1 r2 : Rectangle) : ℕ :=
  max (max r1.width r2.height) (r1.height + r2.width)

/-- Theorem: The smallest square area to fit a 3x4 and a 4x5 rectangle with one rotated is 81 -/
theorem min_square_area_for_rectangles :
  let r1 : Rectangle := ⟨3, 4⟩
  let r2 : Rectangle := ⟨4, 5⟩
  (minSquareSide r1 r2) ^ 2 = 81 := by
  sorry

#eval (minSquareSide ⟨3, 4⟩ ⟨4, 5⟩) ^ 2

end min_square_area_for_rectangles_l3736_373636


namespace elise_remaining_money_l3736_373621

/-- Calculates the remaining money for Elise given her initial amount, savings, and expenditures. -/
def remaining_money (initial : ℕ) (savings : ℕ) (comic_cost : ℕ) (puzzle_cost : ℕ) : ℕ :=
  initial + savings - comic_cost - puzzle_cost

/-- Proves that Elise is left with $1 given her initial amount, savings, and expenditures. -/
theorem elise_remaining_money :
  remaining_money 8 13 2 18 = 1 := by
  sorry

end elise_remaining_money_l3736_373621


namespace triangle_problem_l3736_373683

open Real

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : cos t.B = 4/5) : 
  (t.a = 5/3 → t.A = π/6) ∧ 
  (t.a + t.c = 2 * Real.sqrt 10 → 
    1/2 * t.a * t.c * sin t.B = 3) := by
  sorry

end triangle_problem_l3736_373683


namespace bank_queue_properties_l3736_373670

/-- Represents a bank queue with simple and long operations -/
structure BankQueue where
  total_people : Nat
  simple_ops : Nat
  long_ops : Nat
  simple_time : Nat
  long_time : Nat

/-- Calculates the minimum possible total wasted person-minutes -/
def min_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the maximum possible total wasted person-minutes -/
def max_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the expected value of wasted person-minutes assuming random order -/
def expected_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Theorem stating the properties of the bank queue problem -/
theorem bank_queue_properties (q : BankQueue) 
  (h1 : q.total_people = 8)
  (h2 : q.simple_ops = 5)
  (h3 : q.long_ops = 3)
  (h4 : q.simple_time = 1)
  (h5 : q.long_time = 5) :
  min_wasted_time q = 40 ∧ 
  max_wasted_time q = 100 ∧ 
  expected_wasted_time q = 84 :=
by sorry

end bank_queue_properties_l3736_373670


namespace product_divisible_by_twelve_l3736_373646

theorem product_divisible_by_twelve (a b c d : ℤ) : 
  ∃ k : ℤ, (b - a) * (c - a) * (d - a) * (b - c) * (d - c) * (d - b) = 12 * k := by
  sorry

end product_divisible_by_twelve_l3736_373646


namespace solve_for_y_l3736_373651

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := by
  sorry

end solve_for_y_l3736_373651


namespace probability_arrives_before_l3736_373688

/-- Represents a student -/
structure Student :=
  (name : String)

/-- Represents the arrival order of students -/
def ArrivalOrder := List Student

/-- Given a list of students, generates all possible arrival orders -/
def allPossibleArrivals (students : List Student) : List ArrivalOrder :=
  sorry

/-- Checks if student1 arrives before student2 in a given arrival order -/
def arrivesBeforeIn (student1 student2 : Student) (order : ArrivalOrder) : Bool :=
  sorry

/-- Counts the number of arrival orders where student1 arrives before student2 -/
def countArrivesBeforeOrders (student1 student2 : Student) (orders : List ArrivalOrder) : Nat :=
  sorry

theorem probability_arrives_before (student1 student2 student3 : Student) :
  let students := [student1, student2, student3]
  let allOrders := allPossibleArrivals students
  let favorableOrders := countArrivesBeforeOrders student1 student2 allOrders
  favorableOrders = allOrders.length / 2 := by
  sorry

end probability_arrives_before_l3736_373688


namespace regular_polygon_interior_angle_l3736_373623

theorem regular_polygon_interior_angle (n : ℕ) : 
  (n ≥ 3) → (((n - 2) * 180 : ℝ) / n = 144) → n = 10 := by
  sorry

end regular_polygon_interior_angle_l3736_373623


namespace P_Q_disjoint_l3736_373676

def P : Set ℕ := {n | ∃ k, n = 2 * k^2 - 2 * k + 1}

def Q : Set ℕ := {n | n > 0 ∧ (Complex.I + 1)^(2*n) = 2^n * Complex.I}

theorem P_Q_disjoint : P ∩ Q = ∅ := by sorry

end P_Q_disjoint_l3736_373676


namespace simple_interest_rate_calculation_l3736_373671

/-- Simple interest rate calculation -/
theorem simple_interest_rate_calculation
  (principal : ℝ)
  (time : ℝ)
  (final_amount : ℝ)
  (h1 : principal = 1000)
  (h2 : time = 3)
  (h3 : final_amount = 1300) :
  (final_amount - principal) / (principal * time) = 0.1 := by
  sorry

end simple_interest_rate_calculation_l3736_373671


namespace jorkins_christmas_spending_l3736_373693

-- Define the type for British currency
structure BritishCurrency where
  pounds : ℕ
  shillings : ℕ

def BritishCurrency.toShillings (bc : BritishCurrency) : ℕ :=
  20 * bc.pounds + bc.shillings

def BritishCurrency.halfValue (bc : BritishCurrency) : ℕ :=
  bc.toShillings / 2

theorem jorkins_christmas_spending (initial : BritishCurrency) 
  (h1 : initial.halfValue = 20 * (initial.shillings / 2) + initial.pounds)
  (h2 : initial.shillings / 2 = initial.pounds)
  (h3 : initial.pounds = initial.shillings / 2) :
  initial = BritishCurrency.mk 19 18 := by
  sorry

#check jorkins_christmas_spending

end jorkins_christmas_spending_l3736_373693


namespace triangle_area_triangle_area_proof_l3736_373672

/-- Given a triangle with side lengths 7, 24, and 25 units, its area is 84 square units. -/
theorem triangle_area : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c area =>
    a = 7 ∧ b = 24 ∧ c = 25 → area = 84

/-- The statement of the theorem -/
theorem triangle_area_proof : ∃ (area : ℝ), triangle_area 7 24 25 area :=
  sorry

end triangle_area_triangle_area_proof_l3736_373672


namespace polynomial_division_degree_l3736_373611

theorem polynomial_division_degree (f q d r : Polynomial ℝ) :
  Polynomial.degree f = 17 →
  Polynomial.degree q = 10 →
  Polynomial.degree r = 5 →
  f = d * q + r →
  Polynomial.degree d = 7 := by
sorry

end polynomial_division_degree_l3736_373611


namespace smallest_number_l3736_373624

theorem smallest_number : 
  let a := (2010 : ℝ) ^ (1 / 209)
  let b := (2009 : ℝ) ^ (1 / 200)
  let c := (2010 : ℝ)
  let d := (2010 : ℝ) / 2009
  let e := (2009 : ℝ) / 2010
  (e ≤ a) ∧ (e ≤ b) ∧ (e ≤ c) ∧ (e ≤ d) ∧ (e ≤ e) :=
by sorry

end smallest_number_l3736_373624


namespace max_x_value_l3736_373691

theorem max_x_value (x : ℝ) : 
  ((6*x - 15)/(4*x - 5))^2 - 3*((6*x - 15)/(4*x - 5)) - 10 = 0 → x ≤ 25/14 :=
by
  sorry

end max_x_value_l3736_373691


namespace hypotenuse_product_equals_area_l3736_373662

/-- A right-angled triangle with an incircle -/
structure RightTriangleWithIncircle where
  /-- The area of the triangle -/
  area : ℝ
  /-- The radius of the incircle -/
  incircle_radius : ℝ
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The first part of the hypotenuse divided by the incircle's point of contact -/
  x : ℝ
  /-- The second part of the hypotenuse divided by the incircle's point of contact -/
  y : ℝ
  /-- The sum of x and y is equal to the hypotenuse -/
  hypotenuse_division : x + y = hypotenuse
  /-- All lengths are positive -/
  all_positive : 0 < area ∧ 0 < incircle_radius ∧ 0 < hypotenuse ∧ 0 < x ∧ 0 < y

/-- The theorem stating that the product of the two parts of the hypotenuse 
    is equal to the area of the right-angled triangle with an incircle -/
theorem hypotenuse_product_equals_area (t : RightTriangleWithIncircle) : t.x * t.y = t.area := by
  sorry

end hypotenuse_product_equals_area_l3736_373662


namespace max_blocks_fit_l3736_373613

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The dimensions of the large box -/
def largeBox : BoxDimensions := ⟨3, 3, 2⟩

/-- The dimensions of the small block -/
def smallBlock : BoxDimensions := ⟨1, 2, 2⟩

/-- Calculates the volume of a box given its dimensions -/
def volume (box : BoxDimensions) : ℝ :=
  box.length * box.width * box.height

/-- Represents the number of small blocks that can fit in the large box -/
def maxBlocks : ℕ := 3

/-- Theorem stating that the maximum number of small blocks that can fit in the large box is 3 -/
theorem max_blocks_fit :
  maxBlocks = 3 ∧
  maxBlocks * volume smallBlock ≤ volume largeBox ∧
  ∀ n : ℕ, n > maxBlocks → n * volume smallBlock > volume largeBox :=
by sorry

end max_blocks_fit_l3736_373613


namespace square_triangle_perimeter_equality_l3736_373690

theorem square_triangle_perimeter_equality (x : ℝ) :
  x = 4 →
  4 * (x + 2) = 3 * (2 * x) := by
  sorry

end square_triangle_perimeter_equality_l3736_373690


namespace equation_proof_l3736_373608

theorem equation_proof : 
  (Real.sqrt (7^2 + 24^2)) / (Real.sqrt (49 + 16)) = (25 * Real.sqrt 65) / 65 := by
  sorry

end equation_proof_l3736_373608


namespace units_digit_of_23_times_51_squared_l3736_373642

theorem units_digit_of_23_times_51_squared : ∃ n : ℕ, 23 * 51^2 = 10 * n + 3 := by
  sorry

end units_digit_of_23_times_51_squared_l3736_373642


namespace root_sum_ratio_l3736_373615

theorem root_sum_ratio (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 4 * x₁ + 1 = 0) → 
  (2 * x₂^2 - 4 * x₂ + 1 = 0) → 
  (x₁ ≠ x₂) →
  (x₁ / x₂ + x₂ / x₁ = 6) := by
sorry

end root_sum_ratio_l3736_373615


namespace rational_abs_eq_neg_l3736_373648

theorem rational_abs_eq_neg (a : ℚ) (h : |a| = -a) : a ≤ 0 := by
  sorry

end rational_abs_eq_neg_l3736_373648


namespace stacy_height_last_year_l3736_373617

/-- Represents Stacy's height measurements and growth --/
structure StacyHeight where
  current : ℕ
  brother_growth : ℕ
  growth_difference : ℕ

/-- Calculates Stacy's height last year given her current measurements --/
def height_last_year (s : StacyHeight) : ℕ :=
  s.current - (s.brother_growth + s.growth_difference)

/-- Theorem stating Stacy's height last year was 50 inches --/
theorem stacy_height_last_year :
  let s : StacyHeight := {
    current := 57,
    brother_growth := 1,
    growth_difference := 6
  }
  height_last_year s = 50 := by
  sorry

end stacy_height_last_year_l3736_373617


namespace optimal_height_minimizes_surface_area_l3736_373607

/-- Represents a rectangular box with a lid -/
structure Box where
  x : ℝ  -- Length of one side of the base
  y : ℝ  -- Height of the box

/-- Calculates the volume of the box -/
def volume (b : Box) : ℝ := 2 * b.x^2 * b.y

/-- Calculates the surface area of the box -/
def surfaceArea (b : Box) : ℝ := 4 * b.x^2 + 6 * b.x * b.y

/-- States that the volume of the box is 72 -/
def volumeConstraint (b : Box) : Prop := volume b = 72

/-- Finds the height that minimizes the surface area -/
def optimalHeight : ℝ := 4

theorem optimal_height_minimizes_surface_area :
  ∃ (b : Box), volumeConstraint b ∧
    ∀ (b' : Box), volumeConstraint b' → surfaceArea b ≤ surfaceArea b' ∧
    b.y = optimalHeight := by sorry

end optimal_height_minimizes_surface_area_l3736_373607


namespace luke_trips_l3736_373650

/-- The number of trays Luke can carry in one trip -/
def trays_per_trip : ℕ := 4

/-- The number of trays on the first table -/
def trays_table1 : ℕ := 20

/-- The number of trays on the second table -/
def trays_table2 : ℕ := 16

/-- The total number of trips Luke will make -/
def total_trips : ℕ := (trays_table1 / trays_per_trip) + (trays_table2 / trays_per_trip)

theorem luke_trips : total_trips = 9 := by
  sorry

end luke_trips_l3736_373650


namespace exponent_sum_zero_polynomial_simplification_l3736_373641

-- Problem 1
theorem exponent_sum_zero (m : ℝ) : m^3 * m^6 + (-m^3)^3 = 0 := by sorry

-- Problem 2
theorem polynomial_simplification (a : ℝ) : a*(a-2) - 2*a*(1-3*a) = 7*a^2 - 4*a := by sorry

end exponent_sum_zero_polynomial_simplification_l3736_373641


namespace potato_peeling_time_l3736_373649

theorem potato_peeling_time (julie_rate ted_rate initial_time : ℝ) 
  (h1 : julie_rate = 1 / 10)  -- Julie's peeling rate per hour
  (h2 : ted_rate = 1 / 8)     -- Ted's peeling rate per hour
  (h3 : initial_time = 4)     -- Time they work together
  : (1 - (julie_rate + ted_rate) * initial_time) / julie_rate = 1 := by
  sorry

end potato_peeling_time_l3736_373649


namespace december_november_difference_l3736_373622

def october_visitors : ℕ := 100

def november_visitors : ℕ := (october_visitors * 115) / 100

def total_visitors : ℕ := 345

theorem december_november_difference :
  ∃ (december_visitors : ℕ),
    december_visitors > november_visitors ∧
    october_visitors + november_visitors + december_visitors = total_visitors ∧
    december_visitors - november_visitors = 15 :=
by
  sorry

end december_november_difference_l3736_373622


namespace perfect_square_3_6_4_5_5_4_l3736_373666

theorem perfect_square_3_6_4_5_5_4 : ∃ n : ℕ, n ^ 2 = 3^6 * 4^5 * 5^4 := by
  sorry

end perfect_square_3_6_4_5_5_4_l3736_373666


namespace football_team_throwers_l3736_373634

/-- Represents the number of throwers on a football team given specific conditions -/
def number_of_throwers (total_players : ℕ) (right_handed : ℕ) : ℕ :=
  total_players - (3 * (total_players - (right_handed - (total_players - right_handed))) / 2)

/-- Theorem stating that under given conditions, there are 28 throwers on the team -/
theorem football_team_throwers :
  let total_players : ℕ := 70
  let right_handed : ℕ := 56
  number_of_throwers total_players right_handed = 28 := by
  sorry

end football_team_throwers_l3736_373634


namespace carpet_width_l3736_373628

/-- Proves that a rectangular carpet covering 75% of a 48 sq ft room with a length of 9 ft has a width of 4 ft -/
theorem carpet_width (room_area : ℝ) (carpet_length : ℝ) (coverage_percent : ℝ) :
  room_area = 48 →
  carpet_length = 9 →
  coverage_percent = 0.75 →
  (room_area * coverage_percent) / carpet_length = 4 := by
  sorry

end carpet_width_l3736_373628


namespace absolute_value_greater_than_x_l3736_373640

theorem absolute_value_greater_than_x (x : ℝ) : (x < 0) ↔ (abs x > x) := by sorry

end absolute_value_greater_than_x_l3736_373640


namespace geometric_series_common_ratio_l3736_373685

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 12/7
  let r : ℚ := a₂ / a₁
  r = 3 := by sorry

end geometric_series_common_ratio_l3736_373685


namespace polynomial_factor_l3736_373673

theorem polynomial_factor (x y z : ℝ) :
  ∃ (q : ℝ → ℝ → ℝ → ℝ), 
    x^2 - y^2 - z^2 - 2*y*z + x - y - z + 2 = (x - y - z + 1) * q x y z := by
  sorry

end polynomial_factor_l3736_373673


namespace largest_prime_divisor_l3736_373665

/-- Converts a base 4 number to decimal --/
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The base 4 representation of the number --/
def number : List Nat := [1, 2, 0, 1, 0, 0, 2, 0, 1]

/-- The decimal representation of the number --/
def decimalNumber : Nat := base4ToDecimal number

theorem largest_prime_divisor :
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ decimalNumber ∧ ∀ (q : Nat), Nat.Prime q → q ∣ decimalNumber → q ≤ p ∧ p = 181 := by
  sorry

end largest_prime_divisor_l3736_373665


namespace cone_volume_l3736_373616

theorem cone_volume (cylinder_volume : ℝ) (h : cylinder_volume = 30) :
  let cone_volume := cylinder_volume / 3
  cone_volume = 10 := by
  sorry

end cone_volume_l3736_373616


namespace no_prime_pair_sum_51_l3736_373654

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Theorem statement
theorem no_prime_pair_sum_51 :
  ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 51 :=
sorry

end no_prime_pair_sum_51_l3736_373654


namespace roger_spent_calculation_l3736_373694

/-- Calculates the amount of money Roger spent given his initial amount,
    the amount received from his mom, and his current amount. -/
def money_spent (initial : ℕ) (received : ℕ) (current : ℕ) : ℕ :=
  initial + received - current

theorem roger_spent_calculation :
  money_spent 45 46 71 = 20 := by
  sorry

end roger_spent_calculation_l3736_373694


namespace sufficient_condition_range_l3736_373604

theorem sufficient_condition_range (m : ℝ) : m > 0 →
  (∀ x : ℝ, x^2 - 8*x - 20 ≤ 0 → (1 - m ≤ x ∧ x ≤ 1 + m)) ∧ 
  (∃ x : ℝ, 1 - m ≤ x ∧ x ≤ 1 + m ∧ x^2 - 8*x - 20 > 0) ↔ 
  m ≥ 9 := by
sorry

end sufficient_condition_range_l3736_373604


namespace smallest_positive_solution_l3736_373632

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt (x + 1) = 3 * x - 1 ∧
  ∀ (y : ℝ), y > 0 ∧ Real.sqrt (y + 1) = 3 * y - 1 → x ≤ y :=
by
  -- The solution is x = 7/9
  use 7/9
  sorry

end smallest_positive_solution_l3736_373632


namespace sqrt_sum_equals_seven_l3736_373679

theorem sqrt_sum_equals_seven (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end sqrt_sum_equals_seven_l3736_373679


namespace sequence_max_ratio_l3736_373620

theorem sequence_max_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → S n = (n + 1) / 2 * a n) →
  (∃ M : ℝ, ∀ n : ℕ, n > 1 → a n / a (n - 1) ≤ M) ∧
  (∀ ε > 0, ∃ n : ℕ, n > 1 ∧ a n / a (n - 1) > 2 - ε) :=
by sorry

end sequence_max_ratio_l3736_373620


namespace trash_can_ratio_l3736_373612

/-- Represents the number of trash cans added to the streets -/
def street_cans : ℕ := 14

/-- Represents the total number of trash cans -/
def total_cans : ℕ := 42

/-- Represents the number of trash cans added to the back of stores -/
def store_cans : ℕ := total_cans - street_cans

/-- The ratio of trash cans added to the back of stores to trash cans added to the streets -/
theorem trash_can_ratio : (store_cans : ℚ) / street_cans = 2 / 1 := by
  sorry

end trash_can_ratio_l3736_373612


namespace gaochun_population_eq_scientific_l3736_373633

/-- The population of Gaochun County -/
def gaochun_population : ℕ := 425000

/-- Scientific notation representation of Gaochun County's population -/
def gaochun_population_scientific : ℝ := 4.25 * (10 ^ 5)

/-- Theorem stating that the scientific notation representation is equal to the actual population -/
theorem gaochun_population_eq_scientific : ↑gaochun_population = gaochun_population_scientific := by
  sorry

end gaochun_population_eq_scientific_l3736_373633


namespace inscribed_rectangle_area_l3736_373609

/-- The area of a rectangle inscribed in an ellipse -/
theorem inscribed_rectangle_area :
  ∀ (a b : ℝ),
  (a^2 / 4 + b^2 / 8 = 1) →  -- Rectangle vertices satisfy ellipse equation
  (2 * a = b) →             -- Length along x-axis is twice the length along y-axis
  (4 * a * b = 16 / 3) :=   -- Area of the rectangle is 16/3
by
  sorry

end inscribed_rectangle_area_l3736_373609


namespace sally_takes_home_17_pens_l3736_373602

/-- Calculates the number of pens Sally takes home --/
def pens_taken_home (total_pens : ℕ) (num_students : ℕ) (pens_per_student : ℕ) : ℕ :=
  let pens_given := num_students * pens_per_student
  let pens_left := total_pens - pens_given
  let pens_in_locker := pens_left / 2
  pens_left - pens_in_locker

/-- Proves that Sally takes home 17 pens --/
theorem sally_takes_home_17_pens :
  pens_taken_home 342 44 7 = 17 := by
  sorry

#eval pens_taken_home 342 44 7

end sally_takes_home_17_pens_l3736_373602


namespace bipin_chandan_age_ratio_l3736_373687

/-- Proves that the ratio of Bipin's age to Chandan's age after 10 years is 2:1 -/
theorem bipin_chandan_age_ratio :
  let alok_age : ℕ := 5
  let bipin_age : ℕ := 6 * alok_age
  let chandan_age : ℕ := 7 + 3
  let bipin_future_age : ℕ := bipin_age + 10
  let chandan_future_age : ℕ := chandan_age + 10
  (bipin_future_age : ℚ) / chandan_future_age = 2 := by
  sorry

end bipin_chandan_age_ratio_l3736_373687


namespace total_earnings_l3736_373619

theorem total_earnings (jerusha_earnings lottie_earnings : ℕ) :
  jerusha_earnings = 68 →
  jerusha_earnings = 4 * lottie_earnings →
  jerusha_earnings + lottie_earnings = 85 :=
by
  sorry

end total_earnings_l3736_373619


namespace abs_value_inequality_l3736_373626

theorem abs_value_inequality (x : ℝ) : |x| < 5 ↔ -5 < x ∧ x < 5 := by sorry

end abs_value_inequality_l3736_373626


namespace complex_argument_range_l3736_373668

theorem complex_argument_range (z : ℂ) (h : Complex.abs (2 * z + z⁻¹) = 1) :
  let arg := Complex.arg z
  arg ∈ (Set.Icc (Real.arccos (Real.sqrt 2 / 4)) (Real.pi - Real.arccos (Real.sqrt 2 / 4))) ∪
           (Set.Icc (Real.pi + Real.arccos (Real.sqrt 2 / 4)) (2 * Real.pi - Real.arccos (Real.sqrt 2 / 4))) :=
by sorry

end complex_argument_range_l3736_373668


namespace irrationality_of_sqrt_two_and_rationality_of_others_l3736_373658

-- Define rationality
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- Theorem statement
theorem irrationality_of_sqrt_two_and_rationality_of_others :
  IsIrrational (Real.sqrt 2) ∧ 
  IsRational 3.14 ∧ 
  IsRational (22 / 7) ∧ 
  IsRational 0 :=
sorry

end irrationality_of_sqrt_two_and_rationality_of_others_l3736_373658


namespace product_of_fractions_l3736_373647

def fraction (n : ℕ) : ℚ := (n^3 - 1) / (n^3 + 1)

theorem product_of_fractions :
  (fraction 7) * (fraction 8) * (fraction 9) * (fraction 10) * (fraction 11) = 133 / 946 := by
  sorry

end product_of_fractions_l3736_373647
