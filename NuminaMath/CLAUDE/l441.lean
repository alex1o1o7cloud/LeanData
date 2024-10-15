import Mathlib

namespace NUMINAMATH_CALUDE_total_cost_separate_tickets_l441_44170

def adult_ticket_cost : ℕ := 35
def child_ticket_cost : ℕ := 20
def num_adults : ℕ := 2
def num_children : ℕ := 5

theorem total_cost_separate_tickets :
  num_adults * adult_ticket_cost + num_children * child_ticket_cost = 170 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_separate_tickets_l441_44170


namespace NUMINAMATH_CALUDE_calculation_proofs_l441_44123

theorem calculation_proofs :
  (∃ (x : ℝ), x = Real.sqrt 12 * Real.sqrt (1/3) - Real.sqrt 18 + |Real.sqrt 2 - 2| ∧ x = 4 - 4 * Real.sqrt 2) ∧
  (∃ (y : ℝ), y = (7 + 4 * Real.sqrt 3) * (7 - 4 * Real.sqrt 3) - (Real.sqrt 3 - 1)^2 ∧ y = 2 * Real.sqrt 3 - 3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l441_44123


namespace NUMINAMATH_CALUDE_prob_odd_divisor_18_factorial_l441_44115

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of positive integer divisors of n -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- The number of odd positive integer divisors of n -/
def numOddDivisors (n : ℕ) : ℕ := sorry

/-- The probability of a randomly chosen positive integer divisor of n being odd -/
def probOddDivisor (n : ℕ) : ℚ := (numOddDivisors n : ℚ) / (numDivisors n : ℚ)

theorem prob_odd_divisor_18_factorial :
  probOddDivisor (factorial 18) = 1 / 17 := by sorry

end NUMINAMATH_CALUDE_prob_odd_divisor_18_factorial_l441_44115


namespace NUMINAMATH_CALUDE_exists_fibonacci_divisible_by_10000_l441_44129

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem exists_fibonacci_divisible_by_10000 :
  ∃ k, k ≤ 10^8 + 1 ∧ fibonacci k % 10000 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_fibonacci_divisible_by_10000_l441_44129


namespace NUMINAMATH_CALUDE_pond_length_l441_44110

/-- Given a rectangular field with length 112 m and width half of its length,
    and a square-shaped pond inside the field with an area 1/98 of the field's area,
    prove that the length of the pond is 8 meters. -/
theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_area : ℝ) :
  field_length = 112 →
  field_width = field_length / 2 →
  pond_area = (field_length * field_width) / 98 →
  Real.sqrt pond_area = 8 := by
  sorry

end NUMINAMATH_CALUDE_pond_length_l441_44110


namespace NUMINAMATH_CALUDE_pizza_slices_remaining_l441_44179

/-- Given a pizza with 8 slices, if two people each eat 3/2 slices, then 5 slices remain. -/
theorem pizza_slices_remaining (total_slices : ℕ) (slices_per_person : ℚ) (people : ℕ) : 
  total_slices = 8 → slices_per_person = 3/2 → people = 2 → 
  total_slices - (↑people * slices_per_person).num = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_remaining_l441_44179


namespace NUMINAMATH_CALUDE_remainder_a_37_mod_45_l441_44104

def sequence_number (n : ℕ) : ℕ :=
  -- Definition of a_n: integer obtained by writing all integers from 1 to n sequentially
  sorry

theorem remainder_a_37_mod_45 : sequence_number 37 % 45 = 37 := by
  sorry

end NUMINAMATH_CALUDE_remainder_a_37_mod_45_l441_44104


namespace NUMINAMATH_CALUDE_prism_volume_l441_44186

/-- The volume of a right rectangular prism with given face areas and one side length -/
theorem prism_volume (side_area front_area bottom_area : ℝ) (known_side : ℝ)
  (h_side : side_area = 20)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 15)
  (h_known : known_side = 5) :
  ∃ (a b c : ℝ),
    a * b = side_area ∧
    b * c = front_area ∧
    a * c = bottom_area ∧
    b = known_side ∧
    a * b * c = 75 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l441_44186


namespace NUMINAMATH_CALUDE_track_length_l441_44150

/-- The length of a circular track given race conditions -/
theorem track_length (s t a : ℝ) (h₁ : s > 0) (h₂ : t > 0) (h₃ : a > 0) : 
  ∃ x : ℝ, x > 0 ∧ x = (s / (120 * t)) * (Real.sqrt (a^2 + 240 * a * t) - a) :=
by sorry

end NUMINAMATH_CALUDE_track_length_l441_44150


namespace NUMINAMATH_CALUDE_raisin_distribution_l441_44152

/-- Given 5 boxes of raisins with a total of 437 raisins, where one box has 72 raisins,
    another has 74 raisins, and the remaining three boxes have an equal number of raisins,
    prove that each of the three equal boxes contains 97 raisins. -/
theorem raisin_distribution (total_raisins : ℕ) (total_boxes : ℕ) 
  (box1_raisins : ℕ) (box2_raisins : ℕ) (h1 : total_raisins = 437) 
  (h2 : total_boxes = 5) (h3 : box1_raisins = 72) (h4 : box2_raisins = 74) :
  ∃ (equal_box_raisins : ℕ), 
    equal_box_raisins * 3 + box1_raisins + box2_raisins = total_raisins ∧ 
    equal_box_raisins = 97 := by
  sorry

end NUMINAMATH_CALUDE_raisin_distribution_l441_44152


namespace NUMINAMATH_CALUDE_initial_observations_l441_44128

theorem initial_observations (initial_average : ℝ) (new_observation : ℝ) (average_decrease : ℝ) :
  initial_average = 12 →
  new_observation = 5 →
  average_decrease = 1 →
  ∃ n : ℕ, 
    (n : ℝ) * initial_average + new_observation = (n + 1) * (initial_average - average_decrease) ∧
    n = 6 :=
by sorry

end NUMINAMATH_CALUDE_initial_observations_l441_44128


namespace NUMINAMATH_CALUDE_grandmother_age_l441_44184

def cody_age : ℕ := 14
def grandmother_age_multiplier : ℕ := 6

theorem grandmother_age : 
  cody_age * grandmother_age_multiplier = 84 := by
  sorry

end NUMINAMATH_CALUDE_grandmother_age_l441_44184


namespace NUMINAMATH_CALUDE_tan_alpha_value_l441_44121

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = -5) : 
  Real.tan α = -23/16 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l441_44121


namespace NUMINAMATH_CALUDE_train_length_calculation_l441_44163

/-- Calculates the length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 60 →
  man_speed = 6 →
  passing_time = 23.998080153587715 →
  ∃ (train_length : ℝ), abs (train_length - 440) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_train_length_calculation_l441_44163


namespace NUMINAMATH_CALUDE_ratio_problem_l441_44125

theorem ratio_problem (a b : ℚ) (h : (12 * a - 5 * b) / (14 * a - 3 * b) = 4 / 7) :
  a / b = 23 / 28 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l441_44125


namespace NUMINAMATH_CALUDE_speed_is_48_l441_44167

-- Define the duration of the drive in hours
def drive_duration : ℚ := 7/4

-- Define the distance driven in km
def distance_driven : ℚ := 84

-- Theorem stating that the speed is 48 km/h
theorem speed_is_48 : distance_driven / drive_duration = 48 := by
  sorry

end NUMINAMATH_CALUDE_speed_is_48_l441_44167


namespace NUMINAMATH_CALUDE_f_greater_than_one_range_l441_44155

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/2)^x else x^(1/2)

theorem f_greater_than_one_range :
  {a : ℝ | f a > 1} = Set.Ioi 1 ∪ Set.Iic 0 := by sorry

end NUMINAMATH_CALUDE_f_greater_than_one_range_l441_44155


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l441_44161

theorem sum_of_absolute_coefficients (x a a₁ a₂ a₃ a₄ : ℝ) 
  (h : (1 - 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4)
  (ha : a > 0)
  (ha₂ : a₂ > 0)
  (ha₄ : a₄ > 0)
  (ha₁ : a₁ < 0)
  (ha₃ : a₃ < 0) :
  |a| + |a₁| + |a₂| + |a₃| + |a₄| = 81 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l441_44161


namespace NUMINAMATH_CALUDE_pool_volume_l441_44141

/-- Proves that the pool holds 84 gallons of water given the specified conditions. -/
theorem pool_volume (bucket_fill_time : ℕ) (bucket_capacity : ℕ) (total_fill_time : ℕ) :
  bucket_fill_time = 20 →
  bucket_capacity = 2 →
  total_fill_time = 14 * 60 →
  (total_fill_time / bucket_fill_time) * bucket_capacity = 84 := by
  sorry

end NUMINAMATH_CALUDE_pool_volume_l441_44141


namespace NUMINAMATH_CALUDE_perfect_square_and_cube_is_sixth_power_l441_44113

theorem perfect_square_and_cube_is_sixth_power (n : ℕ) :
  (∃ a : ℕ, n = a^2) ∧ (∃ b : ℕ, n = b^3) → ∃ c : ℕ, n = c^6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_and_cube_is_sixth_power_l441_44113


namespace NUMINAMATH_CALUDE_officers_from_six_people_l441_44153

/-- The number of ways to choose three distinct officers from a group of 6 people -/
def choose_officers (n : ℕ) : ℕ :=
  if n ≥ 3 then n * (n - 1) * (n - 2) else 0

/-- Theorem stating that choosing three distinct officers from 6 people results in 120 ways -/
theorem officers_from_six_people :
  choose_officers 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_officers_from_six_people_l441_44153


namespace NUMINAMATH_CALUDE_quadratic_roots_l441_44178

theorem quadratic_roots (p q : ℤ) (h1 : p + q = 198) :
  ∃ x₁ x₂ : ℤ, (x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0) →
  ((x₁ = 2 ∧ x₂ = 200) ∨ (x₁ = 0 ∧ x₂ = -198)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l441_44178


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l441_44187

/-- Proves that the ratio of average speed to still water speed is 42/65 for a boat traveling in a river --/
theorem boat_speed_ratio :
  let still_water_speed : ℝ := 20
  let current_speed : ℝ := 8
  let downstream_distance : ℝ := 10
  let upstream_distance : ℝ := 10
  let downstream_speed : ℝ := still_water_speed + current_speed
  let upstream_speed : ℝ := still_water_speed - current_speed
  let total_time : ℝ := downstream_distance / downstream_speed + upstream_distance / upstream_speed
  let total_distance : ℝ := downstream_distance + upstream_distance
  let average_speed : ℝ := total_distance / total_time
  average_speed / still_water_speed = 42 / 65 := by
  sorry


end NUMINAMATH_CALUDE_boat_speed_ratio_l441_44187


namespace NUMINAMATH_CALUDE_jacks_water_bottles_l441_44160

/-- Represents the problem of determining how many bottles of water Jack initially bought. -/
theorem jacks_water_bottles :
  ∀ (initial_bottles : ℕ),
    (100 : ℚ) - (2 : ℚ) * (initial_bottles : ℚ) - (2 : ℚ) * (2 : ℚ) * (initial_bottles : ℚ) - (5 : ℚ) = (71 : ℚ) →
    initial_bottles = 4 := by
  sorry

end NUMINAMATH_CALUDE_jacks_water_bottles_l441_44160


namespace NUMINAMATH_CALUDE_no_savings_on_group_purchase_l441_44118

def window_price : ℕ := 120

def free_windows (n : ℕ) : ℕ := (n / 10) * 2

def cost (n : ℕ) : ℕ := (n - free_windows n) * window_price

def alice_windows : ℕ := 9
def bob_windows : ℕ := 11
def celina_windows : ℕ := 10

theorem no_savings_on_group_purchase :
  cost (alice_windows + bob_windows + celina_windows) =
  cost alice_windows + cost bob_windows + cost celina_windows :=
by sorry

end NUMINAMATH_CALUDE_no_savings_on_group_purchase_l441_44118


namespace NUMINAMATH_CALUDE_michaels_pets_cats_percentage_l441_44165

/-- Proves that the percentage of cats among Michael's pets is 50% -/
theorem michaels_pets_cats_percentage
  (total_pets : ℕ)
  (dog_percentage : ℚ)
  (bunny_count : ℕ)
  (h1 : total_pets = 36)
  (h2 : dog_percentage = 1/4)
  (h3 : bunny_count = 9)
  (h4 : (dog_percentage * total_pets).num + bunny_count + (total_pets - (dog_percentage * total_pets).num - bunny_count) = total_pets) :
  (total_pets - (dog_percentage * total_pets).num - bunny_count) / total_pets = 1/2 := by
  sorry

#check michaels_pets_cats_percentage

end NUMINAMATH_CALUDE_michaels_pets_cats_percentage_l441_44165


namespace NUMINAMATH_CALUDE_no_real_solutions_l441_44135

theorem no_real_solutions :
  ∀ x : ℝ, (5 * x^2 - 3 * x + 2) / (x + 2) ≠ 2 * x - 3 :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l441_44135


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_diff_l441_44196

-- Define the polynomial
def p (x : ℝ) : ℝ := 45 * x^3 - 75 * x^2 + 33 * x - 2

-- Define the theorem
theorem root_sum_reciprocal_diff (a b c : ℝ) :
  p a = 0 → p b = 0 → p c = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  0 < a → a < 1 →
  0 < b → b < 1 →
  0 < c → c < 1 →
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_diff_l441_44196


namespace NUMINAMATH_CALUDE_scrap_metal_collection_l441_44158

theorem scrap_metal_collection (a b : Nat) :
  a < 10 ∧ b < 10 ∧ 
  (900 + 10 * a + b) - (100 * a + 10 * b + 9) = 216 →
  900 + 10 * a + b = 975 ∧ 100 * a + 10 * b + 9 = 759 :=
by sorry

end NUMINAMATH_CALUDE_scrap_metal_collection_l441_44158


namespace NUMINAMATH_CALUDE_jonah_fish_exchange_l441_44101

/-- The number of new fish Jonah received in exchange -/
def exchange_fish (initial : ℕ) (added : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial + added - eaten)

/-- Theorem stating the number of new fish Jonah received -/
theorem jonah_fish_exchange :
  exchange_fish 14 2 6 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_jonah_fish_exchange_l441_44101


namespace NUMINAMATH_CALUDE_bird_count_proof_l441_44130

/-- The number of birds initially on a branch -/
def initial_birds (initial_parrots : ℕ) (initial_crows : ℕ) : ℕ :=
  initial_parrots + initial_crows

theorem bird_count_proof 
  (initial_parrots : ℕ) 
  (initial_crows : ℕ) 
  (remaining_parrots : ℕ) 
  (remaining_crow : ℕ) 
  (h1 : initial_parrots = 7)
  (h2 : remaining_parrots = 2)
  (h3 : remaining_crow = 1)
  (h4 : initial_parrots - remaining_parrots = initial_crows - remaining_crow) :
  initial_birds initial_parrots initial_crows = 13 := by
  sorry

end NUMINAMATH_CALUDE_bird_count_proof_l441_44130


namespace NUMINAMATH_CALUDE_sine_function_omega_l441_44198

/-- Given a function f(x) = 2sin(ωx + π/6) with ω > 0, if it intersects the y-axis at (0, 1) 
    and has two adjacent x-intercepts A and B such that the area of triangle PAB is π, 
    then ω = 1/2 -/
theorem sine_function_omega (ω : ℝ) (f : ℝ → ℝ) (A B : ℝ) : 
  ω > 0 →
  (∀ x, f x = 2 * Real.sin (ω * x + π / 6)) →
  f 0 = 1 →
  f A = 0 →
  f B = 0 →
  A < B →
  (B - A) * 1 / 2 = π →
  ω = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_omega_l441_44198


namespace NUMINAMATH_CALUDE_probability_at_least_one_six_all_different_l441_44108

-- Define the number of faces on a die
def num_faces : ℕ := 6

-- Define the total number of possible outcomes when rolling three dice
def total_outcomes : ℕ := num_faces ^ 3

-- Define the number of favorable outcomes (at least one 6 and all different)
def favorable_outcomes : ℕ := 60

-- Define the number of outcomes with at least one 6
def outcomes_with_six : ℕ := total_outcomes - (num_faces - 1) ^ 3

-- Theorem statement
theorem probability_at_least_one_six_all_different :
  (favorable_outcomes : ℚ) / outcomes_with_six = 60 / 91 := by
  sorry


end NUMINAMATH_CALUDE_probability_at_least_one_six_all_different_l441_44108


namespace NUMINAMATH_CALUDE_ducks_cows_relationship_l441_44195

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (group : AnimalGroup) : ℕ := 2 * group.ducks + 4 * group.cows

/-- The total number of heads in the group -/
def totalHeads (group : AnimalGroup) : ℕ := group.ducks + group.cows

/-- The theorem stating the relationship between ducks and cows -/
theorem ducks_cows_relationship (group : AnimalGroup) :
  totalLegs group = 3 * totalHeads group + 26 → group.cows = group.ducks + 26 := by
  sorry


end NUMINAMATH_CALUDE_ducks_cows_relationship_l441_44195


namespace NUMINAMATH_CALUDE_container_volume_l441_44112

/-- Given a cube with surface area 864 square units placed inside a cuboidal container
    with a 1 unit gap on all sides, the volume of the container is 2744 cubic units. -/
theorem container_volume (cube_surface_area : ℝ) (gap : ℝ) :
  cube_surface_area = 864 →
  gap = 1 →
  (cube_surface_area / 6).sqrt + 2 * gap ^ 3 = 2744 :=
by sorry

end NUMINAMATH_CALUDE_container_volume_l441_44112


namespace NUMINAMATH_CALUDE_quadratic_expression_equality_l441_44169

theorem quadratic_expression_equality (x : ℝ) (h : 2 * x^2 + 3 * x + 1 = 10) :
  4 * x^2 + 6 * x + 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_equality_l441_44169


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l441_44119

theorem sum_of_two_numbers (x y : ℝ) 
  (sum_eq : x + y = 10)
  (diff_eq : x - y = 8)
  (sq_diff_eq : x^2 - y^2 = 80) : 
  x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l441_44119


namespace NUMINAMATH_CALUDE_complex_fraction_product_l441_44175

theorem complex_fraction_product (a b : ℝ) : 
  (1 + 7 * Complex.I) / (2 - Complex.I) = Complex.mk a b → a * b = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_product_l441_44175


namespace NUMINAMATH_CALUDE_no_partition_sum_product_l441_44142

theorem no_partition_sum_product : ¬ ∃ (x y : ℕ), 
  1 ≤ x ∧ x ≤ 15 ∧ 1 ≤ y ∧ y ≤ 15 ∧ x ≠ y ∧
  x * y = (List.range 16).sum - x - y := by
  sorry

end NUMINAMATH_CALUDE_no_partition_sum_product_l441_44142


namespace NUMINAMATH_CALUDE_cookie_problem_solution_l441_44131

/-- Represents the number of cookies decorated by each person in one cycle -/
structure DecoratingCycle where
  grandmother : ℕ
  mary : ℕ
  john : ℕ

/-- Represents the problem setup -/
structure CookieDecoratingProblem where
  cycle : DecoratingCycle
  trays : ℕ
  cookies_per_tray : ℕ
  grandmother_time_per_cookie : ℕ

def solve_cookie_problem (problem : CookieDecoratingProblem) :
  (ℕ × ℕ × ℕ) :=
sorry

theorem cookie_problem_solution
  (problem : CookieDecoratingProblem)
  (h_cycle : problem.cycle = ⟨5, 3, 2⟩)
  (h_trays : problem.trays = 5)
  (h_cookies_per_tray : problem.cookies_per_tray = 12)
  (h_grandmother_time : problem.grandmother_time_per_cookie = 4) :
  solve_cookie_problem problem = (4, 140, 40) :=
sorry

end NUMINAMATH_CALUDE_cookie_problem_solution_l441_44131


namespace NUMINAMATH_CALUDE_toothpicks_count_l441_44149

/-- The number of small triangles in the base row of the large triangle -/
def base_triangles : ℕ := 1001

/-- The total number of small triangles in the large triangle -/
def total_triangles : ℕ := (base_triangles * (base_triangles + 1)) / 2

/-- The number of toothpicks required to construct the large triangle -/
def toothpicks_required : ℕ := (3 * total_triangles) / 2 + 3 * base_triangles

/-- Theorem stating that the number of toothpicks required is 755255 -/
theorem toothpicks_count : toothpicks_required = 755255 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_count_l441_44149


namespace NUMINAMATH_CALUDE_oak_trees_cut_down_l441_44192

theorem oak_trees_cut_down (initial_trees : ℕ) (remaining_trees : ℕ) :
  initial_trees = 9 →
  remaining_trees = 7 →
  initial_trees - remaining_trees = 2 :=
by sorry

end NUMINAMATH_CALUDE_oak_trees_cut_down_l441_44192


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l441_44139

def a : Fin 2 → ℝ := ![4, 4]
def b (m : ℝ) : Fin 2 → ℝ := ![5, m]
def c : Fin 2 → ℝ := ![1, 3]

theorem perpendicular_vectors (m : ℝ) :
  (∀ i : Fin 2, (a i - 2 * c i) * b m i = 0) ↔ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l441_44139


namespace NUMINAMATH_CALUDE_min_value_problem_l441_44100

theorem min_value_problem (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_6 : x + y + z = 6) : 
  (x^2 + 2*y^2)/(x + y) + (x^2 + 2*z^2)/(x + z) + (y^2 + 2*z^2)/(y + z) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l441_44100


namespace NUMINAMATH_CALUDE_second_to_fourth_l441_44147

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem: If A(a,b) is in the second quadrant, then B(b,a) is in the fourth quadrant -/
theorem second_to_fourth (a b : ℝ) :
  is_in_second_quadrant (Point.mk a b) →
  is_in_fourth_quadrant (Point.mk b a) := by
  sorry

end NUMINAMATH_CALUDE_second_to_fourth_l441_44147


namespace NUMINAMATH_CALUDE_train_crossing_time_l441_44180

/-- The time taken for a train to cross a platform -/
theorem train_crossing_time (train_length : Real) (train_speed_kmph : Real) (platform_length : Real) :
  train_length = 120 ∧ 
  train_speed_kmph = 72 ∧ 
  platform_length = 380.04 →
  (train_length + platform_length) / (train_speed_kmph * 1000 / 3600) = 25.002 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l441_44180


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l441_44173

-- Define set A as a subset of real numbers
variable (A : Set ℝ)

-- Define proposition p
def p (A : Set ℝ) : Prop :=
  ∃ x ∈ A, x^2 - 2*x - 3 < 0

-- Define proposition q
def q (A : Set ℝ) : Prop :=
  ∀ x ∈ A, x^2 - 2*x - 3 < 0

-- Theorem stating that p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient :
  (∀ A : Set ℝ, q A → p A) ∧ (∃ A : Set ℝ, p A ∧ ¬q A) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l441_44173


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l441_44162

theorem mistaken_multiplication (x : ℝ) : 67 * x - 59 * x = 4828 → x = 603.5 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l441_44162


namespace NUMINAMATH_CALUDE_mango_apple_not_orange_count_l441_44117

/-- Given information about fruit preferences --/
structure FruitPreferences where
  apple : Nat
  orange_mango_not_apple : Nat
  all_fruits : Nat
  total_apple : Nat

/-- Calculate the number of people who like mango and apple and dislike orange --/
def mango_apple_not_orange (prefs : FruitPreferences) : Nat :=
  prefs.total_apple - prefs.all_fruits - prefs.orange_mango_not_apple

/-- Theorem stating the result of the calculation --/
theorem mango_apple_not_orange_count 
  (prefs : FruitPreferences) 
  (h1 : prefs.apple = 40)
  (h2 : prefs.orange_mango_not_apple = 7)
  (h3 : prefs.all_fruits = 4)
  (h4 : prefs.total_apple = 47) :
  mango_apple_not_orange prefs = 36 := by
  sorry

#eval mango_apple_not_orange ⟨40, 7, 4, 47⟩

end NUMINAMATH_CALUDE_mango_apple_not_orange_count_l441_44117


namespace NUMINAMATH_CALUDE_polynomial_d_abs_l441_44120

/-- A polynomial with complex roots 3 + i and 3 - i -/
def polynomial (a b c d e : ℤ) : ℂ → ℂ := fun z ↦ 
  a * (z - (3 + Complex.I))^4 + b * (z - (3 + Complex.I))^3 + 
  c * (z - (3 + Complex.I))^2 + d * (z - (3 + Complex.I)) + e

/-- The coefficients have no common factors other than 1 -/
def coprime (a b c d e : ℤ) : Prop := 
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs) d.natAbs) e.natAbs = 1

theorem polynomial_d_abs (a b c d e : ℤ) 
  (h1 : polynomial a b c d e (3 + Complex.I) = 0)
  (h2 : coprime a b c d e) : 
  Int.natAbs d = 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_d_abs_l441_44120


namespace NUMINAMATH_CALUDE_unique_condition_implies_sum_l441_44159

-- Define the set of possible values
def S : Set ℕ := {1, 2, 5}

-- Define the conditions
def condition1 (a b c : ℕ) : Prop := a ≠ 5
def condition2 (a b c : ℕ) : Prop := b = 5
def condition3 (a b c : ℕ) : Prop := c ≠ 2

-- Main theorem
theorem unique_condition_implies_sum (a b c : ℕ) :
  a ∈ S → b ∈ S → c ∈ S →
  a ≠ b → b ≠ c → a ≠ c →
  (condition1 a b c ∨ condition2 a b c ∨ condition3 a b c) →
  (¬condition1 a b c ∨ ¬condition2 a b c) →
  (¬condition1 a b c ∨ ¬condition3 a b c) →
  (¬condition2 a b c ∨ ¬condition3 a b c) →
  100 * a + 10 * b + c = 521 :=
by sorry

end NUMINAMATH_CALUDE_unique_condition_implies_sum_l441_44159


namespace NUMINAMATH_CALUDE_max_value_fraction_l441_44111

theorem max_value_fraction (a b : ℝ) (h1 : a * b = 1) (h2 : a > b) (h3 : b ≥ 2/3) :
  (a - b) / (a^2 + b^2) ≤ 30/97 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l441_44111


namespace NUMINAMATH_CALUDE_problem_solution_l441_44168

theorem problem_solution (a : ℚ) : a + a/3 - a/9 = 10/3 → a = 30/11 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l441_44168


namespace NUMINAMATH_CALUDE_balanced_equation_oxygen_coefficient_l441_44122

/-- Represents a chemical element in a molecule --/
inductive Element
  | As
  | S
  | O

/-- Represents a molecule in a chemical equation --/
structure Molecule where
  elements : List (Element × Nat)

/-- Represents a side of a chemical equation --/
structure EquationSide where
  molecules : List (Molecule × Nat)

/-- Represents a chemical equation --/
structure ChemicalEquation where
  leftSide : EquationSide
  rightSide : EquationSide

/-- Checks if a chemical equation is balanced --/
def isBalanced (eq : ChemicalEquation) : Bool :=
  sorry

/-- Checks if coefficients are the smallest possible integers --/
def hasSmallestCoefficients (eq : ChemicalEquation) : Bool :=
  sorry

/-- The coefficient of O₂ in the balanced equation --/
def oxygenCoefficient (eq : ChemicalEquation) : Nat :=
  sorry

theorem balanced_equation_oxygen_coefficient :
  ∀ (eq : ChemicalEquation),
    eq.leftSide.molecules = [
      (Molecule.mk [(Element.As, 2), (Element.S, 3)], 2),
      (Molecule.mk [(Element.O, 2)], oxygenCoefficient eq)
    ] →
    eq.rightSide.molecules = [
      (Molecule.mk [(Element.As, 2), (Element.O, 3)], 4),
      (Molecule.mk [(Element.S, 1), (Element.O, 2)], 6)
    ] →
    isBalanced eq →
    hasSmallestCoefficients eq →
    oxygenCoefficient eq = 9 :=
  sorry

end NUMINAMATH_CALUDE_balanced_equation_oxygen_coefficient_l441_44122


namespace NUMINAMATH_CALUDE_math_exam_questions_l441_44185

theorem math_exam_questions (english_questions : ℕ) (english_time : ℕ) (math_time : ℕ) (extra_time_per_question : ℕ) : 
  english_questions = 30 →
  english_time = 60 →
  math_time = 90 →
  extra_time_per_question = 4 →
  (math_time / (english_time / english_questions + extra_time_per_question) : ℕ) = 15 := by
sorry

end NUMINAMATH_CALUDE_math_exam_questions_l441_44185


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l441_44103

/-- A hyperbola with center at the origin, focus on the y-axis, and eccentricity √5 -/
structure Hyperbola where
  /-- The eccentricity of the hyperbola -/
  e : ℝ
  /-- The eccentricity is √5 -/
  h_e : e = Real.sqrt 5
  /-- The center is at the origin -/
  center : ℝ × ℝ
  h_center : center = (0, 0)
  /-- The focus is on the y-axis -/
  focus : ℝ × ℝ
  h_focus : focus.1 = 0

/-- The equations of the asymptotes of the hyperbola -/
def asymptotes (h : Hyperbola) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (1/2) * p.1 ∨ p.2 = -(1/2) * p.1}

/-- Theorem: The asymptotes of the given hyperbola are y = ± (1/2)x -/
theorem hyperbola_asymptotes (h : Hyperbola) : 
  asymptotes h = {p : ℝ × ℝ | p.2 = (1/2) * p.1 ∨ p.2 = -(1/2) * p.1} := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l441_44103


namespace NUMINAMATH_CALUDE_hyperbola_perimeter_l441_44124

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the properties of the hyperbola and points
def hyperbola_properties (F₁ F₂ P Q : ℝ × ℝ) : Prop :=
  ∃ (l : Set (ℝ × ℝ)),
    hyperbola P.1 P.2 ∧ 
    hyperbola Q.1 Q.2 ∧
    P ∈ l ∧ Q ∈ l ∧
    F₁.1 < P.1 ∧ F₁.1 < Q.1 ∧
    F₂.1 > F₁.1 ∧
    ‖P - Q‖ = 4

-- Theorem statement
theorem hyperbola_perimeter (F₁ F₂ P Q : ℝ × ℝ) 
  (h : hyperbola_properties F₁ F₂ P Q) :
  ‖P - F₂‖ + ‖Q - F₂‖ + ‖P - Q‖ = 12 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_perimeter_l441_44124


namespace NUMINAMATH_CALUDE_tile_problem_l441_44176

theorem tile_problem (total_tiles : ℕ) : 
  (∃ n : ℕ, total_tiles = n^2 + 36 ∧ total_tiles = (n + 1)^2 + 3) → 
  total_tiles = 292 := by
sorry

end NUMINAMATH_CALUDE_tile_problem_l441_44176


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l441_44102

theorem largest_n_with_unique_k : ∀ n : ℕ,
  n > 112 →
  ¬(∃! k : ℤ, (8 : ℚ)/15 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 7/13) ∧
  (∃! k : ℤ, (8 : ℚ)/15 < (112 : ℚ)/(112 + k) ∧ (112 : ℚ)/(112 + k) < 7/13) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l441_44102


namespace NUMINAMATH_CALUDE_garden_minimum_width_l441_44193

theorem garden_minimum_width :
  ∀ w : ℝ,
  w > 0 →
  w * (w + 12) ≥ 120 →
  w ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_garden_minimum_width_l441_44193


namespace NUMINAMATH_CALUDE_triangle_side_length_l441_44183

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a = 4 →
  B = π / 3 →
  A = π / 4 →
  b = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l441_44183


namespace NUMINAMATH_CALUDE_rectangle_area_l441_44140

/-- Rectangle ABCD with given properties -/
structure Rectangle where
  -- Length of the rectangle
  length : ℝ
  -- Width of the rectangle
  width : ℝ
  -- Point E on AB
  BE : ℝ
  -- Point F on CD
  CF : ℝ
  -- Length is thrice the width
  length_eq : length = 3 * width
  -- BE is twice CF
  BE_eq : BE = 2 * CF
  -- BE is less than AB (length)
  BE_lt_length : BE < length
  -- CF is less than CD (width)
  CF_lt_width : CF < width
  -- AB is 18 cm
  AB_eq : length = 18
  -- BE is 12 cm
  BE_eq_12 : BE = 12

/-- Theorem stating the area of the rectangle -/
theorem rectangle_area (rect : Rectangle) : rect.length * rect.width = 108 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l441_44140


namespace NUMINAMATH_CALUDE_shawna_situps_wednesday_l441_44177

/-- Calculates the number of situps Shawna needs to do on Wednesday -/
def situps_needed_wednesday (daily_goal : ℕ) (monday_situps : ℕ) (tuesday_situps : ℕ) : ℕ :=
  daily_goal + (daily_goal - monday_situps) + (daily_goal - tuesday_situps)

/-- Theorem: Given Shawna's daily goal and her performance on Monday and Tuesday,
    she needs to do 59 situps on Wednesday to meet her goal and make up for missed situps -/
theorem shawna_situps_wednesday :
  situps_needed_wednesday 30 12 19 = 59 := by
  sorry

end NUMINAMATH_CALUDE_shawna_situps_wednesday_l441_44177


namespace NUMINAMATH_CALUDE_salty_cookies_left_l441_44106

/-- Given the initial number of salty cookies and the number of salty cookies eaten,
    prove that the number of salty cookies left is equal to their difference. -/
theorem salty_cookies_left (initial : ℕ) (eaten : ℕ) (h : eaten ≤ initial) :
  initial - eaten = initial - eaten :=
by sorry

end NUMINAMATH_CALUDE_salty_cookies_left_l441_44106


namespace NUMINAMATH_CALUDE_container_weight_sum_l441_44126

theorem container_weight_sum (x y z : ℝ) 
  (h1 : x + y = 162) 
  (h2 : y + z = 168) 
  (h3 : z + x = 174) : 
  x + y + z = 252 := by
sorry

end NUMINAMATH_CALUDE_container_weight_sum_l441_44126


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l441_44132

def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l441_44132


namespace NUMINAMATH_CALUDE_average_of_numbers_l441_44164

def numbers : List ℕ := [1, 2, 4, 5, 6, 9, 9, 10, 12, 12]

theorem average_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 7 := by sorry

end NUMINAMATH_CALUDE_average_of_numbers_l441_44164


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l441_44189

/-- The number of ways to place distinguishable balls into distinguishable boxes -/
def place_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 2187 ways to place 7 distinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : place_balls 7 3 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l441_44189


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_iff_l441_44151

/-- A curve in the xy-plane parameterized by k -/
def curve (k : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 / (4 + k) + y^2 / (1 - k) = 1}

/-- The condition for the curve to be a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  (4 + k) * (1 - k) < 0

/-- The range of k for which the curve is a hyperbola -/
def hyperbola_range : Set ℝ :=
  {k | k < -4 ∨ k > 1}

/-- Theorem stating that the curve is a hyperbola if and only if k is in the hyperbola_range -/
theorem curve_is_hyperbola_iff (k : ℝ) :
  is_hyperbola k ↔ k ∈ hyperbola_range := by sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_iff_l441_44151


namespace NUMINAMATH_CALUDE_binomial_difference_divisibility_l441_44172

theorem binomial_difference_divisibility (p n : ℕ) (hp : Prime p) (hn : n > p) :
  ∃ k : ℤ, (Nat.choose (n + p - 1) p : ℤ) - (Nat.choose n p : ℤ) = k * n :=
sorry

end NUMINAMATH_CALUDE_binomial_difference_divisibility_l441_44172


namespace NUMINAMATH_CALUDE_specific_figure_perimeter_l441_44114

/-- A figure composed of a triangle and an adjacent quadrilateral -/
structure TriangleQuadrilateralFigure where
  /-- The three sides of the triangle -/
  triangle_side1 : ℝ
  triangle_side2 : ℝ
  triangle_side3 : ℝ
  /-- The side length of the quadrilateral (all sides equal) -/
  quad_side : ℝ

/-- The perimeter of the TriangleQuadrilateralFigure -/
def perimeter (figure : TriangleQuadrilateralFigure) : ℝ :=
  figure.triangle_side1 + figure.triangle_side2 + figure.triangle_side3 + 4 * figure.quad_side

/-- Theorem stating that the perimeter of the specific figure is 44 -/
theorem specific_figure_perimeter :
  ∃ (figure : TriangleQuadrilateralFigure),
    figure.triangle_side1 = 6 ∧
    figure.triangle_side2 = 8 ∧
    figure.triangle_side3 = 10 ∧
    figure.quad_side = 5 ∧
    perimeter figure = 44 :=
sorry

end NUMINAMATH_CALUDE_specific_figure_perimeter_l441_44114


namespace NUMINAMATH_CALUDE_max_distribution_girls_l441_44116

theorem max_distribution_girls (bags : Nat) (eyeliners : Nat) 
  (h1 : bags = 2923) (h2 : eyeliners = 3239) : 
  Nat.gcd bags eyeliners = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_distribution_girls_l441_44116


namespace NUMINAMATH_CALUDE_line_intercept_product_l441_44157

/-- Given a line with equation y + 3 = -2(x + 5), 
    the product of its x-intercept and y-intercept is 84.5 -/
theorem line_intercept_product : 
  ∀ (x y : ℝ), y + 3 = -2 * (x + 5) → 
  ∃ (x_int y_int : ℝ), 
    (x_int + 5 = -13/2) ∧ 
    (y_int + 3 = -2 * 5) ∧ 
    (x_int * y_int = 84.5) := by
  sorry


end NUMINAMATH_CALUDE_line_intercept_product_l441_44157


namespace NUMINAMATH_CALUDE_curve_symmetric_about_y_eq_neg_x_l441_44171

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop := x * y^2 - x^2 * y = -2

-- Define symmetry about y = -x
def symmetric_about_y_eq_neg_x (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ f (-y) (-x)

-- Theorem statement
theorem curve_symmetric_about_y_eq_neg_x :
  symmetric_about_y_eq_neg_x curve_equation :=
sorry

end NUMINAMATH_CALUDE_curve_symmetric_about_y_eq_neg_x_l441_44171


namespace NUMINAMATH_CALUDE_intersection_M_N_l441_44137

-- Define set M
def M : Set ℝ := {x | Real.sqrt (x + 1) ≥ 0}

-- Define set N
def N : Set ℝ := {x | x^2 + x - 2 < 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l441_44137


namespace NUMINAMATH_CALUDE_magnitude_of_vector_difference_l441_44182

/-- Given two vectors a and b in a plane with an angle of π/2 between them,
    |a| = 1, and |b| = √3, prove that |2a - b| = √7 -/
theorem magnitude_of_vector_difference (a b : ℝ × ℝ) :
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- angle between a and b is π/2
  (a.1^2 + a.2^2 = 1) →  -- |a| = 1
  (b.1^2 + b.2^2 = 3) →  -- |b| = √3
  ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2 = 7) :=  -- |2a - b| = √7
by sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_difference_l441_44182


namespace NUMINAMATH_CALUDE_import_tax_percentage_l441_44194

/-- The import tax percentage problem -/
theorem import_tax_percentage 
  (total_value : ℝ) 
  (tax_threshold : ℝ) 
  (tax_paid : ℝ) 
  (h1 : total_value = 2590)
  (h2 : tax_threshold = 1000)
  (h3 : tax_paid = 111.30)
  : (tax_paid / (total_value - tax_threshold)) = 0.07 := by
  sorry

end NUMINAMATH_CALUDE_import_tax_percentage_l441_44194


namespace NUMINAMATH_CALUDE_decrease_amount_l441_44156

theorem decrease_amount (x y : ℝ) : x = 50 → (1/5) * x - y = 5 → y = 5 := by sorry

end NUMINAMATH_CALUDE_decrease_amount_l441_44156


namespace NUMINAMATH_CALUDE_tournament_games_32_teams_l441_44134

/-- The number of games required in a single-elimination tournament --/
def games_required (n : ℕ) : ℕ := n - 1

/-- A theorem stating that a single-elimination tournament with 32 teams requires 31 games --/
theorem tournament_games_32_teams :
  games_required 32 = 31 :=
by sorry

end NUMINAMATH_CALUDE_tournament_games_32_teams_l441_44134


namespace NUMINAMATH_CALUDE_roster_adjustment_count_l441_44146

/-- Represents the number of class officers -/
def num_officers : ℕ := 5

/-- Represents the number of days in the duty roster -/
def num_days : ℕ := 5

/-- The number of ways to arrange the original Monday and Friday officers -/
def arrange_mon_fri : ℕ := 6

/-- The number of ways to choose an officer for each of Tuesday, Wednesday, and Thursday -/
def arrange_tue_thu : ℕ := 2

/-- The number of ways to arrange the remaining two officers for each of Tuesday, Wednesday, and Thursday -/
def arrange_remaining : ℕ := 2

/-- Theorem stating the total number of ways to adjust the roster -/
theorem roster_adjustment_count :
  (arrange_mon_fri * arrange_tue_thu * arrange_remaining) = 24 :=
sorry

end NUMINAMATH_CALUDE_roster_adjustment_count_l441_44146


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l441_44191

theorem perfect_square_quadratic (x k : ℝ) : 
  (∃ b : ℝ, ∀ x, x^2 - 20*x + k = (x + b)^2) ↔ k = 100 := by sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l441_44191


namespace NUMINAMATH_CALUDE_adjacent_angles_theorem_l441_44174

/-- Given two adjacent angles forming a straight line, where one angle is 4x and the other is x, 
    prove that x = 18°. -/
theorem adjacent_angles_theorem (x : ℝ) : 
  (4 * x + x = 180) → x = 18 := by sorry

end NUMINAMATH_CALUDE_adjacent_angles_theorem_l441_44174


namespace NUMINAMATH_CALUDE_expression_value_l441_44188

theorem expression_value (x y : ℝ) (h : x / (2 * y) = 3 / 2) :
  (7 * x + 6 * y) / (x - 2 * y) = 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l441_44188


namespace NUMINAMATH_CALUDE_division_remainder_zero_l441_44105

theorem division_remainder_zero : 
  1234567 % 112 = 0 := by sorry

end NUMINAMATH_CALUDE_division_remainder_zero_l441_44105


namespace NUMINAMATH_CALUDE_factorial_ratio_l441_44199

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l441_44199


namespace NUMINAMATH_CALUDE_unique_c_value_l441_44144

theorem unique_c_value : ∃! (c : ℝ), c ≠ 0 ∧
  (∃! (b : ℝ), b > 0 ∧
    (∃! (x : ℝ), x^2 + (b + 1/b + 1) * x + c = 0)) ∧
  c = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_value_l441_44144


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l441_44136

theorem sqrt_difference_equality : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l441_44136


namespace NUMINAMATH_CALUDE_relay_team_arrangements_l441_44138

/-- The number of ways to arrange 4 people in a line with one fixed in the second position -/
def fixed_second_arrangements : ℕ := 6

/-- The total number of team members -/
def team_size : ℕ := 4

/-- The position where Jordan is fixed -/
def jordans_position : ℕ := 2

theorem relay_team_arrangements :
  (team_size = 4) →
  (jordans_position = 2) →
  (fixed_second_arrangements = 6) := by
sorry

end NUMINAMATH_CALUDE_relay_team_arrangements_l441_44138


namespace NUMINAMATH_CALUDE_remaining_time_for_finger_exerciser_l441_44143

theorem remaining_time_for_finger_exerciser (total_time piano_time writing_time history_time : ℕ) :
  total_time = 120 ∧ piano_time = 30 ∧ writing_time = 25 ∧ history_time = 38 →
  total_time - (piano_time + writing_time + history_time) = 27 := by
sorry

end NUMINAMATH_CALUDE_remaining_time_for_finger_exerciser_l441_44143


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l441_44154

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 7 → x = 54 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l441_44154


namespace NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l441_44127

-- Define the concept of a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop := sorry

-- Define what it means for two lines to have no common point
def no_common_point (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem skew_lines_sufficient_not_necessary :
  ∀ (l1 l2 : Line3D),
    (are_skew l1 l2 → no_common_point l1 l2) ∧
    ¬(no_common_point l1 l2 → are_skew l1 l2) :=
by sorry

end NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l441_44127


namespace NUMINAMATH_CALUDE_squared_sum_geq_one_l441_44148

theorem squared_sum_geq_one (a b c : ℝ) (h : a * b + b * c + c * a = 1) :
  a^2 + b^2 + c^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_squared_sum_geq_one_l441_44148


namespace NUMINAMATH_CALUDE_decimal_to_binary_2008_l441_44145

theorem decimal_to_binary_2008 :
  ∃ (binary : List Bool),
    binary.length = 11 ∧
    (binary.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0 = 2008) ∧
    binary = [true, true, true, true, true, false, true, true, false, false, false] := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_2008_l441_44145


namespace NUMINAMATH_CALUDE_subset_sum_exists_l441_44190

theorem subset_sum_exists (nums : List ℕ) : 
  nums.length = 100 → 
  (∀ n ∈ nums, n ≤ 100) → 
  nums.sum = 200 → 
  ∃ subset : List ℕ, subset ⊆ nums ∧ subset.sum = 100 := by
  sorry

end NUMINAMATH_CALUDE_subset_sum_exists_l441_44190


namespace NUMINAMATH_CALUDE_smallest_angle_theorem_l441_44109

open Real

theorem smallest_angle_theorem : 
  let θ : ℝ := 90
  ∀ φ : ℝ, φ > 0 → φ < θ → 
    cos (φ * π / 180) ≠ sin (50 * π / 180) + cos (32 * π / 180) - sin (22 * π / 180) - cos (16 * π / 180) →
    cos (θ * π / 180) = sin (50 * π / 180) + cos (32 * π / 180) - sin (22 * π / 180) - cos (16 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_theorem_l441_44109


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l441_44107

/-- Linear function f(x) = -2x - 7 -/
def f (x : ℝ) : ℝ := -2 * x - 7

theorem y1_less_than_y2 (x₁ y₁ y₂ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f (x₁ - 1) = y₂) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l441_44107


namespace NUMINAMATH_CALUDE_smallest_exponent_of_ten_l441_44197

theorem smallest_exponent_of_ten (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 2012 → 
  a.factorial * b.factorial * c.factorial = m * 10^n → 
  ¬(10 ∣ m) → 
  (∀ k : ℕ, k < n → ∃ (m' : ℕ), a.factorial * b.factorial * c.factorial = m' * 10^k ∧ 10 ∣ m') →
  n = 501 := by
sorry

end NUMINAMATH_CALUDE_smallest_exponent_of_ten_l441_44197


namespace NUMINAMATH_CALUDE_johns_wrong_marks_l441_44166

/-- Proves that John's wrongly entered marks are 102 given the conditions of the problem -/
theorem johns_wrong_marks (n : ℕ) (actual_marks wrong_marks : ℝ) 
  (h1 : n = 80)  -- Number of students in the class
  (h2 : actual_marks = 62)  -- John's actual marks
  (h3 : (wrong_marks - actual_marks) / n = 1/2)  -- Average increase due to wrong entry
  : wrong_marks = 102 := by
  sorry

end NUMINAMATH_CALUDE_johns_wrong_marks_l441_44166


namespace NUMINAMATH_CALUDE_stating_parallelogram_count_theorem_l441_44133

/-- 
Given a triangle ABC where each side is divided into n equal parts and lines are drawn 
parallel to the sides through each division point, the function returns the total number 
of parallelograms formed in the resulting figure.
-/
def parallelogram_count (n : ℕ) : ℕ :=
  3 * Nat.choose (n + 2) 4

/-- 
Theorem stating that the number of parallelograms in a triangle with sides divided into 
n equal parts and lines drawn parallel to sides through division points is 
3 * (n+2 choose 4).
-/
theorem parallelogram_count_theorem (n : ℕ) : 
  parallelogram_count n = 3 * Nat.choose (n + 2) 4 := by
  sorry

#eval parallelogram_count 5  -- Example evaluation

end NUMINAMATH_CALUDE_stating_parallelogram_count_theorem_l441_44133


namespace NUMINAMATH_CALUDE_smallest_product_of_two_digit_numbers_l441_44181

-- Define a function to create all possible two-digit numbers from four digits
def twoDigitNumbers (a b c d : Nat) : List (Nat × Nat) :=
  [(10*a + b, 10*c + d), (10*a + c, 10*b + d), (10*a + d, 10*b + c),
   (10*b + a, 10*c + d), (10*b + c, 10*a + d), (10*b + d, 10*a + c),
   (10*c + a, 10*b + d), (10*c + b, 10*a + d), (10*c + d, 10*a + b),
   (10*d + a, 10*b + c), (10*d + b, 10*a + c), (10*d + c, 10*a + b)]

-- Define the theorem
theorem smallest_product_of_two_digit_numbers :
  let digits := [2, 4, 5, 8]
  let products := (twoDigitNumbers 2 4 5 8).map (fun (x, y) => x * y)
  (products.minimum? : Option Nat) = some 1200 := by sorry

end NUMINAMATH_CALUDE_smallest_product_of_two_digit_numbers_l441_44181
