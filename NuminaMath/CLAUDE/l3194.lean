import Mathlib

namespace smallest_number_l3194_319435

theorem smallest_number (a b c d : ℤ) (ha : a = -2) (hb : b = -1) (hc : c = 1) (hd : d = 0) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by
  sorry

end smallest_number_l3194_319435


namespace proposition_implication_l3194_319477

theorem proposition_implication (P : ℕ → Prop) :
  (∀ k : ℕ, k ≥ 1 → (P k → P (k + 1))) →
  (¬ P 10) →
  (¬ P 9) := by
  sorry

end proposition_implication_l3194_319477


namespace at_most_four_greater_than_one_l3194_319496

theorem at_most_four_greater_than_one 
  (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : (Real.sqrt (a * b) - 1) * (Real.sqrt (b * c) - 1) * (Real.sqrt (c * a) - 1) = 1) : 
  ∃ (S : Finset ℝ), S ⊆ {a - b/c, a - c/b, b - a/c, b - c/a, c - a/b, c - b/a} ∧ 
    S.card ≤ 4 ∧ 
    (∀ x ∈ S, x > 1) ∧
    (∀ y ∈ {a - b/c, a - c/b, b - a/c, b - c/a, c - a/b, c - b/a} \ S, y ≤ 1) :=
by sorry

end at_most_four_greater_than_one_l3194_319496


namespace factorial_fraction_equality_l3194_319455

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 22 / 21 := by
  sorry

end factorial_fraction_equality_l3194_319455


namespace chair_and_vase_cost_indeterminate_l3194_319428

/-- Represents the cost of items at a garage sale. -/
structure GarageSale where
  total : ℝ
  table : ℝ
  chairs : ℕ
  mirror : ℝ
  lamp : ℝ
  vases : ℕ
  chair_cost : ℝ
  vase_cost : ℝ

/-- Conditions of Nadine's garage sale purchase -/
def nadines_purchase : GarageSale where
  total := 105
  table := 34
  chairs := 2
  mirror := 15
  lamp := 6
  vases := 3
  chair_cost := 0  -- placeholder, actual value unknown
  vase_cost := 0   -- placeholder, actual value unknown

/-- Theorem stating that the sum of one chair and one vase cost cannot be uniquely determined -/
theorem chair_and_vase_cost_indeterminate (g : GarageSale) (h : g = nadines_purchase) :
  ¬ ∃! x : ℝ, x = g.chair_cost + g.vase_cost ∧
    g.total = g.table + g.mirror + g.lamp + g.chairs * g.chair_cost + g.vases * g.vase_cost :=
sorry

end chair_and_vase_cost_indeterminate_l3194_319428


namespace sum_of_coefficients_l3194_319494

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem sum_of_coefficients (a b c : ℝ) :
  (∀ x, f a b c (x + 5) = 4 * x^2 + 9 * x + 2) →
  a + b + c = 30 := by
  sorry

end sum_of_coefficients_l3194_319494


namespace third_quadrant_condition_l3194_319414

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (3 + Complex.I) * m - (2 + Complex.I)

-- Define what it means for a complex number to be in the third quadrant
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- State the theorem
theorem third_quadrant_condition (m : ℝ) :
  in_third_quadrant (z m) ↔ m < 0 := by sorry

end third_quadrant_condition_l3194_319414


namespace sample_size_correct_l3194_319478

/-- The sample size that satisfies the given conditions -/
def sample_size : ℕ := 6

/-- The total population size -/
def total_population : ℕ := 36

/-- Theorem stating that the sample size satisfies all conditions -/
theorem sample_size_correct : 
  (sample_size ∣ total_population) ∧ 
  (6 ∣ sample_size) ∧
  (∃ k : ℕ, 35 = k * (sample_size + 1)) := by
  sorry

end sample_size_correct_l3194_319478


namespace quadratic_function_a_range_l3194_319497

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_a_range 
  (a b c : ℝ) 
  (h1 : f a b c (-2) = 1) 
  (h2 : f a b c 2 = 3) 
  (h3 : 0 < c) 
  (h4 : c < 1) : 
  1/4 < a ∧ a < 1/2 :=
sorry

end quadratic_function_a_range_l3194_319497


namespace sum_of_a_equals_two_l3194_319408

theorem sum_of_a_equals_two (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (eq1 : 2*a₁ + a₂ + a₃ + a₄ + a₅ = 1 + (1/8)*a₄)
  (eq2 : 2*a₂ + a₃ + a₄ + a₅ = 2 + (1/4)*a₃)
  (eq3 : 2*a₃ + a₄ + a₅ = 4 + (1/2)*a₂)
  (eq4 : 2*a₄ + a₅ = 6 + a₁) :
  a₁ + a₂ + a₃ + a₄ + a₅ = 2 := by
  sorry


end sum_of_a_equals_two_l3194_319408


namespace y_range_l3194_319422

theorem y_range (x y : ℝ) (h1 : |y - 2*x| = x^2) (h2 : -1 < x) (h3 : x < 0) :
  ∃ (a b : ℝ), a = -3 ∧ b = 0 ∧ a < y ∧ y < b ∧
  ∀ (z : ℝ), (∃ (w : ℝ), -1 < w ∧ w < 0 ∧ |z - 2*w| = w^2) → a ≤ z ∧ z ≤ b :=
sorry

end y_range_l3194_319422


namespace type_b_sample_count_l3194_319445

/-- Represents the number of items of type B in a stratified sample -/
def stratifiedSampleCount (totalPopulation : ℕ) (typeBPopulation : ℕ) (sampleSize : ℕ) : ℕ :=
  (typeBPopulation * sampleSize) / totalPopulation

/-- Theorem stating that the number of type B items in the sample is 15 -/
theorem type_b_sample_count :
  stratifiedSampleCount 5000 1250 60 = 15 := by
  sorry

end type_b_sample_count_l3194_319445


namespace product_of_fractions_l3194_319406

theorem product_of_fractions : (2 : ℚ) / 3 * 3 / 4 * 4 / 5 = 2 / 5 := by
  sorry

end product_of_fractions_l3194_319406


namespace system_solutions_l3194_319453

/-- The system of equations has two solutions with distance 10 between them -/
theorem system_solutions (a : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁^2 + y₁^2 = 26 * (y₁ * Real.sin (2 * a) - x₁ * Real.cos (2 * a))) ∧
    (x₁^2 + y₁^2 = 26 * (y₁ * Real.cos (3 * a) - x₁ * Real.sin (3 * a))) ∧
    (x₂^2 + y₂^2 = 26 * (y₂ * Real.sin (2 * a) - x₂ * Real.cos (2 * a))) ∧
    (x₂^2 + y₂^2 = 26 * (y₂ * Real.cos (3 * a) - x₂ * Real.sin (3 * a))) ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 100)) ↔
  (∃ n : ℤ, 
    (a = π / 10 + 2 * π * n / 5) ∨
    (a = π / 10 + (2 / 5) * Real.arctan (12 / 5) + 2 * π * n / 5) ∨
    (a = π / 10 - (2 / 5) * Real.arctan (12 / 5) + 2 * π * n / 5)) :=
by
  sorry

end system_solutions_l3194_319453


namespace games_this_month_l3194_319451

/-- Represents the number of football games Nancy attended or plans to attend -/
structure FootballGames where
  total : Nat
  lastMonth : Nat
  nextMonth : Nat

/-- Theorem stating that Nancy attended 9 games this month -/
theorem games_this_month (nancy : FootballGames) 
  (h1 : nancy.total = 24) 
  (h2 : nancy.lastMonth = 8) 
  (h3 : nancy.nextMonth = 7) : 
  nancy.total - nancy.lastMonth - nancy.nextMonth = 9 := by
  sorry

#check games_this_month

end games_this_month_l3194_319451


namespace ratio_sum_squares_theorem_l3194_319448

theorem ratio_sum_squares_theorem (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : b = 2 * a) (h5 : c = 4 * a) (h6 : a^2 + b^2 + c^2 = 4725) :
  a + b + c = 105 := by
sorry

end ratio_sum_squares_theorem_l3194_319448


namespace inclination_angle_range_l3194_319484

theorem inclination_angle_range (θ : ℝ) :
  let k := Real.cos θ
  let α := Real.arctan k
  α ∈ Set.Icc 0 (π / 4) ∪ Set.Ico (3 * π / 4) π :=
by sorry

end inclination_angle_range_l3194_319484


namespace equation_solution_l3194_319407

theorem equation_solution : ∃ x : ℝ, (81 : ℝ) ^ (x - 1) / (9 : ℝ) ^ (x + 1) = (729 : ℝ) ^ (x + 2) ∧ x = -9/2 := by
  sorry

end equation_solution_l3194_319407


namespace cos_alpha_value_l3194_319452

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (-3/5, 4/5), prove that cos(α) = -3/5 -/
theorem cos_alpha_value (α : Real) (h1 : ∃ (x y : Real), x = -3/5 ∧ y = 4/5 ∧ 
  (Real.cos α = x ∧ Real.sin α = y)) : Real.cos α = -3/5 := by
  sorry

end cos_alpha_value_l3194_319452


namespace art_project_markers_l3194_319413

/-- Calculates the total number of markers needed for an art project given the distribution of markers among student groups. -/
theorem art_project_markers (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ) 
  (group1_markers_per_student : ℕ) (group2_markers_per_student : ℕ) (group3_markers_per_student : ℕ) :
  total_students = 30 →
  group1_students = 10 →
  group2_students = 15 →
  group1_markers_per_student = 2 →
  group2_markers_per_student = 4 →
  group3_markers_per_student = 6 →
  (group1_students * group1_markers_per_student + 
   group2_students * group2_markers_per_student + 
   (total_students - group1_students - group2_students) * group3_markers_per_student) = 110 :=
by sorry


end art_project_markers_l3194_319413


namespace p_sufficient_not_necessary_for_q_l3194_319489

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (|2*x - 3| < 1 → x*(x - 3) < 0)) ∧
  (∃ x : ℝ, x*(x - 3) < 0 ∧ ¬(|2*x - 3| < 1)) :=
by sorry

end p_sufficient_not_necessary_for_q_l3194_319489


namespace coefficient_x_squared_sum_binomials_l3194_319404

theorem coefficient_x_squared_sum_binomials : 
  let f (n : ℕ) := (1 + X : Polynomial ℚ)^n
  let sum := (f 4) + (f 5) + (f 6) + (f 7) + (f 8) + (f 9)
  (sum.coeff 2 : ℚ) = 116 := by sorry

end coefficient_x_squared_sum_binomials_l3194_319404


namespace conference_arrangements_l3194_319425

def number_of_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  n.factorial / (2^k)

theorem conference_arrangements :
  number_of_arrangements 8 2 = 10080 := by
  sorry

end conference_arrangements_l3194_319425


namespace spinner_points_south_l3194_319454

/-- Represents the four cardinal directions --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a rotation of the spinner --/
structure Rotation :=
  (revolutions : ℚ)
  (clockwise : Bool)

/-- Calculates the final direction after applying a net rotation --/
def finalDirection (netRotation : ℚ) : Direction :=
  match netRotation.num % 4 with
  | 0 => Direction.North
  | 1 => Direction.East
  | 2 => Direction.South
  | _ => Direction.West

/-- Theorem stating that the given sequence of rotations results in the spinner pointing south --/
theorem spinner_points_south (initialDirection : Direction)
    (rotation1 : Rotation)
    (rotation2 : Rotation)
    (rotation3 : Rotation) :
    initialDirection = Direction.North ∧
    rotation1 = { revolutions := 7/2, clockwise := true } ∧
    rotation2 = { revolutions := 16/3, clockwise := false } ∧
    rotation3 = { revolutions := 13/6, clockwise := true } →
    finalDirection (
      rotation1.revolutions * (if rotation1.clockwise then 1 else -1) +
      rotation2.revolutions * (if rotation2.clockwise then 1 else -1) +
      rotation3.revolutions * (if rotation3.clockwise then 1 else -1)
    ) = Direction.South :=
by sorry

end spinner_points_south_l3194_319454


namespace room_population_problem_l3194_319483

theorem room_population_problem (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →  -- Initial ratio of men to women is 4:5
  (initial_men + 2) = 14 →  -- After 2 men entered, there are now 14 men
  (2 * (initial_women - 3)) = 24 :=  -- Number of women after changes
by
  sorry

end room_population_problem_l3194_319483


namespace constant_term_expansion_l3194_319491

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), (∀ y, f y = (y + 2 + y⁻¹)^3) ∧ 
  (∃ c : ℝ, ∀ z ≠ 0, f z = c + z * (z⁻¹ * (f z - c))) ∧ 
  (∃ c : ℝ, ∀ z ≠ 0, f z = c + z * (z⁻¹ * (f z - c)) ∧ c = 20) :=
sorry

end constant_term_expansion_l3194_319491


namespace prob_one_makes_shot_is_point_seven_l3194_319416

/-- The probability that at least one player makes a shot -/
def prob_at_least_one_makes_shot (prob_a prob_b : ℝ) : ℝ :=
  1 - (1 - prob_a) * (1 - prob_b)

/-- Theorem: Given the shooting success rates of players A and B,
    the probability that at least one of them makes a shot is 0.7 -/
theorem prob_one_makes_shot_is_point_seven :
  prob_at_least_one_makes_shot 0.5 0.4 = 0.7 := by
  sorry

end prob_one_makes_shot_is_point_seven_l3194_319416


namespace binomial_700_700_l3194_319443

theorem binomial_700_700 : Nat.choose 700 700 = 1 := by
  sorry

end binomial_700_700_l3194_319443


namespace boat_speed_in_still_water_l3194_319486

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (stream_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  stream_speed = 5 →
  downstream_distance = 70 →
  downstream_time = 2 →
  ∃ (boat_speed : ℝ), boat_speed = 30 ∧ downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by sorry

end boat_speed_in_still_water_l3194_319486


namespace infinitely_many_primes_4k_plus_1_l3194_319481

theorem infinitely_many_primes_4k_plus_1 :
  ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4*k + 1) →
  ∃ q, Nat.Prime q ∧ (∃ m, q = 4*m + 1) ∧ q ∉ S :=
sorry

end infinitely_many_primes_4k_plus_1_l3194_319481


namespace airplane_cost_is_428_l3194_319418

/-- The cost of an airplane, given the initial amount and change received. -/
def airplane_cost (initial_amount change : ℚ) : ℚ :=
  initial_amount - change

/-- Theorem stating that the cost of the airplane is $4.28 -/
theorem airplane_cost_is_428 :
  airplane_cost 5 0.72 = 4.28 := by
  sorry

end airplane_cost_is_428_l3194_319418


namespace complete_square_quadratic_l3194_319475

theorem complete_square_quadratic (a b c : ℝ) (h : a = 1 ∧ b = -6 ∧ c = -16) :
  ∃ (k m : ℝ), ∀ x, (a * x^2 + b * x + c = 0) ↔ ((x + k)^2 = m) ∧ m = 25 := by
  sorry

end complete_square_quadratic_l3194_319475


namespace polynomial_value_l3194_319430

theorem polynomial_value (m n : ℤ) (h : m - 2*n = 7) : 
  2023 - 2*m + 4*n = 2009 := by
sorry

end polynomial_value_l3194_319430


namespace extremum_property_l3194_319409

/-- Given a function f(x) = 1 - x*sin(x) that attains an extremum at x₀,
    prove that (1 + x₀²)(1 + cos(2x₀)) = 2 -/
theorem extremum_property (x₀ : ℝ) :
  let f : ℝ → ℝ := fun x ↦ 1 - x * Real.sin x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x₀ ≥ f x ∨ f x₀ ≤ f x) →
  (1 + x₀^2) * (1 + Real.cos (2 * x₀)) = 2 := by
  sorry

end extremum_property_l3194_319409


namespace initial_girls_count_l3194_319468

theorem initial_girls_count (initial_boys : ℕ) (boys_left : ℕ) (girls_entered : ℕ) (final_total : ℕ) :
  initial_boys = 5 →
  boys_left = 3 →
  girls_entered = 2 →
  final_total = 8 →
  ∃ initial_girls : ℕ, 
    initial_girls = 4 ∧
    final_total = (initial_boys - boys_left) + (initial_girls + girls_entered) :=
by sorry

end initial_girls_count_l3194_319468


namespace min_squares_6x7_l3194_319419

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- A tiling of a rectangle with squares -/
def Tiling (r : Rectangle) := List Square

/-- The area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ :=
  r.width * r.height

/-- The area of a square -/
def squareArea (s : Square) : ℕ :=
  s.side * s.side

/-- Check if a tiling is valid for a given rectangle -/
def isValidTiling (r : Rectangle) (t : Tiling r) : Prop :=
  (t.map squareArea).sum = rectangleArea r

/-- The main theorem -/
theorem min_squares_6x7 :
  ∃ (t : Tiling ⟨6, 7⟩), 
    isValidTiling ⟨6, 7⟩ t ∧ 
    t.length = 7 ∧ 
    (∀ (t' : Tiling ⟨6, 7⟩), isValidTiling ⟨6, 7⟩ t' → t'.length ≥ 7) :=
  sorry

end min_squares_6x7_l3194_319419


namespace sphere_volume_equals_surface_area_l3194_319456

theorem sphere_volume_equals_surface_area (r : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 := by
  sorry

end sphere_volume_equals_surface_area_l3194_319456


namespace bulb_arrangement_count_l3194_319460

/-- The number of ways to arrange blue, red, and white bulbs in a garland with no consecutive white bulbs -/
def bulb_arrangements (blue red white : ℕ) : ℕ :=
  Nat.choose (blue + red) blue * Nat.choose (blue + red + 1) white

/-- Theorem: The number of ways to arrange 5 blue, 8 red, and 11 white bulbs in a garland 
    with no consecutive white bulbs is equal to (13 choose 5) * (14 choose 11) -/
theorem bulb_arrangement_count : bulb_arrangements 5 8 11 = 468468 := by
  sorry

#eval bulb_arrangements 5 8 11

end bulb_arrangement_count_l3194_319460


namespace evaluate_expression_l3194_319447

theorem evaluate_expression (x : ℝ) (h : x = -3) : 
  (5 + x*(5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
  sorry

end evaluate_expression_l3194_319447


namespace max_value_fraction_l3194_319429

theorem max_value_fraction (x y : ℝ) : 
  (2*x + 3*y + 4) / Real.sqrt (2*x^2 + 3*y^2 + 5) ≤ Real.sqrt 28 := by
  sorry

end max_value_fraction_l3194_319429


namespace shirt_cost_l3194_319402

theorem shirt_cost (j s : ℝ) 
  (eq1 : 3 * j + 2 * s = 69) 
  (eq2 : 2 * j + 3 * s = 81) : 
  s = 21 := by
  sorry

end shirt_cost_l3194_319402


namespace work_completion_time_l3194_319479

/-- The number of days it takes for a group to complete a work -/
def days_to_complete (women : ℕ) (children : ℕ) : ℚ :=
  1 / ((women / 50 : ℚ) + (children / 100 : ℚ))

/-- The theorem stating that 5 women and 10 children working together will complete the work in 5 days -/
theorem work_completion_time :
  days_to_complete 5 10 = 5 := by sorry

end work_completion_time_l3194_319479


namespace exponential_function_passes_through_one_l3194_319433

theorem exponential_function_passes_through_one (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  (fun x => a^x) 0 = 1 := by
  sorry

end exponential_function_passes_through_one_l3194_319433


namespace sum_remainder_mod_11_l3194_319442

theorem sum_remainder_mod_11 : (103104 + 103105 + 103106 + 103107 + 103108 + 103109 + 103110 + 103111 + 103112) % 11 = 4 := by
  sorry

end sum_remainder_mod_11_l3194_319442


namespace book_words_per_page_l3194_319490

theorem book_words_per_page 
  (total_pages : ℕ) 
  (words_per_page : ℕ) 
  (max_words_per_page : ℕ) 
  (total_words_mod : ℕ) :
  total_pages = 150 →
  words_per_page ≤ max_words_per_page →
  max_words_per_page = 120 →
  (total_pages * words_per_page) % 221 = total_words_mod →
  total_words_mod = 200 →
  words_per_page = 118 := by
sorry

end book_words_per_page_l3194_319490


namespace max_profit_is_200_l3194_319492

/-- Represents a neighborhood with its characteristics --/
structure Neighborhood where
  homes : ℕ
  boxes_per_home : ℕ
  price_per_box : ℚ

/-- Calculates the total sales for a neighborhood --/
def total_sales (n : Neighborhood) : ℚ :=
  n.homes * n.boxes_per_home * n.price_per_box

/-- The four neighborhoods with their respective characteristics --/
def neighborhood_A : Neighborhood := ⟨12, 3, 3⟩
def neighborhood_B : Neighborhood := ⟨8, 6, 4⟩
def neighborhood_C : Neighborhood := ⟨15, 2, 5/2⟩
def neighborhood_D : Neighborhood := ⟨5, 8, 5⟩

/-- List of all neighborhoods --/
def neighborhoods : List Neighborhood := [neighborhood_A, neighborhood_B, neighborhood_C, neighborhood_D]

/-- Theorem stating that the maximum profit among the neighborhoods is $200 --/
theorem max_profit_is_200 : 
  (neighborhoods.map total_sales).maximum? = some 200 := by sorry

end max_profit_is_200_l3194_319492


namespace prob_heart_then_king_is_one_fiftytwo_l3194_319423

/-- A standard deck of cards. -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- The suit of a card. -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- The rank of a card. -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A playing card. -/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- The probability of drawing a heart first and a king second from a standard 52-card deck. -/
def prob_heart_then_king (d : Deck) : ℚ :=
  1 / 52

/-- Theorem stating that the probability of drawing a heart first and a king second
    from a standard 52-card deck is 1/52. -/
theorem prob_heart_then_king_is_one_fiftytwo (d : Deck) :
  prob_heart_then_king d = 1 / 52 := by
  sorry

end prob_heart_then_king_is_one_fiftytwo_l3194_319423


namespace fewest_printers_equal_spend_l3194_319410

def printer_cost_1 : ℕ := 400
def printer_cost_2 : ℕ := 350

theorem fewest_printers_equal_spend (cost1 cost2 : ℕ) (h1 : cost1 = printer_cost_1) (h2 : cost2 = printer_cost_2) :
  ∃ (n1 n2 : ℕ), n1 * cost1 = n2 * cost2 ∧ n1 + n2 = 15 ∧ ∀ (m1 m2 : ℕ), m1 * cost1 = m2 * cost2 → m1 + m2 ≥ 15 :=
sorry

end fewest_printers_equal_spend_l3194_319410


namespace quadratic_root_zero_l3194_319474

theorem quadratic_root_zero (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + a^2 - 9 = 0 → x = 0 ∨ x ≠ 0) →
  (0^2 - 2*0 + a^2 - 9 = 0) →
  (a = 3 ∨ a = -3) := by
sorry

end quadratic_root_zero_l3194_319474


namespace distance_less_than_radius_l3194_319421

/-- A circle with center O and radius 3 -/
structure Circle :=
  (O : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 3)

/-- A point P inside the circle -/
structure PointInside (c : Circle) :=
  (P : ℝ × ℝ)
  (h_inside : dist P c.O < c.radius)

/-- Theorem: The distance between the center and a point inside the circle is less than 3 -/
theorem distance_less_than_radius (c : Circle) (p : PointInside c) :
  dist p.P c.O < 3 := by sorry

end distance_less_than_radius_l3194_319421


namespace cars_produced_in_north_america_l3194_319400

theorem cars_produced_in_north_america :
  ∀ (total_cars europe_cars north_america_cars : ℕ),
    total_cars = 6755 →
    europe_cars = 2871 →
    total_cars = north_america_cars + europe_cars →
    north_america_cars = 3884 := by
  sorry

end cars_produced_in_north_america_l3194_319400


namespace intersection_complement_l3194_319482

def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem intersection_complement : S ∩ (U \ T) = {1, 2, 4} := by sorry

end intersection_complement_l3194_319482


namespace two_digit_number_condition_l3194_319431

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ :=
  n / 10

def units_digit (n : ℕ) : ℕ :=
  n % 10

def satisfies_condition (n : ℕ) : Prop :=
  2 * (tens_digit n + units_digit n) = tens_digit n * units_digit n

theorem two_digit_number_condition :
  ∀ n : ℕ, is_valid_two_digit_number n ∧ satisfies_condition n ↔ n = 36 ∨ n = 44 ∨ n = 63 := by
  sorry

end two_digit_number_condition_l3194_319431


namespace line_tangent_to_circle_l3194_319457

/-- The value of 'a' for which the line ax - y + 2 = 0 is tangent to the circle
    x = 2 + 2cos(θ), y = 1 + 2sin(θ) -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ θ : ℝ, (a * (2 + 2 * Real.cos θ) - (1 + 2 * Real.sin θ) + 2 = 0) →
   ∃ θ' : ℝ, (a * (2 + 2 * Real.cos θ') - (1 + 2 * Real.sin θ') + 2 = 0 ∧
              ∀ θ'' : ℝ, θ'' ≠ θ' → 
                a * (2 + 2 * Real.cos θ'') - (1 + 2 * Real.sin θ'') + 2 ≠ 0)) →
  a = 3/4 := by
sorry

end line_tangent_to_circle_l3194_319457


namespace walking_students_speed_l3194_319495

/-- Two students walking towards each other -/
structure WalkingStudents where
  distance : ℝ
  time : ℝ
  speed1 : ℝ
  speed2 : ℝ

/-- The conditions of the problem -/
def problem : WalkingStudents where
  distance := 350
  time := 100
  speed1 := 1.9
  speed2 := 1.6  -- The speed we want to prove

theorem walking_students_speed (w : WalkingStudents) 
  (h1 : w.distance = 350)
  (h2 : w.time = 100)
  (h3 : w.speed1 = 1.9)
  (h4 : w.speed2 * w.time + w.speed1 * w.time = w.distance) :
  w.speed2 = 1.6 := by
  sorry

end walking_students_speed_l3194_319495


namespace increasing_function_condition_l3194_319439

/-- A function f is increasing on ℝ if for all x y, x < y implies f x < f y -/
def IncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_function_condition (f : ℝ → ℝ) (h : IncreasingOn f) :
  ∀ a b : ℝ, a + b < 0 ↔ f a + f b < f (-a) + f (-b) := by
  sorry

end increasing_function_condition_l3194_319439


namespace reading_pages_solution_l3194_319467

/-- The number of pages Xiao Ming's father reads per day -/
def father_pages : ℕ := sorry

/-- The number of pages Xiao Ming reads per day -/
def xiao_ming_pages : ℕ := sorry

/-- Xiao Ming reads 5 pages more than his father every day -/
axiom pages_difference : xiao_ming_pages = father_pages + 5

/-- The time it takes for Xiao Ming to read 100 pages is equal to the time it takes for his father to read 80 pages -/
axiom reading_time_equality : (100 : ℚ) / xiao_ming_pages = (80 : ℚ) / father_pages

theorem reading_pages_solution :
  father_pages = 20 ∧ xiao_ming_pages = 25 :=
sorry

end reading_pages_solution_l3194_319467


namespace rectangle_length_is_one_point_five_times_width_l3194_319450

/-- Represents the configuration of squares and rectangles in a larger square -/
structure SquareConfiguration where
  /-- Side length of a small square -/
  s : ℝ
  /-- Length of a rectangle -/
  l : ℝ
  /-- The configuration forms a square -/
  is_square : 3 * s = 2 * l
  /-- The width of each rectangle equals the side of a small square -/
  width_eq_side : l > s

/-- Theorem stating that the length of each rectangle is 1.5 times its width -/
theorem rectangle_length_is_one_point_five_times_width (config : SquareConfiguration) :
  config.l = 1.5 * config.s := by
  sorry

end rectangle_length_is_one_point_five_times_width_l3194_319450


namespace factorization_proof_l3194_319436

theorem factorization_proof (a x y : ℝ) : 2*a*(x-y) - (x-y) = (x-y)*(2*a-1) := by
  sorry

end factorization_proof_l3194_319436


namespace probability_of_flush_l3194_319498

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Number of cards in each suit -/
def CardsPerSuit : ℕ := 13

/-- Size of a poker hand -/
def HandSize : ℕ := 5

/-- Probability of drawing a flush in a 5-card poker hand -/
theorem probability_of_flush (deck : ℕ) (suits : ℕ) (cards_per_suit : ℕ) (hand_size : ℕ) :
  deck = StandardDeck →
  suits = NumSuits →
  cards_per_suit = CardsPerSuit →
  hand_size = HandSize →
  (suits * (Nat.choose cards_per_suit hand_size) : ℚ) / (Nat.choose deck hand_size) = 33 / 16660 :=
sorry

end probability_of_flush_l3194_319498


namespace georgia_muffins_l3194_319411

/-- Calculates the number of batches of muffins made over a period of months -/
def muffin_batches (students : ℕ) (muffins_per_batch : ℕ) (months : ℕ) : ℕ :=
  (students / muffins_per_batch) * months

theorem georgia_muffins :
  muffin_batches 24 6 9 = 36 := by
  sorry

end georgia_muffins_l3194_319411


namespace no_possible_values_for_a_l3194_319473

def M (a : ℝ) : Set ℝ := {1, 9, a}
def P (a : ℝ) : Set ℝ := {1, a, 2}

theorem no_possible_values_for_a :
  ∀ a : ℝ, (P a) ⊆ (M a) → False :=
sorry

end no_possible_values_for_a_l3194_319473


namespace function_inequality_implies_a_bound_l3194_319459

theorem function_inequality_implies_a_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, 
    x₁ + 4/x₁ ≥ 2^x₂ + a) → a ≤ 1 := by
  sorry

end function_inequality_implies_a_bound_l3194_319459


namespace number_problem_l3194_319458

theorem number_problem (N : ℚ) : 
  (4/15 * 5/7 * N) - (4/9 * 2/5 * N) = 24 → N/2 = 945 := by
sorry

end number_problem_l3194_319458


namespace salamander_population_decline_l3194_319499

def decrease_rate : ℝ := 0.3
def target_percentage : ℝ := 0.05
def start_year : ℕ := 2007

def population_percentage (n : ℕ) : ℝ := (1 - decrease_rate) ^ n

theorem salamander_population_decline :
  ∃ n : ℕ, 
    population_percentage n ≤ target_percentage ∧
    population_percentage (n - 1) > target_percentage ∧
    start_year + n = 2016 :=
  sorry

end salamander_population_decline_l3194_319499


namespace card_distribution_events_l3194_319461

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card
| White : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the events
def EventAGetsRed (d : Distribution) : Prop := d Person.A = Card.Red
def EventBGetsRed (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem card_distribution_events :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventAGetsRed d ∧ EventBGetsRed d)) ∧
  -- The events are not opposite (there exists a distribution where neither event occurs)
  (∃ d : Distribution, ¬EventAGetsRed d ∧ ¬EventBGetsRed d) :=
sorry

end card_distribution_events_l3194_319461


namespace robin_photos_count_l3194_319470

/-- Given that each page holds six photos and Robin can fill 122 full pages,
    prove that Robin has 732 photos in total. -/
theorem robin_photos_count :
  let photos_per_page : ℕ := 6
  let full_pages : ℕ := 122
  photos_per_page * full_pages = 732 :=
by sorry

end robin_photos_count_l3194_319470


namespace square_perimeter_difference_l3194_319469

theorem square_perimeter_difference (a b : ℝ) 
  (h1 : a^2 + b^2 = 85)
  (h2 : a^2 - b^2 = 45) :
  4*a - 4*b = 4*(Real.sqrt 65 - 2*Real.sqrt 5) :=
sorry

end square_perimeter_difference_l3194_319469


namespace hyperbola_asymptote_l3194_319449

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 where a > 0,
    if one of its asymptotes is y = 3/5 * x, then a = 5 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1) →
  (∃ x y : ℝ, y = 3/5 * x) →
  a = 5 := by
sorry

end hyperbola_asymptote_l3194_319449


namespace consecutive_right_triangle_iff_345_l3194_319424

/-- A right-angled triangle with consecutive integer side lengths -/
structure ConsecutiveRightTriangle where
  n : ℕ
  n_pos : 0 < n
  is_right : (n + 1)^2 + n^2 = (n + 2)^2

/-- The property of having sides 3, 4, and 5 -/
def has_sides_345 (t : ConsecutiveRightTriangle) : Prop :=
  t.n = 3

theorem consecutive_right_triangle_iff_345 :
  ∀ t : ConsecutiveRightTriangle, has_sides_345 t ↔ True :=
sorry

end consecutive_right_triangle_iff_345_l3194_319424


namespace min_questions_100_boxes_l3194_319462

/-- Represents the setup of the box guessing game -/
structure BoxGame where
  num_boxes : ℕ
  num_questions : ℕ

/-- Checks if the number of questions is sufficient to determine the prize box -/
def is_sufficient (game : BoxGame) : Prop :=
  game.num_questions + 1 ≥ game.num_boxes

/-- The minimum number of questions needed for a given number of boxes -/
def min_questions (n : ℕ) : ℕ :=
  n - 1

/-- Theorem stating the minimum number of questions needed for 100 boxes -/
theorem min_questions_100_boxes :
  ∃ (game : BoxGame), game.num_boxes = 100 ∧ game.num_questions = 99 ∧ 
  is_sufficient game ∧ 
  ∀ (g : BoxGame), g.num_boxes = 100 → g.num_questions < 99 → ¬is_sufficient g :=
by sorry


end min_questions_100_boxes_l3194_319462


namespace initial_birds_on_fence_l3194_319440

theorem initial_birds_on_fence (initial_birds : ℕ) (initial_storks : ℕ) : 
  (initial_birds + 5 = initial_storks + 4 + 3) → initial_birds = 6 := by
  sorry

end initial_birds_on_fence_l3194_319440


namespace new_person_weight_l3194_319480

/-- The weight of a new person joining a group, given certain conditions -/
theorem new_person_weight (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : 
  n = 8 → 
  avg_increase = 3.5 →
  replaced_weight = 65 →
  (n : ℝ) * avg_increase + replaced_weight = 93 :=
by sorry

end new_person_weight_l3194_319480


namespace inequality_of_cube_roots_l3194_319417

theorem inequality_of_cube_roots (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.rpow ((a / (b + c))^2) (1/3) + Real.rpow ((b / (c + a))^2) (1/3) + Real.rpow ((c / (a + b))^2) (1/3) ≥ 3 / Real.rpow 4 (1/3) := by
  sorry

end inequality_of_cube_roots_l3194_319417


namespace total_legs_bees_and_spiders_l3194_319488

theorem total_legs_bees_and_spiders :
  let bee_legs : ℕ := 6
  let spider_legs : ℕ := 8
  let num_bees : ℕ := 5
  let num_spiders : ℕ := 2
  (num_bees * bee_legs + num_spiders * spider_legs) = 46 :=
by
  sorry

end total_legs_bees_and_spiders_l3194_319488


namespace like_terms_exponents_l3194_319444

/-- Given that 3x^(2n-1)y^m and -5x^m y^3 are like terms, prove that m = 3 and n = 2 -/
theorem like_terms_exponents (m n : ℤ) : 
  (∀ x y : ℝ, 3 * x^(2*n - 1) * y^m = -5 * x^m * y^3) → 
  m = 3 ∧ n = 2 := by
sorry

end like_terms_exponents_l3194_319444


namespace min_value_theorem_l3194_319432

theorem min_value_theorem (x : ℝ) (h : x > -3) :
  x + 2 / (x + 3) ≥ 2 * Real.sqrt 2 - 3 ∧
  (x + 2 / (x + 3) = 2 * Real.sqrt 2 - 3 ↔ x = -3 + Real.sqrt 2) :=
by sorry

end min_value_theorem_l3194_319432


namespace not_all_perfect_squares_l3194_319466

theorem not_all_perfect_squares (k : ℕ) : ¬(∃ a b c : ℤ, (2 * k - 1 = a^2) ∧ (5 * k - 1 = b^2) ∧ (13 * k - 1 = c^2)) := by
  sorry

end not_all_perfect_squares_l3194_319466


namespace fishing_line_length_l3194_319427

/-- Given information about fishing line reels and sections, prove the length of each reel. -/
theorem fishing_line_length (num_reels : ℕ) (section_length : ℝ) (num_sections : ℕ) :
  num_reels = 3 →
  section_length = 10 →
  num_sections = 30 →
  (num_sections * section_length) / num_reels = 100 := by
  sorry

#check fishing_line_length

end fishing_line_length_l3194_319427


namespace triangle_rotation_l3194_319438

/-- Triangle OPQ with specific properties -/
structure TriangleOPQ where
  O : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_O : O = (0, 0)
  h_Q : Q = (6, 0)
  h_P_first_quadrant : P.1 > 0 ∧ P.2 > 0
  h_right_angle : (P.1 - Q.1) * (Q.1 - O.1) + (P.2 - Q.2) * (Q.2 - O.2) = 0
  h_45_degree : (P.1 - O.1) * (Q.1 - O.1) + (P.2 - O.2) * (Q.2 - O.2) = 
                Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) * Real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2) / Real.sqrt 2

/-- Rotation of a point 90 degrees counterclockwise about the origin -/
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)

/-- The main theorem -/
theorem triangle_rotation (t : TriangleOPQ) : rotate90 t.P = (-6, 6) := by
  sorry


end triangle_rotation_l3194_319438


namespace mary_score_l3194_319446

def AHSME_score (c w : ℕ) : ℕ := 30 + 4 * c - w

def unique_solution (s : ℕ) : Prop :=
  ∃! (c w : ℕ), AHSME_score c w = s ∧ c + w ≤ 30

def multiple_solutions (s : ℕ) : Prop :=
  ∃ (c₁ w₁ c₂ w₂ : ℕ), c₁ ≠ c₂ ∧ AHSME_score c₁ w₁ = s ∧ AHSME_score c₂ w₂ = s ∧ c₁ + w₁ ≤ 30 ∧ c₂ + w₂ ≤ 30

theorem mary_score :
  ∃ (s : ℕ),
    s = 119 ∧
    s > 80 ∧
    unique_solution s ∧
    ∀ s', 80 < s' ∧ s' < s → multiple_solutions s' :=
by sorry

end mary_score_l3194_319446


namespace linear_function_property_l3194_319405

/-- A linear function is a function of the form f(x) = mx + b for some constants m and b -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) 
  (hlinear : LinearFunction g) (hcond : g 4 - g 1 = 9) : 
  g 10 - g 1 = 27 := by
  sorry

end linear_function_property_l3194_319405


namespace simplify_expression_l3194_319463

theorem simplify_expression (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end simplify_expression_l3194_319463


namespace parallelepiped_has_twelve_edges_l3194_319412

/-- A parallelepiped is a three-dimensional figure formed by six parallelograms. -/
structure Parallelepiped where
  faces : Fin 6 → Parallelogram
  -- Additional properties ensuring the faces form a valid parallelepiped could be added here

/-- The number of edges in a geometric figure. -/
def numEdges (figure : Type) : ℕ := sorry

/-- Theorem stating that a parallelepiped has 12 edges. -/
theorem parallelepiped_has_twelve_edges (P : Parallelepiped) : numEdges Parallelepiped = 12 := by
  sorry

end parallelepiped_has_twelve_edges_l3194_319412


namespace polynomial_four_positive_roots_l3194_319441

/-- A polynomial with four positive real roots -/
def P (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 - 8*a * x^3 + b * x^2 - 32*c * x + 16*c

/-- The theorem stating the conditions for the polynomial to have four positive real roots -/
theorem polynomial_four_positive_roots :
  ∀ (a : ℝ), a ≠ 0 →
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
    ∀ (x : ℝ), P a (16*a) a x = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) ∧
  (∀ (b c : ℝ), 
    (∃ (x₁ x₂ x₃ x₄ : ℝ), 
      x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
      ∀ (x : ℝ), P a b c x = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) →
    b = 16*a ∧ c = a) := by
  sorry


end polynomial_four_positive_roots_l3194_319441


namespace games_to_give_away_l3194_319437

def initial_games : ℕ := 50
def desired_games : ℕ := 35

theorem games_to_give_away :
  initial_games - desired_games = 15 :=
by sorry

end games_to_give_away_l3194_319437


namespace solve_system_l3194_319465

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end solve_system_l3194_319465


namespace water_fountain_problem_l3194_319472

/-- The number of men needed to build a water fountain of a given length in a given number of days -/
def men_needed (length : ℝ) (days : ℝ) : ℝ :=
  sorry

theorem water_fountain_problem :
  let first_length : ℝ := 56
  let first_days : ℝ := 21
  let second_length : ℝ := 14
  let second_days : ℝ := 3
  let second_men : ℝ := 35

  (men_needed first_length first_days) = 20 :=
by
  sorry

end water_fountain_problem_l3194_319472


namespace gcf_lcm_sum_36_56_84_l3194_319434

theorem gcf_lcm_sum_36_56_84 : 
  let a := 36
  let b := 56
  let c := 84
  Nat.gcd a (Nat.gcd b c) + Nat.lcm a (Nat.lcm b c) = 516 := by
sorry

end gcf_lcm_sum_36_56_84_l3194_319434


namespace shelves_per_closet_l3194_319493

/-- Given the following constraints for stacking cans in a closet:
  * 12 cans fit in one row
  * 4 rows fit on one shelf
  * 480 cans can be stored in one closet
  Prove that 10 shelves can fit in one closet -/
theorem shelves_per_closet (cans_per_row : ℕ) (rows_per_shelf : ℕ) (cans_per_closet : ℕ)
  (h1 : cans_per_row = 12)
  (h2 : rows_per_shelf = 4)
  (h3 : cans_per_closet = 480) :
  cans_per_closet / (cans_per_row * rows_per_shelf) = 10 := by
  sorry

end shelves_per_closet_l3194_319493


namespace price_reduction_l3194_319403

theorem price_reduction (x : ℝ) : 
  (1 - x / 100) * (1 - 25 / 100) = 1 - 77.5 / 100 → x = 70 := by
  sorry

end price_reduction_l3194_319403


namespace seashell_collection_count_l3194_319485

theorem seashell_collection_count (initial_count additional_count : ℕ) :
  initial_count = 19 → additional_count = 6 →
  initial_count + additional_count = 25 :=
by sorry

end seashell_collection_count_l3194_319485


namespace min_value_at_three_l3194_319471

/-- The quadratic function we're minimizing -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- The statement that x = 3 minimizes the function f -/
theorem min_value_at_three :
  ∀ x : ℝ, f 3 ≤ f x :=
by
  sorry

#check min_value_at_three

end min_value_at_three_l3194_319471


namespace right_triangle_hypotenuse_l3194_319487

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by
  sorry

end right_triangle_hypotenuse_l3194_319487


namespace natural_number_pairs_l3194_319464

theorem natural_number_pairs : 
  ∀ a b : ℕ, 
    (90 < a + b ∧ a + b < 100) ∧ 
    (0.9 < (a : ℝ) / (b : ℝ) ∧ (a : ℝ) / (b : ℝ) < 0.91) → 
    ((a = 46 ∧ b = 51) ∨ (a = 47 ∧ b = 52)) :=
by sorry

end natural_number_pairs_l3194_319464


namespace plot_length_l3194_319476

/-- Proves that the length of a rectangular plot is 55 meters given the specified conditions -/
theorem plot_length (breadth : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) : 
  (breadth + 10 = breadth + 10) →  -- Length is 10 more than breadth
  (cost_per_meter = 26.5) →        -- Cost per meter is 26.50 rupees
  (total_cost = 5300) →            -- Total cost is 5300 rupees
  (4 * breadth + 20) * cost_per_meter = total_cost →  -- Perimeter calculation
  (breadth + 10 = 55) :=            -- Length of the plot is 55 meters
by sorry

end plot_length_l3194_319476


namespace range_of_m_l3194_319420

theorem range_of_m (x y m : ℝ) : 
  x > 0 → 
  y > 0 → 
  2 / x + 1 / y = 1 → 
  (∀ (x y : ℝ), x > 0 → y > 0 → 2 / x + 1 / y = 1 → x^2 + 2*x*y > m^2 + 2*m) → 
  m > -4 ∧ m < 2 := by
sorry

end range_of_m_l3194_319420


namespace min_value_of_function_min_value_achievable_l3194_319415

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x - 4 + 9 / (x + 1) ≥ 1 :=
sorry

theorem min_value_achievable :
  ∃ x : ℝ, x > -1 ∧ x - 4 + 9 / (x + 1) = 1 :=
sorry

end min_value_of_function_min_value_achievable_l3194_319415


namespace maria_water_bottles_l3194_319401

/-- The number of bottles Maria initially had -/
def initial_bottles : ℕ := 14

/-- The number of bottles Maria drank -/
def bottles_drunk : ℕ := 8

/-- The number of bottles Maria bought -/
def bottles_bought : ℕ := 45

/-- The final number of bottles Maria has -/
def final_bottles : ℕ := 51

theorem maria_water_bottles : 
  initial_bottles - bottles_drunk + bottles_bought = final_bottles :=
by sorry

end maria_water_bottles_l3194_319401


namespace james_chore_time_l3194_319426

/-- The total time James spends on all chores -/
def total_chore_time (vacuum_time cleaning_time laundry_time organizing_time : ℝ) : ℝ :=
  vacuum_time + cleaning_time + laundry_time + organizing_time

/-- Theorem stating the total time James spends on chores -/
theorem james_chore_time :
  ∃ (vacuum_time cleaning_time laundry_time organizing_time : ℝ),
    vacuum_time = 3 ∧
    cleaning_time = 3 * vacuum_time ∧
    laundry_time = (1/2) * cleaning_time ∧
    organizing_time = 2 * (vacuum_time + cleaning_time + laundry_time) ∧
    total_chore_time vacuum_time cleaning_time laundry_time organizing_time = 49.5 :=
by
  sorry

end james_chore_time_l3194_319426
