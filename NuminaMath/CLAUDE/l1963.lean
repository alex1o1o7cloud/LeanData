import Mathlib

namespace flour_to_add_correct_l1963_196317

/-- Represents the recipe and baking constraints -/
structure BakingProblem where
  total_flour : ℝ  -- Total flour required by the recipe
  total_sugar : ℝ  -- Total sugar required by the recipe
  flour_sugar_diff : ℝ  -- Difference between remaining flour and sugar to be added

/-- Calculates the amount of flour that needs to be added -/
def flour_to_add (problem : BakingProblem) : ℝ :=
  problem.total_flour

/-- Theorem stating that the amount of flour to add is correct -/
theorem flour_to_add_correct (problem : BakingProblem) 
  (h1 : problem.total_flour = 6)
  (h2 : problem.total_sugar = 13)
  (h3 : problem.flour_sugar_diff = 8) :
  flour_to_add problem = 6 ∧ 
  flour_to_add problem = problem.total_sugar - problem.flour_sugar_diff + problem.flour_sugar_diff := by
  sorry

#eval flour_to_add { total_flour := 6, total_sugar := 13, flour_sugar_diff := 8 }

end flour_to_add_correct_l1963_196317


namespace coin_flip_probability_l1963_196372

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the set of coins being flipped -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (half_dollar : CoinOutcome)

/-- The total number of possible outcomes when flipping 5 coins -/
def total_outcomes : ℕ := 32

/-- Predicate for the desired outcome (penny, nickel, and dime are heads) -/
def desired_outcome (cs : CoinSet) : Prop :=
  cs.penny = CoinOutcome.Heads ∧ cs.nickel = CoinOutcome.Heads ∧ cs.dime = CoinOutcome.Heads

/-- The number of outcomes satisfying the desired condition -/
def successful_outcomes : ℕ := 4

/-- The probability of the desired outcome -/
def probability : ℚ := 1 / 8

theorem coin_flip_probability :
  (successful_outcomes : ℚ) / total_outcomes = probability :=
sorry

end coin_flip_probability_l1963_196372


namespace solve_for_S_l1963_196350

theorem solve_for_S : ∃ S : ℚ, (1/2 : ℚ) * (1/7 : ℚ) * S = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ S = 70 := by
  sorry

end solve_for_S_l1963_196350


namespace max_difference_of_reversed_digits_l1963_196368

/-- Represents a three-digit positive integer -/
structure ThreeDigitInt where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value ≤ 999

/-- Returns true if two ThreeDigitInt have the same digits in reverse order -/
def reverse_digits (a b : ThreeDigitInt) : Prop :=
  ∃ (x y z : ℕ), x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧
    a.value = 100 * x + 10 * y + z ∧
    b.value = 100 * z + 10 * y + x

/-- The theorem to be proved -/
theorem max_difference_of_reversed_digits (q r : ThreeDigitInt) :
  reverse_digits q r →
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (q.value - r.value)) →
  q.value - r.value < 300 →
  (∀ s t : ThreeDigitInt, reverse_digits s t →
    (∃ p : ℕ, Nat.Prime p ∧ p ∣ (s.value - t.value)) →
    s.value - t.value < 300 →
    s.value - t.value ≤ q.value - r.value) →
  q.value - r.value = 297 := by
sorry

end max_difference_of_reversed_digits_l1963_196368


namespace log3_20_approximation_l1963_196375

-- Define the approximate values given in the problem
def log10_2_approx : ℝ := 0.301
def log10_3_approx : ℝ := 0.477

-- Define the target approximation
def log3_20_target : ℝ := 2.786

-- State the theorem
theorem log3_20_approximation :
  abs (Real.log 20 / Real.log 3 - log3_20_target) < 0.001 := by
  sorry

end log3_20_approximation_l1963_196375


namespace perpendicular_planes_from_lines_l1963_196320

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel_lines m n → 
  parallel_line_plane n β → 
  perpendicular_planes α β :=
sorry

end perpendicular_planes_from_lines_l1963_196320


namespace gcf_64_80_l1963_196333

theorem gcf_64_80 : Nat.gcd 64 80 = 16 := by
  sorry

end gcf_64_80_l1963_196333


namespace card_cost_correct_l1963_196397

/-- The cost of cards in the first box -/
def cost_box1 : ℝ := 1.25

/-- The cost of cards in the second box -/
def cost_box2 : ℝ := 1.75

/-- The number of cards bought from each box -/
def cards_per_box : ℕ := 6

/-- The total amount spent -/
def total_spent : ℝ := 18

/-- Theorem stating that the cost of cards in the first box is correct -/
theorem card_cost_correct : 
  cost_box1 * cards_per_box + cost_box2 * cards_per_box = total_spent := by
  sorry

end card_cost_correct_l1963_196397


namespace mary_juan_income_ratio_l1963_196300

theorem mary_juan_income_ratio (juan tim mary : ℝ) 
  (h1 : mary = 1.4 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mary = 0.84 * juan := by
  sorry

end mary_juan_income_ratio_l1963_196300


namespace sqrt_product_equality_l1963_196382

theorem sqrt_product_equality : Real.sqrt (49 + 121) * Real.sqrt (64 - 49) = Real.sqrt 2550 := by
  sorry

end sqrt_product_equality_l1963_196382


namespace trigonometric_identities_and_circle_parametrization_l1963_196377

theorem trigonometric_identities_and_circle_parametrization (a t : ℝ) 
  (h : t = Real.tan (a / 2)) : 
  Real.cos a = (1 - t^2) / (1 + t^2) ∧ 
  Real.sin a = 2 * t / (1 + t^2) ∧ 
  Real.tan a = 2 * t / (1 - t^2) ∧ 
  ∀ x y : ℝ, x = (1 - t^2) / (1 + t^2) ∧ y = 2 * t / (1 + t^2) → x^2 + y^2 = 1 := by
  sorry

end trigonometric_identities_and_circle_parametrization_l1963_196377


namespace stratified_sampling_most_appropriate_l1963_196337

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a population with two equal-sized subgroups -/
structure Population where
  total : ℕ
  group1 : ℕ
  group2 : ℕ
  h1 : group1 + group2 = total
  h2 : group1 = group2

/-- Represents a sample drawn from a population -/
structure Sample where
  size : ℕ
  population : Population
  method : SamplingMethod

/-- Predicate to determine if a sampling method is appropriate for comparing subgroups -/
def is_appropriate_for_subgroup_comparison (s : Sample) : Prop :=
  s.method = SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the most appropriate method
    for comparing characteristics between two equal-sized subgroups -/
theorem stratified_sampling_most_appropriate
  (pop : Population)
  (sample_size : ℕ)
  (h_sample_size : sample_size > 0 ∧ sample_size < pop.total) :
  ∀ (s : Sample),
    s.population = pop →
    s.size = sample_size →
    is_appropriate_for_subgroup_comparison s ↔ s.method = SamplingMethod.Stratified :=
sorry

end stratified_sampling_most_appropriate_l1963_196337


namespace number_problem_l1963_196365

theorem number_problem (x : ℝ) : x = 456 ↔ 0.5 * x = 0.4 * 120 + 180 := by
  sorry

end number_problem_l1963_196365


namespace burrito_cheese_amount_l1963_196391

/-- The amount of cheese (in ounces) required for a burrito -/
def cheese_per_burrito : ℝ := 4

/-- The amount of cheese (in ounces) required for a taco -/
def cheese_per_taco : ℝ := 9

/-- The total amount of cheese (in ounces) required for 7 burritos and 1 taco -/
def total_cheese : ℝ := 37

/-- Theorem stating that the amount of cheese required for a burrito is 4 ounces -/
theorem burrito_cheese_amount :
  cheese_per_burrito = 4 ∧
  cheese_per_taco = 9 ∧
  7 * cheese_per_burrito + cheese_per_taco = total_cheese :=
by sorry

end burrito_cheese_amount_l1963_196391


namespace pascal_triangle_30_rows_l1963_196380

/-- The number of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_elements (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

/-- Theorem: The number of elements in the first 30 rows of Pascal's Triangle is 465 -/
theorem pascal_triangle_30_rows : pascal_triangle_elements 29 = 465 := by
  sorry

end pascal_triangle_30_rows_l1963_196380


namespace plank_length_l1963_196349

/-- The length of a plank given specific movements of its ends -/
theorem plank_length (a b : ℝ) : 
  (∀ x y, x^2 + y^2 = a^2 + b^2 → (x - 8)^2 + (y + 4)^2 = a^2 + b^2) →
  (∀ x y, x^2 + y^2 = a^2 + b^2 → (x - 17)^2 + (y + 7)^2 = a^2 + b^2) →
  a^2 + b^2 = 65^2 := by
  sorry

end plank_length_l1963_196349


namespace sum_to_term_ratio_l1963_196318

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  h1 : a 5 - a 3 = 12
  h2 : a 6 - a 4 = 24

/-- The sum of the first n terms of a geometric sequence -/
def sum_n (seq : GeometricSequence) (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating the ratio of sum to nth term -/
theorem sum_to_term_ratio (seq : GeometricSequence) (n : ℕ) :
  sum_n seq n / seq.a n = 2 - 2^(1 - n) :=
sorry

end sum_to_term_ratio_l1963_196318


namespace cone_lateral_area_l1963_196345

/-- The lateral area of a cone with slant height 8 cm and base diameter 6 cm is 24π cm² -/
theorem cone_lateral_area (slant_height : ℝ) (base_diameter : ℝ) :
  slant_height = 8 →
  base_diameter = 6 →
  (1 / 2 : ℝ) * π * base_diameter * slant_height = 24 * π := by
  sorry

end cone_lateral_area_l1963_196345


namespace P_neither_sufficient_nor_necessary_for_Q_l1963_196396

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℝ := x^2 + (a-2)*x + 2*a - 8

-- Define the condition P
def condition_P (a : ℝ) : Prop := -1 < a ∧ a < 1

-- Define the condition Q
def condition_Q (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ < 0 ∧
    quadratic_equation a x₁ = 0 ∧ quadratic_equation a x₂ = 0

-- Theorem stating that P is neither sufficient nor necessary for Q
theorem P_neither_sufficient_nor_necessary_for_Q :
  (¬∀ a : ℝ, condition_P a → condition_Q a) ∧
  (¬∀ a : ℝ, condition_Q a → condition_P a) :=
sorry

end P_neither_sufficient_nor_necessary_for_Q_l1963_196396


namespace binary_is_largest_l1963_196381

/-- Convert a number from base b to decimal --/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

/-- The given numbers in their respective bases --/
def binary : List Nat := [1, 1, 0, 1, 1]
def base_4 : List Nat := [3, 0, 1]
def base_5 : List Nat := [4, 4]
def decimal : Nat := 25

/-- Theorem stating that the binary number is the largest --/
theorem binary_is_largest :
  let a := to_decimal binary 2
  let b := to_decimal base_4 4
  let c := to_decimal base_5 5
  let d := decimal
  a > b ∧ a > c ∧ a > d :=
by sorry


end binary_is_largest_l1963_196381


namespace vojta_sum_problem_l1963_196371

theorem vojta_sum_problem (S A B C : ℕ) : 
  S + 10 * B + C = 2224 →
  S + 10 * A + B = 2198 →
  S + 10 * A + C = 2204 →
  A < 10 →
  B < 10 →
  C < 10 →
  S + 100 * A + 10 * B + C = 2324 :=
by sorry

end vojta_sum_problem_l1963_196371


namespace asterisk_replacement_l1963_196363

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 18) * (x / 162) = 1 := by
  sorry

end asterisk_replacement_l1963_196363


namespace power_function_not_through_origin_l1963_196311

theorem power_function_not_through_origin (n : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (n^2 - 3*n + 3) * x^(n^2 - n - 2) ≠ 0) →
  n = 1 ∨ n = 2 := by
sorry

end power_function_not_through_origin_l1963_196311


namespace f_properties_l1963_196351

def f (b c x : ℝ) : ℝ := x * abs x + b * x + c

theorem f_properties (b c : ℝ) :
  (∀ x, c = 0 → f b c (-x) = -f b c x) ∧
  (∀ x y, b = 0 → x < y → f b c x < f b c y) ∧
  (∀ x, f b c x - c = -(f b c (-x) - c)) ∧
  ¬(∀ b c, ∃ x y, f b c x = 0 ∧ f b c y = 0 ∧ ∀ z, f b c z = 0 → z = x ∨ z = y) :=
by sorry

end f_properties_l1963_196351


namespace class_size_from_marking_error_l1963_196370

/-- The number of pupils in a class where a marking error occurred -/
def number_of_pupils : ℕ := 16

/-- The incorrect mark entered for a pupil -/
def incorrect_mark : ℕ := 73

/-- The correct mark for the pupil -/
def correct_mark : ℕ := 65

/-- The increase in class average due to the error -/
def average_increase : ℚ := 1/2

theorem class_size_from_marking_error :
  (incorrect_mark - correct_mark : ℚ) = number_of_pupils * average_increase :=
sorry

end class_size_from_marking_error_l1963_196370


namespace voice_of_china_sampling_l1963_196357

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (firstSample : ℕ) (n : ℕ) : ℕ :=
  firstSample + (populationSize / sampleSize) * (n - 1)

/-- The Voice of China sampling theorem -/
theorem voice_of_china_sampling :
  let populationSize := 500
  let sampleSize := 20
  let firstSample := 3
  let fifthSample := 5
  systematicSample populationSize sampleSize firstSample fifthSample = 103 := by
sorry

end voice_of_china_sampling_l1963_196357


namespace physics_marks_l1963_196325

theorem physics_marks (P C M : ℝ) 
  (avg_all : (P + C + M) / 3 = 70)
  (avg_pm : (P + M) / 2 = 90)
  (avg_pc : (P + C) / 2 = 70) :
  P = 110 := by sorry

end physics_marks_l1963_196325


namespace family_c_members_l1963_196310

/-- Represents the number of members in each family in Indira Nagar --/
structure FamilyMembers where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat

/-- The initial number of family members before some left for the hostel --/
def initial_members : FamilyMembers := {
  a := 7,
  b := 8,
  c := 10,  -- This is what we want to prove
  d := 13,
  e := 6,
  f := 10
}

/-- The number of family members after one member from each family left for the hostel --/
def members_after_hostel (fm : FamilyMembers) : FamilyMembers :=
  { a := fm.a - 1,
    b := fm.b - 1,
    c := fm.c - 1,
    d := fm.d - 1,
    e := fm.e - 1,
    f := fm.f - 1 }

/-- The total number of families --/
def num_families : Nat := 6

/-- Theorem stating that the initial number of members in family c was 10 --/
theorem family_c_members :
  (members_after_hostel initial_members).a +
  (members_after_hostel initial_members).b +
  (members_after_hostel initial_members).c +
  (members_after_hostel initial_members).d +
  (members_after_hostel initial_members).e +
  (members_after_hostel initial_members).f =
  8 * num_families :=
by sorry

end family_c_members_l1963_196310


namespace square_area_from_diagonal_l1963_196342

theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) :
  diagonal = 16 →
  area = diagonal^2 / 2 →
  area = 128 :=
by sorry

end square_area_from_diagonal_l1963_196342


namespace inscribed_circle_radius_bound_l1963_196374

/-- A convex polygon with area, perimeter, and inscribed circle radius -/
structure ConvexPolygon where
  area : ℝ
  perimeter : ℝ
  inscribed_radius : ℝ
  area_pos : 0 < area
  perimeter_pos : 0 < perimeter
  inscribed_radius_pos : 0 < inscribed_radius

/-- The theorem stating that for any convex polygon, the ratio of its area to its perimeter
    is less than or equal to the radius of its inscribed circle -/
theorem inscribed_circle_radius_bound (poly : ConvexPolygon) :
  poly.area / poly.perimeter ≤ poly.inscribed_radius :=
sorry

end inscribed_circle_radius_bound_l1963_196374


namespace total_blisters_l1963_196353

/-- Given a person with 60 blisters on each arm and 80 blisters on the rest of their body,
    the total number of blisters is 200. -/
theorem total_blisters (blisters_per_arm : ℕ) (blisters_rest : ℕ) :
  blisters_per_arm = 60 →
  blisters_rest = 80 →
  blisters_per_arm * 2 + blisters_rest = 200 :=
by sorry

end total_blisters_l1963_196353


namespace smallest_valid_debt_proof_l1963_196392

/-- The value of one sheep in dollars -/
def sheep_value : ℕ := 250

/-- The value of one lamb in dollars -/
def lamb_value : ℕ := 150

/-- A debt resolution is valid if it can be expressed as an integer combination of sheep and lambs -/
def is_valid_debt (d : ℕ) : Prop :=
  ∃ (s l : ℤ), d = sheep_value * s + lamb_value * l

/-- The smallest positive debt that can be resolved -/
def smallest_valid_debt : ℕ := 50

theorem smallest_valid_debt_proof :
  (∀ d : ℕ, d > 0 ∧ d < smallest_valid_debt → ¬is_valid_debt d) ∧
  is_valid_debt smallest_valid_debt :=
sorry

end smallest_valid_debt_proof_l1963_196392


namespace remainder_problem_l1963_196348

theorem remainder_problem (k : ℕ+) (h : ∃ b : ℕ, 120 = b * k^2 + 12) : 
  ∃ q : ℕ, 200 = q * k + 2 := by
sorry

end remainder_problem_l1963_196348


namespace tree_house_wood_needed_l1963_196394

-- Define the components of the tree house
structure TreeHouse where
  pillar_short : ℝ
  pillar_long : ℝ
  wall_short : ℝ
  wall_long : ℝ
  floor_avg : ℝ
  roof_first : ℝ
  roof_diff : ℝ

-- Define the function to calculate total wood needed
def total_wood (t : TreeHouse) : ℝ :=
  -- Pillars
  4 * t.pillar_short + 4 * t.pillar_long +
  -- Walls
  10 * t.wall_short + 10 * t.wall_long +
  -- Floor
  8 * t.floor_avg +
  -- Roof (arithmetic sequence sum formula)
  6 * t.roof_first + 15 * t.roof_diff

-- Theorem statement
theorem tree_house_wood_needed (t : TreeHouse) 
  (h1 : t.pillar_short = 4)
  (h2 : t.pillar_long = 5 * Real.sqrt t.pillar_short)
  (h3 : t.wall_short = 6)
  (h4 : t.wall_long = (2/3) * (t.wall_short ^ (3/2)))
  (h5 : t.floor_avg = 5.5)
  (h6 : t.roof_first = 2 * t.floor_avg)
  (h7 : t.roof_diff = (1/3) * t.pillar_short) :
  total_wood t = 344 := by
  sorry

end tree_house_wood_needed_l1963_196394


namespace multiple_of_n_divisible_by_60_l1963_196399

theorem multiple_of_n_divisible_by_60 (n : ℕ) :
  0 < n →
  n < 200 →
  (∃ k : ℕ, k > 0 ∧ 60 ∣ (k * n)) →
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ n = p * q * r) →
  (∃ m : ℕ, m > 0 ∧ 60 ∣ (m * n) ∧ ∀ k : ℕ, (k > 0 ∧ 60 ∣ (k * n)) → m ≤ k) →
  (∃ m : ℕ, m > 0 ∧ 60 ∣ (m * n) ∧ ∀ k : ℕ, (k > 0 ∧ 60 ∣ (k * n)) → m ≤ k) ∧ m = 60 :=
by sorry

end multiple_of_n_divisible_by_60_l1963_196399


namespace count_distinct_tetrahedrons_is_423_l1963_196307

/-- Represents a regular tetrahedron with its vertices and edge midpoints -/
structure RegularTetrahedron :=
  (vertices : Finset (Fin 4))
  (edge_midpoints : Finset (Fin 6))

/-- Represents a new tetrahedron formed from points of a regular tetrahedron -/
def NewTetrahedron (t : RegularTetrahedron) := Finset (Fin 4)

/-- Counts the number of distinct new tetrahedrons that can be formed -/
def count_distinct_tetrahedrons (t : RegularTetrahedron) : ℕ :=
  sorry

/-- The main theorem stating that the number of distinct tetrahedrons is 423 -/
theorem count_distinct_tetrahedrons_is_423 (t : RegularTetrahedron) :
  count_distinct_tetrahedrons t = 423 :=
sorry

end count_distinct_tetrahedrons_is_423_l1963_196307


namespace books_for_vacation_l1963_196389

/-- The number of books that can be read given reading speed, book parameters, and reading time -/
def books_to_read (reading_speed : ℕ) (words_per_page : ℕ) (pages_per_book : ℕ) (reading_time : ℕ) : ℕ :=
  (reading_speed * reading_time * 60) / (words_per_page * pages_per_book)

/-- Theorem stating that given the specific conditions, the number of books to read is 6 -/
theorem books_for_vacation : books_to_read 40 100 80 20 = 6 := by
  sorry

end books_for_vacation_l1963_196389


namespace pizza_consumption_order_l1963_196376

/-- Represents the fraction of pizza eaten by each sibling -/
structure PizzaConsumption where
  alex : Rat
  beth : Rat
  cyril : Rat
  dan : Rat

/-- Compares two rational numbers -/
def ratGreater (a b : Rat) : Prop := a > b

theorem pizza_consumption_order (pc : PizzaConsumption) : 
  pc.alex = 1/7 ∧ 
  pc.beth = 2/5 ∧ 
  pc.cyril = 3/10 ∧ 
  pc.dan = 2 * (1 - (pc.alex + pc.beth + pc.cyril)) →
  ratGreater pc.beth pc.dan ∧ 
  ratGreater pc.dan pc.cyril ∧ 
  ratGreater pc.cyril pc.alex :=
by sorry

end pizza_consumption_order_l1963_196376


namespace unique_q_13_l1963_196384

-- Define the cubic polynomial q(x)
def q (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem unique_q_13 (a b c d : ℝ) :
  (∀ x : ℝ, (q a b c d x)^3 - x = 0 → x = 2 ∨ x = -2 ∨ x = 5) →
  q a b c d 2 = 2 →
  q a b c d (-2) = -2 →
  q a b c d 5 = 3 →
  ∃! y : ℝ, q a b c d 13 = y :=
sorry

end unique_q_13_l1963_196384


namespace consecutive_integers_divisibility_l1963_196339

theorem consecutive_integers_divisibility (k : ℤ) : 
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 11 * m) →
  (∃ m : ℤ, n = 6 * m) ∧
  (∃ m : ℤ, n = 22 * m) ∧
  (∃ m : ℤ, n = 33 * m) ∧
  (∃ m : ℤ, n = 66 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 36 * m) :=
by sorry

end consecutive_integers_divisibility_l1963_196339


namespace rational_function_identity_l1963_196362

def is_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n

theorem rational_function_identity 
  (f : ℚ → ℚ) 
  (h1 : ∃ a : ℚ, ¬is_integer (f a))
  (h2 : ∀ x y : ℚ, is_integer (f (x + y) - f x - f y))
  (h3 : ∀ x y : ℚ, is_integer (f (x * y) - f x * f y)) :
  ∀ x : ℚ, f x = x :=
sorry

end rational_function_identity_l1963_196362


namespace expression_value_l1963_196314

theorem expression_value (a b c d : ℤ) 
  (ha : a = 15) (hb : b = 19) (hc : c = 3) (hd : d = 2) : 
  (a - (b - c)) - ((a - b) - c + d) = 4 := by
  sorry

end expression_value_l1963_196314


namespace square_area_calculation_l1963_196356

theorem square_area_calculation (s : ℝ) (r : ℝ) (l : ℝ) (b : ℝ) : 
  r = s →                -- radius of circle equals side of square
  l = (1 / 6) * r →      -- length of rectangle is one-sixth of circle radius
  l * b = 360 →          -- area of rectangle is 360 sq. units
  b = 10 →               -- breadth of rectangle is 10 units
  s^2 = 46656 :=         -- area of square is 46656 sq. units
by
  sorry

end square_area_calculation_l1963_196356


namespace mysterious_division_l1963_196301

theorem mysterious_division :
  ∃! (d q : ℕ),
    d ∈ Finset.range 900 ∧ d ≥ 100 ∧
    q ∈ Finset.range 90000 ∧ q ≥ 10000 ∧
    10000000 = d * q + (10000000 % d) ∧
    d = 124 ∧ q = 80809 := by
  sorry

end mysterious_division_l1963_196301


namespace reciprocal_of_25_l1963_196387

theorem reciprocal_of_25 (x : ℝ) : (1 / x = 25) → (x = 1 / 25) := by
  sorry

end reciprocal_of_25_l1963_196387


namespace divisibility_statement_l1963_196338

theorem divisibility_statement (a : ℤ) :
  (∃! n : Fin 4, ¬ (
    (n = 0 → a % 2 = 0) ∧
    (n = 1 → a % 4 = 0) ∧
    (n = 2 → a % 12 = 0) ∧
    (n = 3 → a % 24 = 0)
  )) →
  ¬(a % 24 = 0) :=
by sorry


end divisibility_statement_l1963_196338


namespace range_of_x_l1963_196378

def is_meaningful (x : ℝ) : Prop := x ≠ 5

theorem range_of_x : ∀ x : ℝ, is_meaningful x ↔ x ≠ 5 := by sorry

end range_of_x_l1963_196378


namespace pages_read_difference_l1963_196364

theorem pages_read_difference (total_pages : ℕ) (fraction_read : ℚ) : 
  total_pages = 90 → fraction_read = 2/3 → 
  (total_pages : ℚ) * fraction_read - (total_pages : ℚ) * (1 - fraction_read) = 30 := by
  sorry

end pages_read_difference_l1963_196364


namespace total_distance_after_five_days_l1963_196398

/-- The total distance run by Peter and Andrew after 5 days -/
def total_distance (andrew_distance : ℕ) (peter_extra : ℕ) (days : ℕ) : ℕ :=
  (andrew_distance + peter_extra + andrew_distance) * days

/-- Theorem stating the total distance run by Peter and Andrew after 5 days -/
theorem total_distance_after_five_days :
  total_distance 2 3 5 = 35 := by
  sorry

end total_distance_after_five_days_l1963_196398


namespace consecutive_integers_sum_l1963_196336

theorem consecutive_integers_sum (x : ℤ) :
  (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 75 →
  (x - 2) + (x + 2) = 30 := by
  sorry

end consecutive_integers_sum_l1963_196336


namespace tyrah_sarah_pencil_ratio_l1963_196346

/-- Given that Tyrah has 12 pencils and Sarah has 2 pencils, 
    prove that the ratio of Tyrah's pencils to Sarah's pencils is 6. -/
theorem tyrah_sarah_pencil_ratio :
  ∀ (tyrah_pencils sarah_pencils : ℕ),
    tyrah_pencils = 12 →
    sarah_pencils = 2 →
    (tyrah_pencils : ℚ) / sarah_pencils = 6 := by
  sorry

end tyrah_sarah_pencil_ratio_l1963_196346


namespace negation_equivalence_l1963_196330

def original_statement (a b : ℤ) : Prop :=
  (¬(Odd a ∧ Odd b)) → Even (a + b)

def proposed_negation (a b : ℤ) : Prop :=
  (¬(Odd a ∧ Odd b)) → ¬Even (a + b)

def correct_negation (a b : ℤ) : Prop :=
  ¬(Odd a ∧ Odd b) ∧ ¬Even (a + b)

theorem negation_equivalence :
  ∀ a b : ℤ, ¬(original_statement a b) ↔ correct_negation a b :=
sorry

end negation_equivalence_l1963_196330


namespace square_root_equation_l1963_196361

theorem square_root_equation (x : ℝ) : Real.sqrt (x - 5) = 7 → x = 54 := by
  sorry

end square_root_equation_l1963_196361


namespace phoebes_servings_is_one_l1963_196313

/-- The number of servings per jar of peanut butter -/
def servings_per_jar : ℕ := 15

/-- The number of jars needed -/
def jars_needed : ℕ := 4

/-- The number of days the peanut butter should last -/
def days_to_last : ℕ := 30

/-- Phoebe's serving amount equals her dog's serving amount -/
axiom phoebe_dog_equal_servings : True

/-- The number of servings Phoebe eats each night -/
def phoebes_nightly_servings : ℚ :=
  (servings_per_jar * jars_needed : ℚ) / (2 * days_to_last)

theorem phoebes_servings_is_one :
  phoebes_nightly_servings = 1 := by sorry

end phoebes_servings_is_one_l1963_196313


namespace fiftieth_ring_l1963_196359

/-- Represents the number of squares in the nth ring -/
def S (n : ℕ) : ℕ := 10 * n - 2

/-- The properties of the sequence of rings -/
axiom first_ring : S 1 = 8
axiom second_ring : S 2 = 18
axiom ring_increase (n : ℕ) : n ≥ 2 → S (n + 1) - S n = 10

/-- The theorem stating the number of squares in the 50th ring -/
theorem fiftieth_ring : S 50 = 498 := by sorry

end fiftieth_ring_l1963_196359


namespace fundraising_problem_l1963_196332

/-- The fundraising problem -/
theorem fundraising_problem (total_goal : ℕ) (num_people : ℕ) (fee_per_person : ℕ) 
  (h1 : total_goal = 2400)
  (h2 : num_people = 8)
  (h3 : fee_per_person = 20) :
  (total_goal + num_people * fee_per_person) / num_people = 320 := by
  sorry

#check fundraising_problem

end fundraising_problem_l1963_196332


namespace rounded_number_problem_l1963_196309

theorem rounded_number_problem (x : ℝ) (n : ℤ) :
  x > 0 ∧ n = ⌈1.28 * x⌉ ∧ (n : ℝ) - 1 < x ∧ x ≤ (n : ℝ) →
  x = 25/32 ∨ x = 25/16 ∨ x = 75/32 ∨ x = 25/8 :=
by sorry

end rounded_number_problem_l1963_196309


namespace greatest_power_of_two_dividing_expression_l1963_196373

theorem greatest_power_of_two_dividing_expression : ∃ k : ℕ, 
  (k = 1007 ∧ 
   2^k ∣ (10^1004 - 4^502) ∧ 
   ∀ m : ℕ, 2^m ∣ (10^1004 - 4^502) → m ≤ k) := by
sorry

end greatest_power_of_two_dividing_expression_l1963_196373


namespace shaded_area_calculation_l1963_196326

theorem shaded_area_calculation (π : ℝ) (h : π > 0) : 
  let square_side : ℝ := 8
  let quarter_circle_radius : ℝ := 0.6 * square_side
  let square_area : ℝ := square_side ^ 2
  let quarter_circles_area : ℝ := π * quarter_circle_radius ^ 2
  square_area - quarter_circles_area = 64 - 23.04 * π := by
  sorry

end shaded_area_calculation_l1963_196326


namespace largest_negative_integer_congruence_l1963_196354

theorem largest_negative_integer_congruence :
  ∃ (x : ℤ), x = -2 ∧ 
  (∀ (y : ℤ), y < 0 → 50 * y + 14 ≡ 10 [ZMOD 24] → y ≤ x) ∧
  50 * x + 14 ≡ 10 [ZMOD 24] := by
  sorry

end largest_negative_integer_congruence_l1963_196354


namespace spaatz_frankie_relation_binkie_frankie_relation_binkie_has_24_gemstones_l1963_196312

/-- The number of gemstones on Spaatz's collar -/
def spaatz_gemstones : ℕ := 1

/-- The number of gemstones on Frankie's collar -/
def frankie_gemstones : ℕ := 6

/-- The relationship between Spaatz's and Frankie's gemstones -/
theorem spaatz_frankie_relation : spaatz_gemstones = frankie_gemstones / 2 - 2 := by sorry

/-- The relationship between Binkie's and Frankie's gemstones -/
theorem binkie_frankie_relation : ∃ (binkie_gemstones : ℕ), binkie_gemstones = 4 * frankie_gemstones := by sorry

/-- The main theorem: Binkie has 24 gemstones -/
theorem binkie_has_24_gemstones : ∃ (binkie_gemstones : ℕ), binkie_gemstones = 24 := by sorry

end spaatz_frankie_relation_binkie_frankie_relation_binkie_has_24_gemstones_l1963_196312


namespace quadratic_function_max_abs_value_ge_one_l1963_196335

/-- Given a quadratic function f(x) = 2x^2 + mx + n, 
    prove that the maximum absolute value of f(1), f(2), and f(3) is at least 1. -/
theorem quadratic_function_max_abs_value_ge_one (m n : ℝ) : 
  let f := fun (x : ℝ) => 2 * x^2 + m * x + n
  max (|f 1|) (max (|f 2|) (|f 3|)) ≥ 1 := by
  sorry

end quadratic_function_max_abs_value_ge_one_l1963_196335


namespace mistaken_calculation_l1963_196308

theorem mistaken_calculation (x : ℚ) : x - 20 = 52 → x / 4 = 18 := by
  sorry

end mistaken_calculation_l1963_196308


namespace solve_equation_l1963_196341

theorem solve_equation (x : ℚ) (h : x - 3*x + 5*x = 200) : x = 200/3 := by
  sorry

end solve_equation_l1963_196341


namespace quadratic_inequality_solution_l1963_196386

theorem quadratic_inequality_solution (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 2*a*x - 8*a^2 < 0 ↔ x₁ < x ∧ x < x₂) →
  a > 0 →
  x₂ + x₁ = 15 →
  a = 15/2 := by
sorry

end quadratic_inequality_solution_l1963_196386


namespace mary_sugar_addition_l1963_196360

/-- The amount of sugar Mary needs to add to her cake mix -/
def sugar_to_add (required_sugar : ℕ) (added_sugar : ℕ) : ℕ :=
  required_sugar - added_sugar

theorem mary_sugar_addition : sugar_to_add 11 10 = 1 := by
  sorry

end mary_sugar_addition_l1963_196360


namespace apple_sorting_probability_l1963_196355

/-- Ratio of large apples to small apples -/
def largeToSmallRatio : ℚ := 9/1

/-- Probability of sorting a large apple as a small apple -/
def largeSortedAsSmall : ℚ := 5/100

/-- Probability of sorting a small apple as a large apple -/
def smallSortedAsLarge : ℚ := 2/100

/-- The probability that a "large apple" selected after sorting is indeed a large apple -/
def probLargeGivenSortedLarge : ℚ := 855/857

theorem apple_sorting_probability :
  let totalApples : ℚ := 10
  let largeApples : ℚ := (largeToSmallRatio * totalApples) / (largeToSmallRatio + 1)
  let smallApples : ℚ := totalApples - largeApples
  let probLarge : ℚ := largeApples / totalApples
  let probSmall : ℚ := smallApples / totalApples
  let probLargeSortedLarge : ℚ := 1 - largeSortedAsSmall
  let probLargeAndSortedLarge : ℚ := probLarge * probLargeSortedLarge
  let probSmallAndSortedLarge : ℚ := probSmall * smallSortedAsLarge
  let probSortedLarge : ℚ := probLargeAndSortedLarge + probSmallAndSortedLarge
  probLargeGivenSortedLarge = probLargeAndSortedLarge / probSortedLarge :=
by sorry

end apple_sorting_probability_l1963_196355


namespace wall_volume_l1963_196352

/-- The volume of a rectangular wall with specific proportions -/
theorem wall_volume (width : ℝ) (height : ℝ) (length : ℝ) : 
  width = 4 → 
  height = 6 * width → 
  length = 7 * height → 
  width * height * length = 16128 := by
sorry

end wall_volume_l1963_196352


namespace mandy_med_school_acceptances_l1963_196331

theorem mandy_med_school_acceptances
  (total_researched : ℕ)
  (applied_fraction : ℚ)
  (accepted_fraction : ℚ)
  (h1 : total_researched = 42)
  (h2 : applied_fraction = 1 / 3)
  (h3 : accepted_fraction = 1 / 2)
  : ℕ :=
  by
    sorry

#check mandy_med_school_acceptances

end mandy_med_school_acceptances_l1963_196331


namespace arithmetic_geometric_sequence_log_l1963_196393

theorem arithmetic_geometric_sequence_log (a b : ℝ) : 
  a ≠ b →
  (2 * a = 1 + b) →
  (b ^ 2 = a) →
  7 * a * (Real.log (-b) / Real.log a) = 7/8 := by sorry

end arithmetic_geometric_sequence_log_l1963_196393


namespace range_of_a_l1963_196302

-- Define the * operation
def star (x y : ℝ) := x * (1 - y)

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, star x (x - a) > 0 → -1 ≤ x ∧ x ≤ 1) → 
  -2 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l1963_196302


namespace distribute_6_balls_3_boxes_l1963_196304

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distributeIndistinguishable (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 5 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_6_balls_3_boxes : distributeIndistinguishable 6 3 = 5 := by
  sorry

end distribute_6_balls_3_boxes_l1963_196304


namespace factorial_equation_solution_l1963_196344

theorem factorial_equation_solution (n : ℕ) : n * n! + n! = 5040 → n = 6 := by
  sorry

end factorial_equation_solution_l1963_196344


namespace circle_areas_and_square_l1963_196303

/-- Given two concentric circles with radii 23 and 33 units, prove that a third circle
    with area equal to the shaded area between the two original circles has a radius of 4√35,
    and when inscribed in a square, the square's side length is 8√35. -/
theorem circle_areas_and_square (r₁ r₂ r₃ : ℝ) (s : ℝ) : 
  r₁ = 23 →
  r₂ = 33 →
  π * r₃^2 = π * (r₂^2 - r₁^2) →
  s = 2 * r₃ →
  r₃ = 4 * Real.sqrt 35 ∧ s = 8 * Real.sqrt 35 := by
  sorry

end circle_areas_and_square_l1963_196303


namespace sum_squares_first_12_base6_l1963_196315

-- Define a function to convert a number from base 10 to base 6
def toBase6 (n : ℕ) : List ℕ := sorry

-- Define a function to square a number
def square (n : ℕ) : ℕ := n * n

-- Define a function to sum a list of numbers in base 6
def sumBase6 (list : List (List ℕ)) : List ℕ := sorry

-- Main theorem
theorem sum_squares_first_12_base6 : 
  sumBase6 (List.map (λ n => toBase6 (square n)) (List.range 12)) = [5, 1, 5, 0, 1] := by sorry

end sum_squares_first_12_base6_l1963_196315


namespace arithmetic_sequence_length_l1963_196306

/-- The number of terms in an arithmetic sequence from -3 to 53 -/
theorem arithmetic_sequence_length : ∀ (a d : ℤ), 
  a = -3 → 
  d = 4 → 
  ∃ n : ℕ, n > 0 ∧ a + (n - 1) * d = 53 → 
  n = 15 := by
sorry

end arithmetic_sequence_length_l1963_196306


namespace only_prime_alternating_base14_l1963_196323

/-- Represents a number in base 14 with alternating 1s and 0s -/
def alternating_base14 (n : ℕ) : ℕ :=
  (14^(2*n) - 1) / 195

/-- Checks if a number is prime -/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, 1 < m → m < p → ¬(p % m = 0)

theorem only_prime_alternating_base14 :
  ∀ n : ℕ, is_prime (alternating_base14 n) ↔ n = 1 :=
sorry

end only_prime_alternating_base14_l1963_196323


namespace appropriate_sampling_methods_l1963_196324

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

-- Define the scenarios
structure Scenario where
  totalPopulation : ℕ
  sampleSize : ℕ
  hasSubgroups : Bool
  isLargeScale : Bool

-- Define the function to determine the appropriate sampling method
def appropriateSamplingMethod (scenario : Scenario) : SamplingMethod :=
  if scenario.hasSubgroups then
    SamplingMethod.Stratified
  else if scenario.isLargeScale then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom

-- Define the three scenarios
def scenario1 : Scenario := ⟨60, 8, false, false⟩
def scenario2 : Scenario := ⟨0, 0, false, true⟩  -- We don't know exact numbers, but it's large scale
def scenario3 : Scenario := ⟨130, 13, true, false⟩

-- State the theorem
theorem appropriate_sampling_methods :
  (appropriateSamplingMethod scenario1 = SamplingMethod.SimpleRandom) ∧
  (appropriateSamplingMethod scenario2 = SamplingMethod.Systematic) ∧
  (appropriateSamplingMethod scenario3 = SamplingMethod.Stratified) :=
by sorry

end appropriate_sampling_methods_l1963_196324


namespace max_a_for_inequality_l1963_196390

theorem max_a_for_inequality : 
  (∃ (a_max : ℝ), 
    (∀ (a : ℝ), (∀ (x : ℝ), 1 - (2/3) * Real.cos (2*x) + a * Real.cos x ≥ 0) → a ≤ a_max) ∧ 
    (∀ (x : ℝ), 1 - (2/3) * Real.cos (2*x) + a_max * Real.cos x ≥ 0)) ∧
  (∀ (a_max : ℝ), 
    ((∀ (a : ℝ), (∀ (x : ℝ), 1 - (2/3) * Real.cos (2*x) + a * Real.cos x ≥ 0) → a ≤ a_max) ∧ 
    (∀ (x : ℝ), 1 - (2/3) * Real.cos (2*x) + a_max * Real.cos x ≥ 0)) → 
    a_max = 1/3) :=
sorry

end max_a_for_inequality_l1963_196390


namespace shopkeeper_sold_450_meters_l1963_196340

/-- Represents the sale of cloth by a shopkeeper -/
structure ClothSale where
  totalSellingPrice : ℕ  -- Total selling price in Rupees
  lossPerMeter : ℕ       -- Loss per meter in Rupees
  costPricePerMeter : ℕ  -- Cost price per meter in Rupees

/-- Calculates the number of meters of cloth sold -/
def metersOfClothSold (sale : ClothSale) : ℕ :=
  sale.totalSellingPrice / (sale.costPricePerMeter - sale.lossPerMeter)

/-- Theorem stating that the shopkeeper sold 450 meters of cloth -/
theorem shopkeeper_sold_450_meters :
  let sale : ClothSale := {
    totalSellingPrice := 18000,
    lossPerMeter := 5,
    costPricePerMeter := 45
  }
  metersOfClothSold sale = 450 := by
  sorry


end shopkeeper_sold_450_meters_l1963_196340


namespace darma_peanut_eating_l1963_196305

/-- Darma's peanut eating rate -/
def peanuts_per_15_seconds : ℕ := 20

/-- Convert minutes to seconds -/
def minutes_to_seconds (minutes : ℕ) : ℕ := minutes * 60

/-- Calculate peanuts eaten in a given time -/
def peanuts_eaten (seconds : ℕ) : ℕ :=
  (seconds / 15) * peanuts_per_15_seconds

theorem darma_peanut_eating (minutes : ℕ) (h : minutes = 6) :
  peanuts_eaten (minutes_to_seconds minutes) = 480 := by
  sorry

end darma_peanut_eating_l1963_196305


namespace lcm_gcf_ratio_l1963_196385

theorem lcm_gcf_ratio : (Nat.lcm 144 756) / (Nat.gcd 144 756) = 84 := by
  sorry

end lcm_gcf_ratio_l1963_196385


namespace system_two_solutions_l1963_196383

/-- The system of equations has exactly two solutions if and only if a ∈ {49, 289} -/
theorem system_two_solutions (a : ℝ) : 
  (∃! x y z w : ℝ, 
    (abs (y + x + 8) + abs (y - x + 8) = 16 ∧
     (abs x - 15)^2 + (abs y - 8)^2 = a) ∧
    (abs (z + w + 8) + abs (z - w + 8) = 16 ∧
     (abs w - 15)^2 + (abs z - 8)^2 = a) ∧
    (x ≠ w ∨ y ≠ z)) ↔ 
  (a = 49 ∨ a = 289) :=
sorry

end system_two_solutions_l1963_196383


namespace opposite_abs_difference_l1963_196328

theorem opposite_abs_difference (a : ℤ) : a = -3 → |a - 2| = 5 := by
  sorry

end opposite_abs_difference_l1963_196328


namespace exists_collatz_greater_than_2012x_l1963_196322

-- Define the Collatz function
def collatz (x : ℕ) : ℕ :=
  if x % 2 = 1 then 3 * x + 1 else x / 2

-- Define the iterated Collatz function
def collatz_iter : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => collatz_iter n (collatz x)

-- State the theorem
theorem exists_collatz_greater_than_2012x : ∃ x : ℕ, x > 0 ∧ collatz_iter 40 x > 2012 * x := by
  sorry

end exists_collatz_greater_than_2012x_l1963_196322


namespace necessary_but_not_sufficient_condition_l1963_196367

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a < b → a < b + 1) ∧ ¬(∀ a b : ℝ, a < b + 1 → a < b) := by
  sorry

end necessary_but_not_sufficient_condition_l1963_196367


namespace monster_consumption_l1963_196343

theorem monster_consumption (a : ℕ → ℕ) (h1 : a 0 = 121) (h2 : ∀ n, a (n + 1) = 2 * a n) : 
  a 0 + a 1 + a 2 = 847 := by
sorry

end monster_consumption_l1963_196343


namespace fraction_simplification_l1963_196334

theorem fraction_simplification (m : ℝ) (h : m ≠ 3) :
  m^2 / (m - 3) + 9 / (3 - m) = m + 3 := by
  sorry

end fraction_simplification_l1963_196334


namespace find_M_l1963_196388

theorem find_M (x y z M : ℝ) : 
  x + y + z = 120 ∧ 
  x - 10 = M ∧ 
  y + 10 = M ∧ 
  z / 10 = M 
  → M = 10 := by
sorry

end find_M_l1963_196388


namespace craft_supplies_ratio_l1963_196316

/-- Represents the craft supplies bought by a person -/
structure CraftSupplies :=
  (glueSticks : ℕ)
  (constructionPaper : ℕ)

/-- The ratio of two natural numbers -/
def ratio (a b : ℕ) : ℚ := a / b

theorem craft_supplies_ratio :
  ∀ (allison marie : CraftSupplies),
    allison.glueSticks = marie.glueSticks + 8 →
    marie.glueSticks = 15 →
    marie.constructionPaper = 30 →
    allison.glueSticks + allison.constructionPaper = 28 →
    ratio marie.constructionPaper allison.constructionPaper = 6 := by
  sorry

end craft_supplies_ratio_l1963_196316


namespace perpendicular_lines_to_plane_are_parallel_perpendicular_line_to_planes_are_parallel_l1963_196327

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  
/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here

/-- Perpendicular relation between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallel relation between two planes -/
def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem perpendicular_lines_to_plane_are_parallel (m n : Line3D) (β : Plane3D) :
  perpendicular m β → perpendicular n β → parallel_lines m n :=
sorry

theorem perpendicular_line_to_planes_are_parallel (m : Line3D) (α β : Plane3D) :
  perpendicular m α → perpendicular m β → parallel_planes α β :=
sorry

end perpendicular_lines_to_plane_are_parallel_perpendicular_line_to_planes_are_parallel_l1963_196327


namespace modified_factor_tree_l1963_196358

theorem modified_factor_tree : 
  ∀ A B C D E : ℕ,
  A = B * C →
  B = 3 * D →
  C = 7 * E →
  D = 5 * 2 →
  E = 7 * 3 →
  A = 4410 := by
  sorry

end modified_factor_tree_l1963_196358


namespace min_cost_to_win_l1963_196369

/-- Represents the possible coin types -/
inductive Coin
| One : Coin
| Two : Coin

/-- The game state -/
structure GameState where
  points : Nat
  cost : Nat

/-- Applies a coin to the game state -/
def apply_coin (s : GameState) (c : Coin) : GameState :=
  match c with
  | Coin.One => { points := s.points + 1, cost := s.cost + 1 }
  | Coin.Two => { points := s.points * 2, cost := s.cost + 2 }

/-- Checks if a game state is valid (50 points or less) -/
def is_valid (s : GameState) : Prop := s.points ≤ 50

/-- Checks if a game state is winning (exactly 50 points) -/
def is_winning (s : GameState) : Prop := s.points = 50

/-- The theorem to prove -/
theorem min_cost_to_win : 
  ∃ (sequence : List Coin), 
    let final_state := sequence.foldl apply_coin { points := 0, cost := 0 }
    is_winning final_state ∧ 
    final_state.cost = 11 ∧ 
    (∀ (other_sequence : List Coin), 
      let other_final_state := other_sequence.foldl apply_coin { points := 0, cost := 0 }
      is_winning other_final_state → other_final_state.cost ≥ 11) :=
by sorry

end min_cost_to_win_l1963_196369


namespace sunscreen_discount_percentage_l1963_196379

/-- Calculate the discount percentage for Juanita's sunscreen purchase -/
theorem sunscreen_discount_percentage : 
  let bottles_per_year : ℕ := 12
  let cost_per_bottle : ℚ := 30
  let discounted_total_cost : ℚ := 252
  let original_total_cost : ℚ := bottles_per_year * cost_per_bottle
  let discount_amount : ℚ := original_total_cost - discounted_total_cost
  let discount_percentage : ℚ := (discount_amount / original_total_cost) * 100
  discount_percentage = 30 := by sorry

end sunscreen_discount_percentage_l1963_196379


namespace max_distance_circle_to_line_l1963_196395

/-- The maximum distance from any point on the circle (x-1)² + (y-1)² = 2 to the line x + y - 4 = 0 is 2√2. -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 2}
  let line := {p : ℝ × ℝ | p.1 + p.2 - 4 = 0}
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧
    (∀ p ∈ circle, ∀ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ d) ∧
    (∃ p ∈ circle, ∃ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = d) :=
by sorry

end max_distance_circle_to_line_l1963_196395


namespace inequality_solution_l1963_196347

theorem inequality_solution (x : ℝ) : 
  (x^2 + x^3 - 2*x^4) / (x + x^2 - 2*x^3) ≥ -1 ↔ 
  (x ≥ -1 ∧ x ≠ -1/2 ∧ x ≠ 0 ∧ x ≠ 1) :=
by sorry

end inequality_solution_l1963_196347


namespace number_multiplied_by_9999_l1963_196329

theorem number_multiplied_by_9999 : ∃ x : ℕ, x * 9999 = 5865863355 ∧ x = 586650 := by
  sorry

end number_multiplied_by_9999_l1963_196329


namespace linda_sales_l1963_196321

/-- Calculates the total amount of money made from selling necklaces and rings -/
def total_money_made (num_necklaces : ℕ) (num_rings : ℕ) (cost_per_necklace : ℕ) (cost_per_ring : ℕ) : ℕ :=
  num_necklaces * cost_per_necklace + num_rings * cost_per_ring

/-- Theorem: The total money made from selling 4 necklaces at $12 each and 8 rings at $4 each is $80 -/
theorem linda_sales : total_money_made 4 8 12 4 = 80 := by
  sorry

end linda_sales_l1963_196321


namespace shaded_areas_equality_l1963_196366

theorem shaded_areas_equality (θ : Real) (h1 : 0 < θ) (h2 : θ < π / 4) :
  (∃ (r : Real), r > 0 ∧ θ * r^2 = (r^2 * Real.tan (2 * θ)) / 2) ↔ Real.tan (2 * θ) = 2 * θ := by
  sorry

end shaded_areas_equality_l1963_196366


namespace f_at_2_l1963_196319

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem f_at_2 : f 2 = 3 := by
  sorry

end f_at_2_l1963_196319
