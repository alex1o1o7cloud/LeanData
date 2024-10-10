import Mathlib

namespace polynomial_product_expansion_l367_36726

theorem polynomial_product_expansion (x : ℝ) :
  (x^2 + 3*x - 4) * (x^2 - 5*x + 6) = x^4 - 2*x^3 - 13*x^2 + 38*x - 24 := by
  sorry

end polynomial_product_expansion_l367_36726


namespace least_positive_integer_multiple_of_53_l367_36706

theorem least_positive_integer_multiple_of_53 :
  ∃ x : ℕ+, (∀ y : ℕ+, y < x → ¬(53 ∣ (3*y)^2 + 2*41*3*y + 41^2)) ∧
             (53 ∣ (3*x)^2 + 2*41*3*x + 41^2) ∧
             x = 4 := by
  sorry

end least_positive_integer_multiple_of_53_l367_36706


namespace cricket_bat_cost_price_l367_36772

theorem cricket_bat_cost_price (profit_a_to_b : ℝ) (profit_b_to_c : ℝ) (price_c : ℝ) :
  profit_a_to_b = 0.20 →
  profit_b_to_c = 0.25 →
  price_c = 228 →
  ∃ (cost_price_a : ℝ),
    cost_price_a * (1 + profit_a_to_b) * (1 + profit_b_to_c) = price_c ∧
    cost_price_a = 152 :=
by sorry

end cricket_bat_cost_price_l367_36772


namespace x_intercept_of_specific_line_l367_36766

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ := sorry

/-- The specific line passing through (2, -2) and (6, 10) -/
def specific_line : Line := { x₁ := 2, y₁ := -2, x₂ := 6, y₂ := 10 }

theorem x_intercept_of_specific_line :
  x_intercept specific_line = 8/3 := by sorry

end x_intercept_of_specific_line_l367_36766


namespace arithmetic_sequence_2011_l367_36756

/-- 
Given an arithmetic sequence with first term a₁ = 1 and common difference d = 3,
prove that 2011 is the 671st term of this sequence.
-/
theorem arithmetic_sequence_2011 : 
  ∀ (a : ℕ → ℕ), 
    a 1 = 1 → 
    (∀ n, a (n + 1) - a n = 3) → 
    a 671 = 2011 :=
by
  sorry

end arithmetic_sequence_2011_l367_36756


namespace journey_distance_l367_36755

theorem journey_distance (total_time : Real) (bike_speed : Real) (walk_speed : Real) 
  (h1 : total_time = 56 / 60) -- 56 minutes converted to hours
  (h2 : bike_speed = 20)
  (h3 : walk_speed = 4) :
  let total_distance := (total_time * bike_speed * walk_speed) / (1/3 * bike_speed + 2/3 * walk_speed)
  let walk_distance := 1/3 * total_distance
  walk_distance = 2.7 := by sorry

end journey_distance_l367_36755


namespace total_pay_is_330_l367_36744

/-- The total weekly pay for two employees, where one earns 120% of the other -/
def total_weekly_pay (y_pay : ℝ) : ℝ :=
  let x_pay := 1.2 * y_pay
  x_pay + y_pay

/-- Proof that the total weekly pay for two employees is 330 when one earns 150 and the other earns 120% of that -/
theorem total_pay_is_330 : total_weekly_pay 150 = 330 := by
  sorry

#eval total_weekly_pay 150

end total_pay_is_330_l367_36744


namespace A_3_2_equals_29_l367_36771

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2_equals_29 : A 3 2 = 29 := by sorry

end A_3_2_equals_29_l367_36771


namespace non_honda_red_percentage_is_51_25_l367_36782

/-- Represents the car population in Chennai -/
structure CarPopulation where
  total : Nat
  honda : Nat
  toyota : Nat
  ford : Nat
  other : Nat
  honda_red_ratio : Rat
  toyota_red_ratio : Rat
  ford_red_ratio : Rat
  other_red_ratio : Rat

/-- Calculates the percentage of non-Honda cars that are red -/
def non_honda_red_percentage (pop : CarPopulation) : Rat :=
  let non_honda_total := pop.toyota + pop.ford + pop.other
  let non_honda_red := pop.toyota * pop.toyota_red_ratio + 
                       pop.ford * pop.ford_red_ratio + 
                       pop.other * pop.other_red_ratio
  (non_honda_red / non_honda_total) * 100

/-- The main theorem stating that the percentage of non-Honda cars that are red is 51.25% -/
theorem non_honda_red_percentage_is_51_25 (pop : CarPopulation) 
  (h1 : pop.total = 900)
  (h2 : pop.honda = 500)
  (h3 : pop.toyota = 200)
  (h4 : pop.ford = 150)
  (h5 : pop.other = 50)
  (h6 : pop.honda_red_ratio = 9/10)
  (h7 : pop.toyota_red_ratio = 3/4)
  (h8 : pop.ford_red_ratio = 3/10)
  (h9 : pop.other_red_ratio = 2/5) :
  non_honda_red_percentage pop = 51.25 := by
  sorry

#eval non_honda_red_percentage {
  total := 900,
  honda := 500,
  toyota := 200,
  ford := 150,
  other := 50,
  honda_red_ratio := 9/10,
  toyota_red_ratio := 3/4,
  ford_red_ratio := 3/10,
  other_red_ratio := 2/5
}

end non_honda_red_percentage_is_51_25_l367_36782


namespace power_sum_of_i_l367_36711

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^67 + i^101 = -i := by sorry

end power_sum_of_i_l367_36711


namespace choose_five_items_eq_48_l367_36776

/-- The number of ways to choose 5 items from 3 distinct types, 
    where no two consecutive items can be of the same type -/
def choose_five_items : ℕ :=
  let first_choice := 3  -- 3 choices for the first item
  let subsequent_choices := 2  -- 2 choices for each subsequent item
  first_choice * subsequent_choices^4

theorem choose_five_items_eq_48 : choose_five_items = 48 := by
  sorry

end choose_five_items_eq_48_l367_36776


namespace cookie_arrangements_count_l367_36736

/-- The number of distinct arrangements of letters in "COOKIE" -/
def cookieArrangements : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1)

/-- Theorem stating that the number of distinct arrangements of letters in "COOKIE" is 360 -/
theorem cookie_arrangements_count : cookieArrangements = 360 := by
  sorry

end cookie_arrangements_count_l367_36736


namespace quadratic_shift_sum_coefficients_l367_36778

def f (x : ℝ) : ℝ := 2 * x^2 - x + 5

def g (x : ℝ) : ℝ := f (x - 7) + 3

theorem quadratic_shift_sum_coefficients :
  ∃ (a b c : ℝ), (∀ x, g x = a * x^2 + b * x + c) ∧ (a + b + c = 86) := by
  sorry

end quadratic_shift_sum_coefficients_l367_36778


namespace min_sum_of_squares_of_roots_l367_36729

theorem min_sum_of_squares_of_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*m*x₁ + (m^2 + 2*m + 3) = 0 →
  x₂^2 - 2*m*x₂ + (m^2 + 2*m + 3) = 0 →
  x₁ ≠ x₂ →
  ∃ (k : ℝ), k = x₁^2 + x₂^2 ∧ k ≥ 9/2 ∧ 
  (∀ (m' : ℝ) (y₁ y₂ : ℝ), 
    y₁^2 - 2*m'*y₁ + (m'^2 + 2*m' + 3) = 0 →
    y₂^2 - 2*m'*y₂ + (m'^2 + 2*m' + 3) = 0 →
    y₁ ≠ y₂ →
    y₁^2 + y₂^2 ≥ 9/2) :=
by sorry

end min_sum_of_squares_of_roots_l367_36729


namespace fibonacci_polynomial_property_l367_36728

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Define the theorem
theorem fibonacci_polynomial_property (p : ℝ → ℝ) :
  (∀ k ∈ Finset.range 991, p (k + 992) = fib (k + 992)) →
  p 1983 = fib 1083 - 1 := by
  sorry

end fibonacci_polynomial_property_l367_36728


namespace chinese_in_group_l367_36746

theorem chinese_in_group (total : ℕ) (americans : ℕ) (australians : ℕ) 
  (h1 : total = 49)
  (h2 : americans = 16)
  (h3 : australians = 11) :
  total - americans - australians = 22 :=
by sorry

end chinese_in_group_l367_36746


namespace team_selection_count_l367_36721

/-- The number of ways to select a team of 5 people from a group of 16 people -/
def select_team (total : ℕ) (team_size : ℕ) : ℕ :=
  Nat.choose total team_size

/-- The total number of students in the math club -/
def total_students : ℕ := 16

/-- The number of boys in the math club -/
def num_boys : ℕ := 7

/-- The number of girls in the math club -/
def num_girls : ℕ := 9

/-- The size of the team to be selected -/
def team_size : ℕ := 5

theorem team_selection_count :
  select_team total_students team_size = 4368 ∧
  total_students = num_boys + num_girls :=
sorry

end team_selection_count_l367_36721


namespace crackers_per_person_is_76_l367_36748

/-- The number of crackers each person receives when Darren and Calvin's crackers are shared equally among themselves and 3 friends. -/
def crackers_per_person : ℕ :=
  let darren_type_a_boxes := 4
  let darren_type_b_boxes := 2
  let crackers_per_type_a_box := 24
  let crackers_per_type_b_box := 30
  let calvin_type_a_boxes := 2 * darren_type_a_boxes - 1
  let calvin_type_b_boxes := darren_type_b_boxes
  let total_crackers := 
    (darren_type_a_boxes + calvin_type_a_boxes) * crackers_per_type_a_box +
    (darren_type_b_boxes + calvin_type_b_boxes) * crackers_per_type_b_box
  let number_of_people := 5
  total_crackers / number_of_people

theorem crackers_per_person_is_76 : crackers_per_person = 76 := by
  sorry

end crackers_per_person_is_76_l367_36748


namespace complement_A_intersect_B_l367_36742

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x : ℝ | x ≤ 0} := by sorry

end complement_A_intersect_B_l367_36742


namespace population_growth_rate_l367_36760

/-- Calculates the average percent increase per year given initial and final populations over a decade. -/
def average_percent_increase (initial_population final_population : ℕ) : ℚ :=
  let total_increase : ℕ := final_population - initial_population
  let average_annual_increase : ℚ := (total_increase : ℚ) / 10
  (average_annual_increase / initial_population) * 100

/-- Theorem stating that the average percent increase per year for the given population change is 7%. -/
theorem population_growth_rate : 
  average_percent_increase 175000 297500 = 7 := by
  sorry

#eval average_percent_increase 175000 297500

end population_growth_rate_l367_36760


namespace two_special_right_triangles_l367_36761

/-- A right-angled triangle with integer sides where the area equals the perimeter -/
structure SpecialRightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a^2 + b^2 = c^2  -- Pythagorean theorem
  h2 : a * b = 2 * (a + b + c)  -- Area equals perimeter

/-- The set of all SpecialRightTriangles -/
def specialRightTriangles : Set SpecialRightTriangle :=
  {t : SpecialRightTriangle | True}

theorem two_special_right_triangles :
  ∃ (t1 t2 : SpecialRightTriangle),
    specialRightTriangles = {t1, t2} ∧
    ((t1.a = 5 ∧ t1.b = 12 ∧ t1.c = 13) ∨ (t1.a = 12 ∧ t1.b = 5 ∧ t1.c = 13)) ∧
    ((t2.a = 6 ∧ t2.b = 8 ∧ t2.c = 10) ∨ (t2.a = 8 ∧ t2.b = 6 ∧ t2.c = 10)) :=
  sorry

#check two_special_right_triangles

end two_special_right_triangles_l367_36761


namespace alice_acorn_price_l367_36779

/-- Given the conditions of Alice and Bob's acorn purchases, prove that Alice paid $15 for each acorn. -/
theorem alice_acorn_price (alice_acorns : ℕ) (bob_price : ℝ) (alice_bob_ratio : ℝ) : 
  alice_acorns = 3600 → 
  bob_price = 6000 → 
  alice_bob_ratio = 9 → 
  (alice_bob_ratio * bob_price) / alice_acorns = 15 := by
sorry

end alice_acorn_price_l367_36779


namespace half_vector_AB_l367_36799

/-- Given two points A and B in a 2D plane, prove that half of the vector from A to B is (2, 1) -/
theorem half_vector_AB (A B : ℝ × ℝ) (h1 : A = (-1, 0)) (h2 : B = (3, 2)) :
  (1 / 2 : ℝ) • (B.1 - A.1, B.2 - A.2) = (2, 1) := by
  sorry

end half_vector_AB_l367_36799


namespace completing_square_equivalence_l367_36768

theorem completing_square_equivalence (x : ℝ) : 
  x^2 - 4*x - 3 = 0 ↔ (x - 2)^2 = 7 := by
  sorry

end completing_square_equivalence_l367_36768


namespace squares_concurrency_l367_36774

-- Define the vertices of the squares as complex numbers
variable (zA zB zC zD zA₁ zB₁ zC₁ zD₁ : ℂ)

-- Define the condition for equally oriented squares
def equally_oriented (zA zB zC zD zA₁ zB₁ zC₁ zD₁ : ℂ) : Prop :=
  ∃ (w t : ℂ), Complex.abs w = 1 ∧
    zA₁ = w * zA + t ∧
    zB₁ = w * zB + t ∧
    zC₁ = w * zC + t ∧
    zD₁ = w * zD + t

-- Define the concurrency condition
def concurrent (zA zB zC zD zA₁ zB₁ zC₁ zD₁ : ℂ) : Prop :=
  ∃ (P : ℂ),
    (zA₁ - zA) / (zB₁ - zB) = (zA₁ - zA) / (zC₁ - zC) ∧
    (zA₁ - zA) / (zB₁ - zB) = (zA₁ - zA) / (zD₁ - zD)

-- State the theorem
theorem squares_concurrency
  (h : equally_oriented zA zB zC zD zA₁ zB₁ zC₁ zD₁) :
  concurrent zA zB zC zD zA₁ zB₁ zC₁ zD₁ :=
by sorry

end squares_concurrency_l367_36774


namespace tino_jellybeans_l367_36725

/-- Proves that Tino has 34 jellybeans given the conditions -/
theorem tino_jellybeans (arnold_jellybeans : ℕ) (lee_jellybeans : ℕ) (tino_jellybeans : ℕ)
  (h1 : arnold_jellybeans = 5)
  (h2 : arnold_jellybeans * 2 = lee_jellybeans)
  (h3 : tino_jellybeans = lee_jellybeans + 24) :
  tino_jellybeans = 34 := by
  sorry

end tino_jellybeans_l367_36725


namespace perfect_square_condition_l367_36795

theorem perfect_square_condition (n : ℕ) : 
  ∃ k : ℕ, n^2 + n + 1 = k^2 ↔ n = 0 :=
by sorry

end perfect_square_condition_l367_36795


namespace sin_2α_plus_π_6_l367_36751

theorem sin_2α_plus_π_6 (α : ℝ) (h : Real.sin (α + π / 3) = 3 / 5) :
  Real.sin (2 * α + π / 6) = -(7 / 25) := by
  sorry

end sin_2α_plus_π_6_l367_36751


namespace max_stickers_purchasable_l367_36701

theorem max_stickers_purchasable (budget : ℚ) (unit_cost : ℚ) : 
  budget = 10 → unit_cost = 3/4 → 
  (∃ (n : ℕ), n * unit_cost ≤ budget ∧ 
    ∀ (m : ℕ), m * unit_cost ≤ budget → m ≤ n) → 
  (∃ (max_stickers : ℕ), max_stickers = 13) :=
by sorry

end max_stickers_purchasable_l367_36701


namespace factorization_equality_l367_36749

theorem factorization_equality (m n : ℝ) : 4 * m^3 * n - 16 * m * n^3 = 4 * m * n * (m + 2*n) * (m - 2*n) := by
  sorry

end factorization_equality_l367_36749


namespace pet_store_bird_count_l367_36750

/-- Calculates the total number of birds in a pet store with specific cage arrangements. -/
theorem pet_store_bird_count : 
  let total_cages : ℕ := 9
  let parrots_per_mixed_cage : ℕ := 2
  let parakeets_per_mixed_cage : ℕ := 3
  let cockatiels_per_mixed_cage : ℕ := 1
  let parakeets_per_special_cage : ℕ := 5
  let special_cage_frequency : ℕ := 3

  let special_cages : ℕ := total_cages / special_cage_frequency
  let mixed_cages : ℕ := total_cages - special_cages

  let total_parrots : ℕ := mixed_cages * parrots_per_mixed_cage
  let total_parakeets : ℕ := (mixed_cages * parakeets_per_mixed_cage) + (special_cages * parakeets_per_special_cage)
  let total_cockatiels : ℕ := mixed_cages * cockatiels_per_mixed_cage

  let total_birds : ℕ := total_parrots + total_parakeets + total_cockatiels
  
  total_birds = 51 := by sorry

end pet_store_bird_count_l367_36750


namespace sum_of_roots_eq_fourteen_l367_36775

theorem sum_of_roots_eq_fourteen : 
  let f : ℝ → ℝ := λ x => (x - 7)^2 - 16
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 14 :=
by sorry

end sum_of_roots_eq_fourteen_l367_36775


namespace arithmetic_calculations_l367_36786

theorem arithmetic_calculations :
  ((-1) + (-6) - (-4) + 0 = -3) ∧
  (24 * (-1/4) / (-3/2) = 4) := by
sorry

end arithmetic_calculations_l367_36786


namespace quadratic_discriminant_l367_36743

theorem quadratic_discriminant : 
  let a : ℚ := 5
  let b : ℚ := 5 + (1 / 5)
  let c : ℚ := 1 / 5
  let discriminant := b^2 - 4*a*c
  discriminant = 576 / 25 := by sorry

end quadratic_discriminant_l367_36743


namespace additive_function_is_scalar_multiple_l367_36770

/-- A function from rationals to rationals satisfying the given additive property -/
def AdditiveFunction (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, f (x + y) = f x + f y

/-- The theorem stating that any additive function on rationals is a scalar multiple -/
theorem additive_function_is_scalar_multiple :
  ∀ f : ℚ → ℚ, AdditiveFunction f → ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
by sorry

end additive_function_is_scalar_multiple_l367_36770


namespace system_solution_l367_36740

noncomputable def solve_system (x : Fin 12 → ℚ) : Prop :=
  x 0 + 12 * x 1 = 15 ∧
  x 0 - 12 * x 1 + 11 * x 2 = 2 ∧
  x 0 - 11 * x 2 + 10 * x 3 = 2 ∧
  x 0 - 10 * x 3 + 9 * x 4 = 2 ∧
  x 0 - 9 * x 4 + 8 * x 5 = 2 ∧
  x 0 - 8 * x 5 + 7 * x 6 = 2 ∧
  x 0 - 7 * x 6 + 6 * x 7 = 2 ∧
  x 0 - 6 * x 7 + 5 * x 8 = 2 ∧
  x 0 - 5 * x 8 + 4 * x 9 = 2 ∧
  x 0 - 4 * x 9 + 3 * x 10 = 2 ∧
  x 0 - 3 * x 10 + 2 * x 11 = 2 ∧
  x 0 - 2 * x 11 = 2

theorem system_solution :
  ∃! x : Fin 12 → ℚ, solve_system x ∧
    x 0 = 37/12 ∧ x 1 = 143/144 ∧ x 2 = 65/66 ∧ x 3 = 39/40 ∧
    x 4 = 26/27 ∧ x 5 = 91/96 ∧ x 6 = 13/14 ∧ x 7 = 65/72 ∧
    x 8 = 13/15 ∧ x 9 = 13/16 ∧ x 10 = 13/18 ∧ x 11 = 13/24 :=
by sorry

end system_solution_l367_36740


namespace eleven_rays_max_regions_l367_36769

/-- The maximum number of regions into which n rays can split a plane -/
def max_regions (n : ℕ) : ℕ := (n^2 - n + 2) / 2

/-- Theorem: 11 rays can split a plane into a maximum of 56 regions -/
theorem eleven_rays_max_regions : max_regions 11 = 56 := by
  sorry

end eleven_rays_max_regions_l367_36769


namespace video_game_earnings_l367_36787

def total_games : ℕ := 15
def non_working_games : ℕ := 9
def price_per_game : ℕ := 5

theorem video_game_earnings : 
  (total_games - non_working_games) * price_per_game = 30 := by
  sorry

end video_game_earnings_l367_36787


namespace parabola_segment_length_l367_36785

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  hp : p < 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the parabola -/
def onParabola (para : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * para.p * pt.x

/-- Represents a line passing through the focus of the parabola at an angle with the x-axis -/
structure FocusLine where
  para : Parabola
  angle : ℝ

/-- Calculates the length of the segment AB formed by the intersection of the focus line with the parabola -/
noncomputable def segmentLength (para : Parabola) (fl : FocusLine) : ℝ :=
  sorry -- Actual calculation would go here

theorem parabola_segment_length 
  (para : Parabola) 
  (ptA : Point) 
  (fl : FocusLine) :
  onParabola para ptA → 
  ptA.x = -2 → 
  ptA.y = -4 → 
  fl.para = para → 
  fl.angle = π/3 → 
  segmentLength para fl = 32/3 := by
  sorry

end parabola_segment_length_l367_36785


namespace ultra_squarish_exists_l367_36789

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

def digits_nonzero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0

def first_three_digits (n : ℕ) : ℕ :=
  (n / 10000) % 1000

def middle_two_digits (n : ℕ) : ℕ :=
  (n / 100) % 100

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

theorem ultra_squarish_exists :
  ∃ M : ℕ,
    1000000 ≤ M ∧ M < 10000000 ∧
    is_perfect_square M ∧
    digits_nonzero M ∧
    is_perfect_square (first_three_digits M) ∧
    is_perfect_square (middle_two_digits M) ∧
    is_perfect_square (last_two_digits M) :=
  sorry

end ultra_squarish_exists_l367_36789


namespace euler_family_mean_age_is_11_l367_36780

def euler_family_mean_age (ages : List ℕ) : ℚ :=
  (ages.sum : ℚ) / ages.length

theorem euler_family_mean_age_is_11 :
  let ages := [8, 8, 8, 13, 13, 16]
  euler_family_mean_age ages = 11 := by
sorry

end euler_family_mean_age_is_11_l367_36780


namespace right_triangle_angle_sum_l367_36705

theorem right_triangle_angle_sum (A B C : ℝ) : 
  A = 20 → C = 90 → A + B + C = 180 → B = 70 := by sorry

end right_triangle_angle_sum_l367_36705


namespace cryptarithmetic_solution_l367_36790

def is_valid_assignment (w h i t e a r p c n : Nat) : Prop :=
  w < 10 ∧ h < 10 ∧ i < 10 ∧ t < 10 ∧ e < 10 ∧ a < 10 ∧ r < 10 ∧ p < 10 ∧ c < 10 ∧ n < 10 ∧
  w ≠ h ∧ w ≠ i ∧ w ≠ t ∧ w ≠ e ∧ w ≠ a ∧ w ≠ r ∧ w ≠ p ∧ w ≠ c ∧ w ≠ n ∧
  h ≠ i ∧ h ≠ t ∧ h ≠ e ∧ h ≠ a ∧ h ≠ r ∧ h ≠ p ∧ h ≠ c ∧ h ≠ n ∧
  i ≠ t ∧ i ≠ e ∧ i ≠ a ∧ i ≠ r ∧ i ≠ p ∧ i ≠ c ∧ i ≠ n ∧
  t ≠ e ∧ t ≠ a ∧ t ≠ r ∧ t ≠ p ∧ t ≠ c ∧ t ≠ n ∧
  e ≠ a ∧ e ≠ r ∧ e ≠ p ∧ e ≠ c ∧ e ≠ n ∧
  a ≠ r ∧ a ≠ p ∧ a ≠ c ∧ a ≠ n ∧
  r ≠ p ∧ r ≠ c ∧ r ≠ n ∧
  p ≠ c ∧ p ≠ n ∧
  c ≠ n

def white_plus_water_equals_picnic (w h i t e a r p c n : Nat) : Prop :=
  10000 * w + 1000 * h + 100 * i + 10 * t + e +
  10000 * w + 1000 * a + 100 * t + 10 * e + r =
  100000 * p + 10000 * i + 1000 * c + 100 * n + 10 * i + c

theorem cryptarithmetic_solution :
  ∃! (w h i t e a r p c n : Nat),
    is_valid_assignment w h i t e a r p c n ∧
    white_plus_water_equals_picnic w h i t e a r p c n ∧
    100000 * p + 10000 * i + 1000 * c + 100 * n + 10 * i + c = 169069 := by
  sorry

end cryptarithmetic_solution_l367_36790


namespace base6_division_equality_l367_36718

-- Define a function to convert from base 6 to base 10
def base6ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 6
def base10ToBase6 (n : ℕ) : ℕ := sorry

-- Define the division operation in base 6
def divBase6 (a b : ℕ) : ℕ := base10ToBase6 (base6ToBase10 a / base6ToBase10 b)

-- Theorem statement
theorem base6_division_equality :
  divBase6 2314 14 = 135 := by sorry

end base6_division_equality_l367_36718


namespace part_to_whole_ratio_l367_36714

theorem part_to_whole_ratio (N : ℚ) (x : ℚ) (h1 : N = 280) (h2 : x + 4 = N / 4 - 10) : x / N = 1 / 5 := by
  sorry

end part_to_whole_ratio_l367_36714


namespace two_points_with_area_three_l367_36712

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := x/4 + y/3 = 1

/-- Point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

/-- Intersection points of the line and ellipse -/
structure IntersectionPoints where
  A : PointOnEllipse
  B : PointOnEllipse
  on_line_A : line A.x A.y
  on_line_B : line B.x B.y

/-- Area of a triangle given three points -/
noncomputable def triangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem two_points_with_area_three (intersections : IntersectionPoints) :
  ∃! (points : Finset PointOnEllipse),
    points.card = 2 ∧
    ∀ P ∈ points,
      triangleArea (P.x, P.y) (intersections.A.x, intersections.A.y) (intersections.B.x, intersections.B.y) = 3 :=
sorry

end two_points_with_area_three_l367_36712


namespace sqrt_simplification_algebraic_simplification_l367_36723

-- Problem 1
theorem sqrt_simplification :
  Real.sqrt 6 * Real.sqrt (2/3) / Real.sqrt 2 = Real.sqrt 2 := by sorry

-- Problem 2
theorem algebraic_simplification :
  (Real.sqrt 2 + Real.sqrt 5)^2 - (Real.sqrt 2 + Real.sqrt 5)*(Real.sqrt 2 - Real.sqrt 5) = 10 + 2*Real.sqrt 10 := by sorry

end sqrt_simplification_algebraic_simplification_l367_36723


namespace stock_ratio_proof_l367_36793

def stock_problem (expensive_shares : ℕ) (other_shares : ℕ) (total_value : ℕ) (expensive_price : ℕ) : Prop :=
  ∃ (other_price : ℕ),
    expensive_shares * expensive_price + other_shares * other_price = total_value ∧
    expensive_price / other_price = 2

theorem stock_ratio_proof :
  stock_problem 14 26 2106 78 := by
  sorry

end stock_ratio_proof_l367_36793


namespace zoo_trip_attendance_l367_36702

/-- The number of buses available for the zoo trip -/
def num_buses : ℕ := 3

/-- The number of people that would go in each bus if evenly distributed -/
def people_per_bus : ℕ := 73

/-- The total number of people going to the zoo -/
def total_people : ℕ := num_buses * people_per_bus

theorem zoo_trip_attendance : total_people = 219 := by
  sorry

end zoo_trip_attendance_l367_36702


namespace externally_tangent_circles_equation_l367_36784

-- Define the radii and angle
variable (r r' φ : ℝ)

-- Define the conditions
variable (hr : r > 0)
variable (hr' : r' > 0)
variable (hφ : 0 < φ ∧ φ < π)

-- Define the externally tangent circles condition
variable (h_tangent : r + r' > 0)

-- Theorem statement
theorem externally_tangent_circles_equation :
  (r + r')^2 * Real.sin φ = 4 * (r - r') * Real.sqrt (r * r') := by
  sorry

end externally_tangent_circles_equation_l367_36784


namespace locus_of_p_l367_36765

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the point A on the hyperbola
def point_on_hyperbola (a b x y : ℝ) : Prop := hyperbola a b x y ∧ x ≠ 0 ∧ y ≠ 0

-- Define the reflection of a point about y-axis
def reflect_y (x y : ℝ) : ℝ × ℝ := (-x, y)

-- Define the reflection of a point about x-axis
def reflect_x (x y : ℝ) : ℝ × ℝ := (x, -y)

-- Define the reflection of a point about origin
def reflect_origin (x y : ℝ) : ℝ × ℝ := (-x, -y)

-- Define perpendicularity of two lines
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem
theorem locus_of_p (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∀ (x y : ℝ), y ≠ 0 →
  (∃ (x0 y0 x1 y1 : ℝ),
    point_on_hyperbola a b x0 y0 ∧
    point_on_hyperbola a b x1 y1 ∧
    perpendicular (x1 - x0) (y1 - y0) (-2*x0) (-2*y0) ∧
    x = ((a^2 + b^2) / (a^2 - b^2)) * x0 ∧
    y = -((a^2 + b^2) / (a^2 - b^2)) * y0) →
  x^2 / a^2 - y^2 / b^2 = (a^2 + b^2)^2 / (a^2 - b^2)^2 :=
by sorry

end locus_of_p_l367_36765


namespace intersection_of_A_and_B_l367_36703

def A : Set Int := {x | |x| < 3}
def B : Set Int := {x | |x| > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} := by sorry

end intersection_of_A_and_B_l367_36703


namespace systematic_sampling_second_group_l367_36767

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (groupNumber : ℕ) : ℕ :=
  let interval := totalStudents / sampleSize
  (groupNumber - 1) * interval + 1

theorem systematic_sampling_second_group
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (h1 : totalStudents = 160)
  (h2 : sampleSize = 20)
  (h3 : systematicSample totalStudents sampleSize 16 = 123) :
  systematicSample totalStudents sampleSize 2 = 11 := by
sorry

end systematic_sampling_second_group_l367_36767


namespace smallest_addition_for_multiple_of_four_l367_36707

theorem smallest_addition_for_multiple_of_four : 
  (∃ n : ℕ+, 4 ∣ (587 + n) ∧ ∀ m : ℕ+, 4 ∣ (587 + m) → n ≤ m) ∧ 
  (∀ n : ℕ+, (4 ∣ (587 + n) ∧ ∀ m : ℕ+, 4 ∣ (587 + m) → n ≤ m) → n = 1) :=
by sorry

end smallest_addition_for_multiple_of_four_l367_36707


namespace totalDispatchPlansIs36_l367_36709

/-- The number of people to choose from -/
def totalPeople : Nat := 5

/-- The number of tasks to be assigned -/
def totalTasks : Nat := 4

/-- The number of people who can only do certain tasks -/
def restrictedPeople : Nat := 2

/-- The number of tasks that restricted people can do -/
def restrictedTasks : Nat := 2

/-- The number of people who can do any task -/
def unrestrictedPeople : Nat := totalPeople - restrictedPeople

/-- Calculate the number of ways to select and arrange k items from n items -/
def arrangementNumber (n k : Nat) : Nat := sorry

/-- Calculate the number of ways to select k items from n items -/
def combinationNumber (n k : Nat) : Nat := sorry

/-- The total number of different dispatch plans -/
def totalDispatchPlans : Nat :=
  combinationNumber restrictedPeople 1 * combinationNumber restrictedTasks 1 * 
    arrangementNumber unrestrictedPeople 3 +
  arrangementNumber restrictedPeople 2 * arrangementNumber unrestrictedPeople 2

/-- Theorem stating that the total number of different dispatch plans is 36 -/
theorem totalDispatchPlansIs36 : totalDispatchPlans = 36 := by sorry

end totalDispatchPlansIs36_l367_36709


namespace cube_equation_solution_l367_36737

theorem cube_equation_solution (x y : ℝ) : x^3 - 8*y^3 = 0 ↔ x = 2*y := by
  sorry

end cube_equation_solution_l367_36737


namespace parabola_line_intersection_l367_36759

/-- The parabola y = 2x^2 intersects with the line y = kx + 2 at points A and B.
    M is the midpoint of AB, and N is the foot of the perpendicular from M to the x-axis.
    If the dot product of NA and NB is zero, then k = ±4√3. -/
theorem parabola_line_intersection (k : ℝ) : 
  let C : ℝ → ℝ := λ x => 2 * x^2
  let L : ℝ → ℝ := λ x => k * x + 2
  let A : ℝ × ℝ := (x₁, C x₁)
  let B : ℝ × ℝ := (x₂, C x₂)
  let M : ℝ × ℝ := ((x₁ + x₂)/2, (C x₁ + C x₂)/2)
  let N : ℝ × ℝ := (M.1, 0)
  C x₁ = L x₁ ∧ C x₂ = L x₂ ∧ x₁ ≠ x₂ →
  (A.1 - N.1) * (B.1 - N.1) + (A.2 - N.2) * (B.2 - N.2) = 0 →
  k = 4 * Real.sqrt 3 ∨ k = -4 * Real.sqrt 3 := by
sorry

end parabola_line_intersection_l367_36759


namespace factorization_equality_l367_36734

theorem factorization_equality (x : ℝ) : 3 * x * (x - 5) + 4 * (x - 5) = (3 * x + 4) * (x - 5) := by
  sorry

end factorization_equality_l367_36734


namespace blakes_initial_money_l367_36788

/-- Blake's grocery shopping problem -/
theorem blakes_initial_money (orange_cost apples_cost mangoes_cost change : ℕ) 
  (h1 : orange_cost = 40)
  (h2 : apples_cost = 50)
  (h3 : mangoes_cost = 60)
  (h4 : change = 150) : 
  orange_cost + apples_cost + mangoes_cost + change = 300 := by
  sorry

#check blakes_initial_money

end blakes_initial_money_l367_36788


namespace center_of_mass_theorem_l367_36747

/-- Represents the center of mass coordinates for an n × n chessboard -/
structure CenterOfMass where
  x : ℚ
  y : ℚ

/-- Calculates the center of mass for the sum-1 rule -/
def centerOfMassSumRule (n : ℕ) : CenterOfMass :=
  { x := ((n + 1) * (7 * n - 1)) / (12 * n),
    y := ((n + 1) * (7 * n - 1)) / (12 * n) }

/-- Calculates the center of mass for the product rule -/
def centerOfMassProductRule (n : ℕ) : CenterOfMass :=
  { x := (2 * n + 1) / 3,
    y := (2 * n + 1) / 3 }

/-- Theorem stating the correctness of the center of mass calculations -/
theorem center_of_mass_theorem (n : ℕ) :
  (centerOfMassSumRule n).x = ((n + 1) * (7 * n - 1)) / (12 * n) ∧
  (centerOfMassSumRule n).y = ((n + 1) * (7 * n - 1)) / (12 * n) ∧
  (centerOfMassProductRule n).x = (2 * n + 1) / 3 ∧
  (centerOfMassProductRule n).y = (2 * n + 1) / 3 := by
  sorry

end center_of_mass_theorem_l367_36747


namespace mikaela_personal_needs_fraction_l367_36783

/-- Calculates the fraction of total earnings spent on personal needs --/
def fraction_spent_on_personal_needs (hourly_rate : ℚ) (first_month_hours : ℕ) (second_month_additional_hours : ℕ) (amount_saved : ℚ) : ℚ :=
  let first_month_earnings := hourly_rate * first_month_hours
  let second_month_hours := first_month_hours + second_month_additional_hours
  let second_month_earnings := hourly_rate * second_month_hours
  let total_earnings := first_month_earnings + second_month_earnings
  let amount_spent := total_earnings - amount_saved
  amount_spent / total_earnings

/-- Proves that Mikaela spent 4/5 of her total earnings on personal needs --/
theorem mikaela_personal_needs_fraction :
  fraction_spent_on_personal_needs 10 35 5 150 = 4/5 := by
  sorry

end mikaela_personal_needs_fraction_l367_36783


namespace area_of_intersection_is_point_eight_l367_36791

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space with slope-intercept form y = mx + b -/
structure Line2D where
  m : ℝ
  b : ℝ

/-- Calculates the area of intersection between two triangles -/
noncomputable def areaOfIntersection (a b c d e : Point2D) (lineAC lineDE : Line2D) : ℝ :=
  sorry

/-- Theorem: The area of intersection between two specific triangles is 0.8 square units -/
theorem area_of_intersection_is_point_eight :
  let a : Point2D := ⟨1, 4⟩
  let b : Point2D := ⟨0, 0⟩
  let c : Point2D := ⟨2, 0⟩
  let d : Point2D := ⟨0, 1⟩
  let e : Point2D := ⟨4, 0⟩
  let lineAC : Line2D := ⟨-4, 8⟩
  let lineDE : Line2D := ⟨-1/4, 1⟩
  areaOfIntersection a b c d e lineAC lineDE = 0.8 :=
by
  sorry

end area_of_intersection_is_point_eight_l367_36791


namespace jake_peaches_l367_36792

/-- Given information about peaches owned by Steven, Jill, and Jake -/
theorem jake_peaches (steven_peaches : ℕ) (jill_peaches : ℕ) (jake_peaches : ℕ)
  (h1 : steven_peaches = 15)
  (h2 : steven_peaches = jill_peaches + 14)
  (h3 : jake_peaches = steven_peaches - 7) :
  jake_peaches = 8 := by
  sorry

end jake_peaches_l367_36792


namespace parabola_intersection_l367_36733

theorem parabola_intersection (x₁ x₂ : ℝ) (m : ℝ) : 
  (∀ x y, x^2 = 4*y → ∃ k, y = k*x + m) →  -- Line passes through (0, m) and intersects parabola
  x₁ * x₂ = -4 →                           -- Product of x-coordinates is -4
  m = 1 :=                                 -- Conclusion: m must be 1
by sorry

end parabola_intersection_l367_36733


namespace select_captains_l367_36727

theorem select_captains (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  Nat.choose n k = 1365 := by
  sorry

end select_captains_l367_36727


namespace det_matrix1_det_matrix2_l367_36753

-- Define the determinant function for 2x2 matrices
def det2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Theorem for the first matrix
theorem det_matrix1 : det2 2 5 (-3) (-4) = 7 := by sorry

-- Theorem for the second matrix
theorem det_matrix2 (a b : ℝ) : det2 (a^2) (a*b) (a*b) (b^2) = 0 := by sorry

end det_matrix1_det_matrix2_l367_36753


namespace min_value_theorem_l367_36715

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 3 * y = 1 → 1 / a + 2 / b ≤ 1 / x + 2 / y) ∧
  1 / a + 2 / b = 3 * Real.sqrt 6 + 9 :=
sorry

end min_value_theorem_l367_36715


namespace geometric_sequence_sum_l367_36777

/-- Given a geometric sequence with first term a and common ratio r, 
    such that the sum of the first 1500 terms is 300 and 
    the sum of the first 3000 terms is 570,
    prove that the sum of the first 4500 terms is 813. -/
theorem geometric_sequence_sum (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300) 
  (h2 : a * (1 - r^3000) / (1 - r) = 570) : 
  a * (1 - r^4500) / (1 - r) = 813 := by
  sorry

end geometric_sequence_sum_l367_36777


namespace arithmetic_sequence_property_l367_36798

/-- Given an arithmetic sequence {a_n} with positive terms, sum S_n, and common difference d,
    if {√S_n} is also arithmetic with the same difference d, then a_n = (2n - 1) / 4 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, S n = n * a 1 + n * (n - 1) / 2 * d) →
  (∀ n, a (n + 1) = a n + d) →
  (∀ n, Real.sqrt (S (n + 1)) = Real.sqrt (S n) + d) →
  ∀ n, a n = (2 * n - 1) / 4 := by
sorry

end arithmetic_sequence_property_l367_36798


namespace book_reorganization_l367_36745

theorem book_reorganization (initial_boxes : Nat) (initial_books_per_box : Nat) (new_books_per_box : Nat) :
  initial_boxes = 1278 →
  initial_books_per_box = 45 →
  new_books_per_box = 46 →
  (initial_boxes * initial_books_per_box) % new_books_per_box = 10 := by
  sorry

end book_reorganization_l367_36745


namespace mobile_price_change_l367_36781

theorem mobile_price_change (initial_price : ℝ) (decrease_percent : ℝ) : 
  (initial_price * 1.4 * (1 - decrease_percent / 100) = initial_price * 1.18999999999999993) →
  decrease_percent = 15 := by
sorry

end mobile_price_change_l367_36781


namespace circumscribed_sphere_surface_area_l367_36794

/-- The surface area of a sphere circumscribing a rectangular solid with edge lengths 2, 3, and 4 is 29π. -/
theorem circumscribed_sphere_surface_area (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 4) :
  let diagonal_squared := a^2 + b^2 + c^2
  let radius := Real.sqrt (diagonal_squared / 4)
  4 * Real.pi * radius^2 = 29 * Real.pi := by
  sorry

end circumscribed_sphere_surface_area_l367_36794


namespace six_digit_permutations_eq_60_l367_36722

/-- The number of different positive, six-digit integers that can be formed using the digits 1, 1, 3, 3, 3, and 6 -/
def six_digit_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 3)

/-- Theorem stating that the number of different positive, six-digit integers
    that can be formed using the digits 1, 1, 3, 3, 3, and 6 is equal to 60 -/
theorem six_digit_permutations_eq_60 : six_digit_permutations = 60 := by
  sorry

end six_digit_permutations_eq_60_l367_36722


namespace trig_identities_l367_36796

/-- Given that sin(3π + α) = 2sin(3π/2 + α), prove two trigonometric identities. -/
theorem trig_identities (α : ℝ) 
  (h : Real.sin (3 * Real.pi + α) = 2 * Real.sin ((3 * Real.pi) / 2 + α)) : 
  (((2 * Real.sin α - 3 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α)) = 7 / 17) ∧ 
  ((Real.sin α)^2 + Real.sin (2 * α) = 0) := by
  sorry

end trig_identities_l367_36796


namespace solution_sets_l367_36797

-- Define the solution sets
def S (c b : ℝ) : Set ℝ := {x | c * x^2 + x + b < 0}
def M (b c : ℝ) : Set ℝ := {x | b * x^2 + x + c > 0}
def N (a : ℝ) : Set ℝ := {x | x^2 + x < a^2 - a}

-- State the theorem
theorem solution_sets :
  ∃ (c b : ℝ),
    (S c b = {x | -1 < x ∧ x < 1/2}) ∧
    (∃ (a : ℝ), M b c ∪ (Set.univ \ N a) = Set.univ) →
    (M b c = {x | -1 < x ∧ x < 2}) ∧
    {a : ℝ | 0 ≤ a ∧ a ≤ 1} = {a : ℝ | ∃ (b c : ℝ), M b c ∪ (Set.univ \ N a) = Set.univ} :=
by sorry

end solution_sets_l367_36797


namespace A_intersect_B_is_empty_l367_36732

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem A_intersect_B_is_empty : A ∩ B = ∅ := by
  sorry

end A_intersect_B_is_empty_l367_36732


namespace white_pairs_coincide_l367_36752

/-- Represents the number of triangles of each color in each half of the figure -/
structure HalfFigure where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs when the figure is folded -/
structure CoincidingPairs where
  redRed : ℕ
  blueBlue : ℕ
  redWhite : ℕ
  whiteWhite : ℕ

theorem white_pairs_coincide (half : HalfFigure) (pairs : CoincidingPairs) : 
  half.red = 4 ∧ 
  half.blue = 6 ∧ 
  half.white = 10 ∧ 
  pairs.redRed = 3 ∧ 
  pairs.blueBlue = 4 ∧ 
  pairs.redWhite = 3 → 
  pairs.whiteWhite = 5 := by
sorry

end white_pairs_coincide_l367_36752


namespace subtraction_of_mixed_numbers_l367_36739

theorem subtraction_of_mixed_numbers : (2 + 5/6) - (1 + 1/3) = 3/2 := by
  sorry

end subtraction_of_mixed_numbers_l367_36739


namespace subtract_correction_l367_36719

theorem subtract_correction (x : ℤ) (h : x - 42 = 50) : x - 24 = 68 := by
  sorry

end subtract_correction_l367_36719


namespace ball_drawing_theorem_l367_36763

/-- Represents the outcome of drawing balls from a bag -/
structure BallDrawing where
  totalBalls : Nat
  redBalls : Nat
  blackBalls : Nat
  drawCount : Nat

/-- Calculates the expectation for drawing with replacement -/
def expectationWithReplacement (bd : BallDrawing) : Rat :=
  bd.drawCount * (bd.redBalls : Rat) / bd.totalBalls

/-- Calculates the variance for drawing with replacement -/
def varianceWithReplacement (bd : BallDrawing) : Rat :=
  bd.drawCount * (bd.redBalls : Rat) / bd.totalBalls * (1 - (bd.redBalls : Rat) / bd.totalBalls)

/-- Calculates the expectation for drawing without replacement -/
noncomputable def expectationWithoutReplacement (bd : BallDrawing) : Rat :=
  sorry

/-- Calculates the variance for drawing without replacement -/
noncomputable def varianceWithoutReplacement (bd : BallDrawing) : Rat :=
  sorry

theorem ball_drawing_theorem (bd : BallDrawing) 
    (h1 : bd.totalBalls = 10)
    (h2 : bd.redBalls = 4)
    (h3 : bd.blackBalls = 6)
    (h4 : bd.drawCount = 3) :
    expectationWithReplacement bd = expectationWithoutReplacement bd ∧
    varianceWithReplacement bd > varianceWithoutReplacement bd :=
  by sorry

end ball_drawing_theorem_l367_36763


namespace circle_equation_equivalence_l367_36731

theorem circle_equation_equivalence :
  ∀ x y : ℝ, x^2 - 6*x + y^2 - 10*y + 18 = 0 ↔ (x-3)^2 + (y-5)^2 = 16 := by
  sorry

end circle_equation_equivalence_l367_36731


namespace forest_width_is_correct_l367_36704

/-- The width of a forest in miles -/
def forest_width : ℝ := 6

/-- The length of the forest in miles -/
def forest_length : ℝ := 4

/-- The number of trees per square mile -/
def trees_per_square_mile : ℕ := 600

/-- The number of trees one logger can cut per day -/
def trees_per_logger_per_day : ℕ := 6

/-- The number of days in a month -/
def days_per_month : ℕ := 30

/-- The number of loggers working -/
def number_of_loggers : ℕ := 8

/-- The number of months it takes to cut down all trees -/
def months_to_cut_all_trees : ℕ := 10

theorem forest_width_is_correct : 
  forest_width * forest_length * trees_per_square_mile = 
  (trees_per_logger_per_day * days_per_month * number_of_loggers * months_to_cut_all_trees : ℝ) := by
  sorry

end forest_width_is_correct_l367_36704


namespace jogger_distance_ahead_l367_36738

/-- Prove that a jogger is 250 meters ahead of a train's engine given the specified conditions -/
theorem jogger_distance_ahead (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (5/18) →
  train_speed = 45 * (5/18) →
  train_length = 120 →
  passing_time = 37 →
  (train_speed - jogger_speed) * passing_time - train_length = 250 := by
  sorry

end jogger_distance_ahead_l367_36738


namespace basketball_team_selection_l367_36773

def num_players : ℕ := 12
def team_size : ℕ := 6
def captain_count : ℕ := 1

theorem basketball_team_selection :
  (num_players.choose captain_count) * ((num_players - captain_count).choose (team_size - captain_count)) = 5544 := by
  sorry

end basketball_team_selection_l367_36773


namespace tangent_line_at_one_one_l367_36730

/-- The equation of the tangent line to y = x^2 at (1, 1) is 2x - y - 1 = 0 -/
theorem tangent_line_at_one_one :
  let f : ℝ → ℝ := λ x ↦ x^2
  let point : ℝ × ℝ := (1, 1)
  let tangent_line : ℝ → ℝ → Prop := λ x y ↦ 2*x - y - 1 = 0
  (∀ x, HasDerivAt f (2*x) x) →
  tangent_line point.1 point.2 ∧
  ∀ x y, tangent_line x y ↔ y - point.2 = (2 * point.1) * (x - point.1) :=
by
  sorry


end tangent_line_at_one_one_l367_36730


namespace dolls_ratio_l367_36720

theorem dolls_ratio (R S G : ℕ) : 
  S = G + 2 →
  G = 50 →
  R + S + G = 258 →
  R / S = 3 := by
sorry

end dolls_ratio_l367_36720


namespace count_four_digit_with_three_is_1000_l367_36757

/-- The count of four-digit positive integers with the thousands digit 3 -/
def count_four_digit_with_three : ℕ :=
  (List.range 10).length * (List.range 10).length * (List.range 10).length

/-- Theorem: The count of four-digit positive integers with the thousands digit 3 is 1000 -/
theorem count_four_digit_with_three_is_1000 :
  count_four_digit_with_three = 1000 := by
  sorry

end count_four_digit_with_three_is_1000_l367_36757


namespace y_derivative_l367_36741

noncomputable def y (x : ℝ) : ℝ := Real.sin x + Real.exp x * Real.cos x

theorem y_derivative (x : ℝ) : 
  deriv y x = (1 + Real.exp x) * Real.cos x - Real.exp x * Real.sin x :=
by sorry

end y_derivative_l367_36741


namespace total_scissors_l367_36764

/-- The total number of scissors after adding more is equal to the sum of the initial number of scissors and the number of scissors added. -/
theorem total_scissors (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end total_scissors_l367_36764


namespace cos_3theta_l367_36762

theorem cos_3theta (θ : ℝ) : Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 2) / 4 → Complex.cos (3 * θ) = 9 / 64 := by
  sorry

end cos_3theta_l367_36762


namespace right_triangle_shorter_leg_l367_36735

theorem right_triangle_shorter_leg : 
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 16 :=          -- The shorter leg is 16
by
  sorry

end right_triangle_shorter_leg_l367_36735


namespace chicken_cost_proof_l367_36716

def total_cost : ℝ := 16
def beef_pounds : ℝ := 3
def beef_price_per_pound : ℝ := 4
def oil_price : ℝ := 1
def people_paying : ℕ := 3
def individual_payment : ℝ := 1

theorem chicken_cost_proof :
  total_cost - (beef_pounds * beef_price_per_pound + oil_price) = people_paying * individual_payment := by
  sorry

end chicken_cost_proof_l367_36716


namespace leila_order_cost_l367_36713

/-- Calculates the total cost of Leila's cake order --/
def total_cost (chocolate_quantity : ℕ) (chocolate_price : ℕ) 
                (strawberry_quantity : ℕ) (strawberry_price : ℕ) : ℕ :=
  chocolate_quantity * chocolate_price + strawberry_quantity * strawberry_price

/-- Proves that the total cost of Leila's order is $168 --/
theorem leila_order_cost : 
  total_cost 3 12 6 22 = 168 := by
  sorry

#eval total_cost 3 12 6 22

end leila_order_cost_l367_36713


namespace gcd_228_1995_l367_36754

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l367_36754


namespace pencil_distribution_l367_36758

theorem pencil_distribution (total_pencils : ℕ) (pencils_per_student : ℕ) (h1 : total_pencils = 42) (h2 : pencils_per_student = 3) :
  total_pencils / pencils_per_student = 14 := by
  sorry

end pencil_distribution_l367_36758


namespace divisibility_by_twelve_l367_36710

theorem divisibility_by_twelve (a b c d : ℤ) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) := by
  sorry

end divisibility_by_twelve_l367_36710


namespace range_of_a_l367_36717

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 3)^2 ≤ 4}
def B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - a)^2 ≤ 1/4}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (A ∩ B a = B a) →
  (-3 - Real.sqrt 5 / 2 ≤ a ∧ a ≤ -3 + Real.sqrt 5 / 2) :=
by sorry

end range_of_a_l367_36717


namespace total_cookies_kept_l367_36724

def oatmeal_baked : ℕ := 40
def sugar_baked : ℕ := 28
def chocolate_baked : ℕ := 55

def oatmeal_given : ℕ := 26
def sugar_given : ℕ := 17
def chocolate_given : ℕ := 34

def cookies_kept (baked given : ℕ) : ℕ := baked - given

theorem total_cookies_kept :
  cookies_kept oatmeal_baked oatmeal_given +
  cookies_kept sugar_baked sugar_given +
  cookies_kept chocolate_baked chocolate_given = 46 := by
  sorry

end total_cookies_kept_l367_36724


namespace abhay_speed_l367_36708

theorem abhay_speed (distance : ℝ) (abhay_speed : ℝ) (sameer_speed : ℝ) : 
  distance = 24 →
  distance / abhay_speed = distance / sameer_speed + 2 →
  distance / (2 * abhay_speed) = distance / sameer_speed - 1 →
  abhay_speed = 12 := by
sorry

end abhay_speed_l367_36708


namespace constant_term_expansion_l367_36700

/-- The constant term in the expansion of (2x + 1/x - 1)^5 is -161 -/
theorem constant_term_expansion : 
  let f : ℝ → ℝ := λ x => (2*x + 1/x - 1)^5
  ∃ g : ℝ → ℝ, ∀ x ≠ 0, f x = g x + (-161) := by
  sorry

end constant_term_expansion_l367_36700
