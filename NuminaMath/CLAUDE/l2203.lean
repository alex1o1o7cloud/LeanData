import Mathlib

namespace NUMINAMATH_CALUDE_similar_terms_and_system_solution_l2203_220371

theorem similar_terms_and_system_solution :
  ∀ (m n : ℤ) (a b : ℝ) (x y : ℝ),
    (m - 1 = n - 2*m ∧ m + n = 3*m + n - 4) →
    (m*x + (n-2)*y = 24 ∧ 2*m*x + n*y = 46) →
    (x = 9 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_similar_terms_and_system_solution_l2203_220371


namespace NUMINAMATH_CALUDE_vector_BC_coordinates_l2203_220361

/-- Given points A and B, and vector AC, prove that vector BC has specific coordinates -/
theorem vector_BC_coordinates (A B C : ℝ × ℝ) : 
  A = (0, 1) → B = (3, 2) → C - A = (-4, -3) → C - B = (-7, -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_coordinates_l2203_220361


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2203_220329

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2203_220329


namespace NUMINAMATH_CALUDE_complex_subtraction_l2203_220381

theorem complex_subtraction (c d : ℂ) (h1 : c = 5 - 3*I) (h2 : d = 2 - I) : 
  c - 3*d = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l2203_220381


namespace NUMINAMATH_CALUDE_m_range_l2203_220378

-- Define the conditions
def P (x : ℝ) : Prop := x^2 - 3*x + 2 > 0
def q (x m : ℝ) : Prop := x < m

-- Define the theorem
theorem m_range (m : ℝ) : 
  (∀ x, ¬(P x) → q x m) ∧ (∃ x, q x m ∧ P x) → m > 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2203_220378


namespace NUMINAMATH_CALUDE_sheridan_fish_count_l2203_220363

/-- Calculate the number of fish Mrs. Sheridan has left -/
def fish_remaining (initial : ℕ) (received : ℕ) (given_away : ℕ) (sold : ℕ) : ℕ :=
  initial + received - given_away - sold

/-- Theorem stating that Mrs. Sheridan has 46 fish left -/
theorem sheridan_fish_count : fish_remaining 22 47 15 8 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_fish_count_l2203_220363


namespace NUMINAMATH_CALUDE_modified_cube_edges_l2203_220332

/-- Represents a cube with a given side length. -/
structure Cube where
  sideLength : ℕ

/-- Represents the structure after removing unit cubes from corners. -/
structure ModifiedCube where
  originalCube : Cube
  removedCubeSize : ℕ

/-- Calculates the number of edges in the modified cube structure. -/
def edgesInModifiedCube (mc : ModifiedCube) : ℕ :=
  12 * 3  -- Each original edge is divided into 3 segments

/-- Theorem stating that a cube of side length 4 with unit cubes removed from corners has 36 edges. -/
theorem modified_cube_edges :
  ∀ (mc : ModifiedCube),
    mc.originalCube.sideLength = 4 →
    mc.removedCubeSize = 1 →
    edgesInModifiedCube mc = 36 := by
  sorry


end NUMINAMATH_CALUDE_modified_cube_edges_l2203_220332


namespace NUMINAMATH_CALUDE_peaches_per_basket_l2203_220345

/-- The number of red peaches in each basket -/
def red_peaches : ℕ := 7

/-- The number of green peaches in each basket -/
def green_peaches : ℕ := 3

/-- The total number of peaches in each basket -/
def total_peaches : ℕ := red_peaches + green_peaches

theorem peaches_per_basket : total_peaches = 10 := by
  sorry

end NUMINAMATH_CALUDE_peaches_per_basket_l2203_220345


namespace NUMINAMATH_CALUDE_cloth_cost_calculation_l2203_220379

/-- The total cost of cloth given its length and price per meter -/
def totalCost (length : ℝ) (pricePerMeter : ℝ) : ℝ :=
  length * pricePerMeter

/-- Theorem: The total cost of 9.25 meters of cloth at $47 per meter is $434.75 -/
theorem cloth_cost_calculation :
  totalCost 9.25 47 = 434.75 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_calculation_l2203_220379


namespace NUMINAMATH_CALUDE_josh_initial_marbles_l2203_220366

/-- The number of marbles Josh found -/
def marbles_found : ℕ := 7

/-- The current total number of marbles Josh has -/
def current_total : ℕ := 28

/-- The initial number of marbles in Josh's collection -/
def initial_marbles : ℕ := current_total - marbles_found

theorem josh_initial_marbles :
  initial_marbles = 21 := by sorry

end NUMINAMATH_CALUDE_josh_initial_marbles_l2203_220366


namespace NUMINAMATH_CALUDE_three_integers_ratio_l2203_220326

theorem three_integers_ratio : ∀ (a b c : ℤ),
  (a : ℚ) / b = 2 / 5 ∧ 
  (b : ℚ) / c = 5 / 8 ∧ 
  ((a + 6 : ℚ) / b = 1 / 3) →
  a = 36 ∧ b = 90 ∧ c = 144 :=
by sorry

end NUMINAMATH_CALUDE_three_integers_ratio_l2203_220326


namespace NUMINAMATH_CALUDE_lcm_of_8_and_12_l2203_220301

theorem lcm_of_8_and_12 :
  let a : ℕ := 8
  let b : ℕ := 12
  let hcf : ℕ := 4
  (Nat.gcd a b = hcf) → (Nat.lcm a b = 24) :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_8_and_12_l2203_220301


namespace NUMINAMATH_CALUDE_solve_system_l2203_220382

theorem solve_system (a b : ℝ) 
  (eq1 : a * (a - 4) = 5)
  (eq2 : b * (b - 4) = 5)
  (neq : a ≠ b)
  (sum : a + b = 4) :
  a = -1 := by sorry

end NUMINAMATH_CALUDE_solve_system_l2203_220382


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2203_220394

theorem nested_fraction_equality : 2 - (1 / (2 - (1 / (2 + 2)))) = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2203_220394


namespace NUMINAMATH_CALUDE_a_minus_b_value_l2203_220368

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 8) (h2 : |b| = 6) (h3 : a * b < 0) :
  a - b = 14 ∨ a - b = -14 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l2203_220368


namespace NUMINAMATH_CALUDE_dave_ticket_problem_l2203_220349

theorem dave_ticket_problem (total_used : ℕ) (difference : ℕ) 
  (h1 : total_used = 12) (h2 : difference = 5) : 
  ∃ (clothes_tickets : ℕ), 
    clothes_tickets + (clothes_tickets + difference) = total_used ∧ 
    clothes_tickets = 7 := by
  sorry

end NUMINAMATH_CALUDE_dave_ticket_problem_l2203_220349


namespace NUMINAMATH_CALUDE_lateral_surface_area_is_four_l2203_220390

/-- A regular quadrilateral pyramid inscribed in a unit sphere -/
structure RegularQuadPyramid where
  /-- The radius of the sphere in which the pyramid is inscribed -/
  radius : ℝ
  /-- The dihedral angle at the apex of the pyramid in radians -/
  dihedral_angle : ℝ
  /-- Assertion that the radius is 1 -/
  radius_is_one : radius = 1
  /-- Assertion that the dihedral angle is π/4 (45 degrees) -/
  angle_is_45 : dihedral_angle = Real.pi / 4

/-- The lateral surface area of a regular quadrilateral pyramid -/
def lateral_surface_area (p : RegularQuadPyramid) : ℝ :=
  sorry

/-- Theorem: The lateral surface area of the specified pyramid is 4 -/
theorem lateral_surface_area_is_four (p : RegularQuadPyramid) :
  lateral_surface_area p = 4 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_is_four_l2203_220390


namespace NUMINAMATH_CALUDE_system_solution_l2203_220338

/- Define the system of equations -/
def equation1 (x y : ℚ) : Prop := 4 * x - 7 * y = -14
def equation2 (x y : ℚ) : Prop := 5 * x + 3 * y = -7

/- Define the solution -/
def solution_x : ℚ := -91/47
def solution_y : ℚ := -42/47

/- Theorem statement -/
theorem system_solution :
  equation1 solution_x solution_y ∧ equation2 solution_x solution_y :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2203_220338


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2203_220343

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo (-3 : ℝ) 2 = {x : ℝ | a * x^2 - 5*x + b > 0}) :
  {x : ℝ | b * x^2 - 5*x + a > 0} = Set.Iic (-1/3 : ℝ) ∪ Set.Ici (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2203_220343


namespace NUMINAMATH_CALUDE_prime_factor_difference_l2203_220397

theorem prime_factor_difference (n : Nat) (h : n = 173459) :
  ∃ (p₁ p₂ p₃ p₄ : Nat),
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧
    n = p₁ * p₂ * p₃ * p₄ ∧
    p₁ ≤ p₂ ∧ p₂ ≤ p₃ ∧ p₃ ≤ p₄ ∧
    p₄ - p₂ = 144 :=
by sorry

end NUMINAMATH_CALUDE_prime_factor_difference_l2203_220397


namespace NUMINAMATH_CALUDE_unique_steakmaker_pair_l2203_220398

/-- A pair of positive integers (m,n) is 'steakmaker' if 1 + 2^m = n^2 -/
def is_steakmaker (m n : ℕ+) : Prop := 1 + 2^(m.val) = n.val^2

theorem unique_steakmaker_pair :
  ∃! (m n : ℕ+), is_steakmaker m n ∧ m.val * n.val = 9 :=
sorry

#check unique_steakmaker_pair

end NUMINAMATH_CALUDE_unique_steakmaker_pair_l2203_220398


namespace NUMINAMATH_CALUDE_other_number_proof_l2203_220369

theorem other_number_proof (a b : ℕ+) 
  (hcf : Nat.gcd a b = 14)
  (lcm : Nat.lcm a b = 396)
  (ha : a = 36) :
  b = 66 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l2203_220369


namespace NUMINAMATH_CALUDE_equation_proof_l2203_220351

theorem equation_proof : 16 * 0.2 * 5 * 0.5 / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2203_220351


namespace NUMINAMATH_CALUDE_cricket_team_size_l2203_220375

/-- Represents the number of players on a cricket team -/
def total_players : ℕ := 61

/-- Represents the number of throwers on the team -/
def throwers : ℕ := 37

/-- Represents the number of right-handed players on the team -/
def right_handed : ℕ := 53

/-- Theorem stating that the total number of players is 61 -/
theorem cricket_team_size :
  total_players = throwers + (right_handed - throwers) * 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_size_l2203_220375


namespace NUMINAMATH_CALUDE_billboard_perimeter_l2203_220341

/-- A rectangular billboard with given area and width has a specific perimeter -/
theorem billboard_perimeter (area : ℝ) (width : ℝ) (h1 : area = 117) (h2 : width = 9) :
  2 * (area / width) + 2 * width = 44 := by
  sorry

#check billboard_perimeter

end NUMINAMATH_CALUDE_billboard_perimeter_l2203_220341


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l2203_220333

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (y : ℝ), (3 - 4*I) * (a + b*I) = y*I) : a/b = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l2203_220333


namespace NUMINAMATH_CALUDE_insect_count_proof_l2203_220304

/-- Calculates the number of insects given the total number of legs and legs per insect -/
def number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Proves that given 48 insect legs and 6 legs per insect, the number of insects is 8 -/
theorem insect_count_proof :
  let total_legs : ℕ := 48
  let legs_per_insect : ℕ := 6
  number_of_insects total_legs legs_per_insect = 8 := by
  sorry

end NUMINAMATH_CALUDE_insect_count_proof_l2203_220304


namespace NUMINAMATH_CALUDE_june_bike_ride_l2203_220312

/-- Given that June rides her bike at a constant rate and travels 2 miles in 6 minutes,
    prove that she will travel 5 miles in 15 minutes. -/
theorem june_bike_ride (rate : ℚ) : 
  (2 : ℚ) / (6 : ℚ) = rate → (5 : ℚ) / rate = (15 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_june_bike_ride_l2203_220312


namespace NUMINAMATH_CALUDE_running_time_difference_l2203_220322

/-- The difference in running time between two runners -/
theorem running_time_difference
  (d : ℝ) -- Total distance
  (lawrence_distance : ℝ) -- Lawrence's running distance
  (lawrence_speed : ℝ) -- Lawrence's speed in minutes per kilometer
  (george_distance : ℝ) -- George's running distance
  (george_speed : ℝ) -- George's speed in minutes per kilometer
  (h1 : lawrence_distance = d / 2)
  (h2 : george_distance = d / 2)
  (h3 : lawrence_speed = 8)
  (h4 : george_speed = 12) :
  george_distance * george_speed - lawrence_distance * lawrence_speed = 2 * d :=
by sorry

end NUMINAMATH_CALUDE_running_time_difference_l2203_220322


namespace NUMINAMATH_CALUDE_sum_of_parts_l2203_220340

/-- Given a number 24 divided into two parts, where the first part is 13.0,
    prove that the sum of 7 times the first part and 5 times the second part is 146. -/
theorem sum_of_parts (first_part second_part : ℝ) : 
  first_part + second_part = 24 →
  first_part = 13 →
  7 * first_part + 5 * second_part = 146 := by
sorry

end NUMINAMATH_CALUDE_sum_of_parts_l2203_220340


namespace NUMINAMATH_CALUDE_cos_pi_sixth_plus_alpha_l2203_220306

theorem cos_pi_sixth_plus_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 4) :
  Real.cos (π / 6 + α) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_plus_alpha_l2203_220306


namespace NUMINAMATH_CALUDE_copy_pages_theorem_l2203_220331

/-- The cost in cents to copy one page -/
def cost_per_page : ℕ := 3

/-- The budget in dollars -/
def budget : ℕ := 15

/-- The maximum number of pages that can be copied -/
def max_pages : ℕ := budget * 100 / cost_per_page

theorem copy_pages_theorem : max_pages = 500 := by
  sorry

end NUMINAMATH_CALUDE_copy_pages_theorem_l2203_220331


namespace NUMINAMATH_CALUDE_inequality_proof_l2203_220348

theorem inequality_proof (x y z : ℝ) : x^2 + y^2 + z^2 ≥ Real.sqrt 2 * (x*y + y*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2203_220348


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2203_220318

def binomialCoefficient (n k : ℕ) : ℕ := sorry

def constantTermInExpansion (n : ℕ) : ℤ :=
  (binomialCoefficient n (n - 2)) * ((-2) ^ 2)

theorem constant_term_expansion :
  constantTermInExpansion 6 = 60 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2203_220318


namespace NUMINAMATH_CALUDE_radius_of_larger_circle_l2203_220334

/-- Two concentric circles with radii r and R, where R = 4r -/
structure ConcentricCircles where
  r : ℝ
  R : ℝ
  h : R = 4 * r

/-- A chord BC tangent to the inner circle -/
structure TangentChord (c : ConcentricCircles) where
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Diameter AC of the larger circle -/
structure Diameter (c : ConcentricCircles) where
  A : ℝ × ℝ
  C : ℝ × ℝ
  h : dist A C = 2 * c.R

theorem radius_of_larger_circle 
  (c : ConcentricCircles) 
  (d : Diameter c) 
  (t : TangentChord c) 
  (h : dist d.A t.B = 8) : 
  c.R = 16 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_larger_circle_l2203_220334


namespace NUMINAMATH_CALUDE_product_plus_one_is_perfect_square_l2203_220389

theorem product_plus_one_is_perfect_square (n m : ℤ) : 
  m - n = 2 → ∃ k : ℤ, n * m + 1 = k^2 := by sorry

end NUMINAMATH_CALUDE_product_plus_one_is_perfect_square_l2203_220389


namespace NUMINAMATH_CALUDE_lily_to_rose_ratio_l2203_220313

def number_of_roses : ℕ := 20
def cost_of_rose : ℕ := 5
def total_spent : ℕ := 250

theorem lily_to_rose_ratio :
  let cost_of_lily : ℕ := 2 * cost_of_rose
  let total_spent_on_roses : ℕ := number_of_roses * cost_of_rose
  let total_spent_on_lilies : ℕ := total_spent - total_spent_on_roses
  let number_of_lilies : ℕ := total_spent_on_lilies / cost_of_lily
  (number_of_lilies : ℚ) / (number_of_roses : ℚ) = 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_lily_to_rose_ratio_l2203_220313


namespace NUMINAMATH_CALUDE_smallest_n_for_sum_equation_l2203_220386

theorem smallest_n_for_sum_equation : ∃ (n : ℕ), n = 835 ∧ 
  (∀ (S : Finset ℕ), S.card = n → (∀ x ∈ S, x ≥ 1 ∧ x ≤ 999) → 
    (∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      a + 2*b + 3*c = d)) ∧
  (∀ (m : ℕ), m < n → 
    ∃ (T : Finset ℕ), T.card = m ∧ (∀ x ∈ T, x ≥ 1 ∧ x ≤ 999) ∧
      ¬(∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧ 
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
        a + 2*b + 3*c = d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_sum_equation_l2203_220386


namespace NUMINAMATH_CALUDE_money_difference_specific_money_difference_l2203_220300

/-- The difference between Dave's and Derek's remaining money after expenses -/
theorem money_difference (derek_initial : ℕ) (derek_lunch1 : ℕ) (derek_lunch_dad : ℕ) (derek_lunch2 : ℕ)
                         (dave_initial : ℕ) (dave_lunch_mom : ℕ) : ℕ :=
  let derek_spent := derek_lunch1 + derek_lunch_dad + derek_lunch2
  let derek_remaining := derek_initial - derek_spent
  let dave_remaining := dave_initial - dave_lunch_mom
  dave_remaining - derek_remaining

/-- Proof of the specific problem -/
theorem specific_money_difference :
  money_difference 40 14 11 5 50 7 = 33 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_specific_money_difference_l2203_220300


namespace NUMINAMATH_CALUDE_m_range_proof_l2203_220328

theorem m_range_proof (m : ℝ) : 
  (∀ x : ℝ, (4 * x - m < 0 → -1 ≤ x ∧ x ≤ 2) ∧ 
  (∃ y : ℝ, -1 ≤ y ∧ y ≤ 2 ∧ ¬(4 * y - m < 0))) → 
  m > 8 := by
sorry

end NUMINAMATH_CALUDE_m_range_proof_l2203_220328


namespace NUMINAMATH_CALUDE_phase_shift_cosine_l2203_220339

theorem phase_shift_cosine (x : Real) :
  let f : Real → Real := fun x ↦ 5 * Real.cos (x - π/3 + π/6)
  (∃ (k : Real), ∀ x, f x = 5 * Real.cos (x - k)) ∧
  (∀ k : Real, (∀ x, f x = 5 * Real.cos (x - k)) → k = π/6) :=
by sorry

end NUMINAMATH_CALUDE_phase_shift_cosine_l2203_220339


namespace NUMINAMATH_CALUDE_inverse_proportional_problem_l2203_220374

/-- Given that x and y are inversely proportional, x + y = 36, and x - y = 12,
    prove that when x = 8, y = 36. -/
theorem inverse_proportional_problem (x y : ℝ) (k : ℝ) 
  (h_inverse : x * y = k)
  (h_sum : x + y = 36)
  (h_diff : x - y = 12)
  (h_x : x = 8) : 
  y = 36 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportional_problem_l2203_220374


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l2203_220302

theorem least_number_with_remainder (n : ℕ) : n = 184 ↔
  n > 0 ∧
  n % 5 = 4 ∧
  n % 9 = 4 ∧
  n % 12 = 4 ∧
  n % 18 = 4 ∧
  ∀ m : ℕ, m > 0 →
    m % 5 = 4 →
    m % 9 = 4 →
    m % 12 = 4 →
    m % 18 = 4 →
    n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l2203_220302


namespace NUMINAMATH_CALUDE_circle_M_fixed_point_l2203_220365

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - Real.sqrt 3)^2 = 4

-- Define the curve on which the center of M lies
def center_curve (x y : ℝ) : Prop :=
  y = Real.sqrt 3 / x

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  y = -Real.sqrt 3 / 3 * x + 4

-- Define the line y = √3
def line_sqrt3 (x y : ℝ) : Prop :=
  y = Real.sqrt 3

-- Define the line x = 5
def line_x5 (x : ℝ) : Prop :=
  x = 5

-- Theorem statement
theorem circle_M_fixed_point :
  ∀ (O C D E F G H P : ℝ × ℝ),
    (O = (0, 0)) →
    (circle_M O.1 O.2) →
    (∃ (cx cy : ℝ), center_curve cx cy ∧ circle_M cx cy) →
    (line_l C.1 C.2) ∧ (line_l D.1 D.2) →
    (circle_M C.1 C.2) ∧ (circle_M D.1 D.2) →
    (Real.sqrt ((C.1 - O.1)^2 + (C.2 - O.2)^2) = Real.sqrt ((D.1 - O.1)^2 + (D.2 - O.2)^2)) →
    (line_sqrt3 E.1 E.2) ∧ (line_sqrt3 F.1 F.2) →
    (circle_M E.1 E.2) ∧ (circle_M F.1 F.2) →
    (line_x5 P.1) →
    (∃ (k b : ℝ), G.2 = k * G.1 + b ∧ H.2 = k * H.1 + b) →
    (circle_M G.1 G.2) ∧ (circle_M H.1 H.2) →
    (∃ (m : ℝ), G.2 - E.2 = m * (G.1 - E.1) ∧ G.2 - P.2 = m * (G.1 - P.1)) →
    (∃ (n : ℝ), H.2 - F.2 = n * (H.1 - F.1) ∧ H.2 - P.2 = n * (H.1 - P.1)) →
    (((E.1 < G.1 ∧ G.1 < F.1) ∧ (F.1 < H.1 ∨ H.1 < E.1)) ∨
     ((E.1 < H.1 ∧ H.1 < F.1) ∧ (F.1 < G.1 ∨ G.1 < E.1))) →
    ∃ (k b : ℝ), G.2 = k * G.1 + b ∧ H.2 = k * H.1 + b ∧ 2 = k * 2 + b ∧ Real.sqrt 3 = k * 2 + b :=
by sorry

end NUMINAMATH_CALUDE_circle_M_fixed_point_l2203_220365


namespace NUMINAMATH_CALUDE_sum_of_n_terms_l2203_220346

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n -/
def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

/-- S_1, S_3, and S_2 form an arithmetic sequence -/
def S_arithmetic (S : ℕ → ℝ) : Prop :=
  S 3 - S 2 = S 2 - S 1

/-- a_1 - a_3 = 3 -/
def a_difference (a : ℕ → ℝ) : Prop :=
  a 1 - a 3 = 3

/-- Theorem: Sum of first n terms of the sequence -/
theorem sum_of_n_terms
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : arithmetic_sequence a S)
  (h2 : S_arithmetic S)
  (h3 : a_difference a) :
  ∀ n, S n = (8/3) * (1 - (-1/2)^n) :=
sorry

end NUMINAMATH_CALUDE_sum_of_n_terms_l2203_220346


namespace NUMINAMATH_CALUDE_lila_seventh_l2203_220357

/-- Represents the finishing position of a racer -/
def Position : Type := Fin 12

/-- Represents a racer in the race -/
structure Racer :=
  (name : String)
  (position : Position)

/-- The race with given conditions -/
structure Race :=
  (racers : List Racer)
  (jessica_behind_esther : ∃ (j e : Racer), j.name = "Jessica" ∧ e.name = "Esther" ∧ j.position.val = e.position.val + 7)
  (ivan_behind_noel : ∃ (i n : Racer), i.name = "Ivan" ∧ n.name = "Noel" ∧ i.position.val = n.position.val + 2)
  (lila_behind_esther : ∃ (l e : Racer), l.name = "Lila" ∧ e.name = "Esther" ∧ l.position.val = e.position.val + 4)
  (noel_behind_omar : ∃ (n o : Racer), n.name = "Noel" ∧ o.name = "Omar" ∧ n.position.val = o.position.val + 4)
  (omar_behind_esther : ∃ (o e : Racer), o.name = "Omar" ∧ e.name = "Esther" ∧ o.position.val = e.position.val + 3)
  (ivan_fourth : ∃ (i : Racer), i.name = "Ivan" ∧ i.position.val = 4)

/-- Theorem stating that Lila finished in 7th place -/
theorem lila_seventh (race : Race) : ∃ (l : Racer), l.name = "Lila" ∧ l.position.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_lila_seventh_l2203_220357


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l2203_220336

theorem bernoulli_inequality (n : ℕ+) (x : ℝ) (h : x > -1) :
  (1 + x)^(n : ℝ) ≥ 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l2203_220336


namespace NUMINAMATH_CALUDE_odd_function_range_l2203_220355

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def range (f : ℝ → ℝ) : Set ℝ := {y | ∃ x, f x = y}

theorem odd_function_range (f : ℝ → ℝ) (h_odd : is_odd f) (h_pos : ∀ x > 0, f x = 2) :
  range f = {-2, 0, 2} := by
  sorry

end NUMINAMATH_CALUDE_odd_function_range_l2203_220355


namespace NUMINAMATH_CALUDE_remaining_cooking_time_l2203_220395

def total_potatoes : ℕ := 15
def cooked_potatoes : ℕ := 8
def cooking_time_per_potato : ℕ := 9

theorem remaining_cooking_time : 
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 63 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cooking_time_l2203_220395


namespace NUMINAMATH_CALUDE_initial_men_count_l2203_220396

/-- Represents the amount of food consumed by one person in one day -/
def FoodPerPersonPerDay : ℝ := 1

/-- Calculates the total amount of food -/
def TotalFood (initialMen : ℕ) : ℝ := initialMen * 22 * FoodPerPersonPerDay

/-- Calculates the amount of food consumed in the first two days -/
def FoodConsumedInTwoDays (initialMen : ℕ) : ℝ := initialMen * 2 * FoodPerPersonPerDay

/-- Calculates the remaining food after two days -/
def RemainingFood (initialMen : ℕ) : ℝ := TotalFood initialMen - FoodConsumedInTwoDays initialMen

theorem initial_men_count (initialMen : ℕ) : 
  TotalFood initialMen = initialMen * 22 * FoodPerPersonPerDay ∧
  RemainingFood initialMen = (initialMen + 760) * 10 * FoodPerPersonPerDay →
  initialMen = 760 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l2203_220396


namespace NUMINAMATH_CALUDE_symposium_partition_exists_l2203_220359

/-- Represents a symposium with delegates and their acquaintances. -/
structure Symposium where
  delegates : Finset Nat
  acquainted : Nat → Nat → Prop
  acquainted_symmetric : ∀ a b, acquainted a b ↔ acquainted b a
  acquainted_irreflexive : ∀ a, ¬acquainted a a
  has_acquaintance : ∀ a ∈ delegates, ∃ b ∈ delegates, a ≠ b ∧ acquainted a b
  not_all_acquainted : ∀ a ∈ delegates, ∃ b ∈ delegates, a ≠ b ∧ ¬acquainted a b

/-- Represents a partition of delegates into two groups. -/
structure Partition (s : Symposium) where
  group1 : Finset Nat
  group2 : Finset Nat
  covers : group1 ∪ group2 = s.delegates
  disjoint : group1 ∩ group2 = ∅
  nonempty : group1.Nonempty ∧ group2.Nonempty

/-- The main theorem stating that a valid partition exists for any symposium. -/
theorem symposium_partition_exists (s : Symposium) :
  ∃ p : Partition s, ∀ a ∈ s.delegates,
    (a ∈ p.group1 → ∃ b ∈ p.group1, a ≠ b ∧ s.acquainted a b) ∧
    (a ∈ p.group2 → ∃ b ∈ p.group2, a ≠ b ∧ s.acquainted a b) :=
  sorry

end NUMINAMATH_CALUDE_symposium_partition_exists_l2203_220359


namespace NUMINAMATH_CALUDE_probability_two_red_balls_probability_two_red_balls_is_5_22_l2203_220347

/-- The probability of picking two red balls from a bag containing 6 red balls, 4 blue balls, and 2 green balls when 2 balls are picked at random -/
theorem probability_two_red_balls : ℚ :=
  let total_balls : ℕ := 6 + 4 + 2
  let red_balls : ℕ := 6
  let prob_first_red : ℚ := red_balls / total_balls
  let prob_second_red : ℚ := (red_balls - 1) / (total_balls - 1)
  prob_first_red * prob_second_red

/-- Proof that the probability of picking two red balls is 5/22 -/
theorem probability_two_red_balls_is_5_22 : 
  probability_two_red_balls = 5 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_probability_two_red_balls_is_5_22_l2203_220347


namespace NUMINAMATH_CALUDE_math_club_smallest_size_l2203_220380

theorem math_club_smallest_size :
  ∀ (total boys girls : ℕ),
    total = boys + girls →
    girls ≥ 2 →
    boys > (91 : ℝ) / 100 * total →
    total ≥ 23 ∧ ∃ (t b g : ℕ), t = 23 ∧ b + g = t ∧ g ≥ 2 ∧ b > (91 : ℝ) / 100 * t :=
by
  sorry

end NUMINAMATH_CALUDE_math_club_smallest_size_l2203_220380


namespace NUMINAMATH_CALUDE_max_blocks_fit_l2203_220356

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular solid given its dimensions -/
def volume (d : Dimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the box -/
def box : Dimensions :=
  { length := 4, width := 3, height := 3 }

/-- The dimensions of a block -/
def block : Dimensions :=
  { length := 2, width := 3, height := 1 }

/-- The maximum number of blocks that can fit in the box -/
def max_blocks : ℕ :=
  volume box / volume block

theorem max_blocks_fit :
  max_blocks = 6 ∧
  (∀ n : ℕ, n > max_blocks → ¬ (n * volume block ≤ volume box)) :=
by sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l2203_220356


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l2203_220399

def num_flips : ℕ := 10
def num_heads : ℕ := 3
def prob_heads : ℚ := 1/3
def prob_tails : ℚ := 2/3

theorem unfair_coin_probability : 
  (Nat.choose num_flips num_heads : ℚ) * prob_heads ^ num_heads * prob_tails ^ (num_flips - num_heads) = 15360/59049 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l2203_220399


namespace NUMINAMATH_CALUDE_commission_rate_proof_l2203_220362

/-- The commission rate for an agent who earned a commission of 12.50 on sales of 250. -/
theorem commission_rate_proof (commission : ℝ) (sales : ℝ) 
  (h1 : commission = 12.50) (h2 : sales = 250) :
  (commission / sales) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_commission_rate_proof_l2203_220362


namespace NUMINAMATH_CALUDE_reduced_price_is_six_l2203_220319

/-- Represents the price of apples and the quantity that can be purchased -/
structure ApplePricing where
  originalPrice : ℚ
  quantityBefore : ℚ
  quantityAfter : ℚ

/-- Calculates the reduced price per dozen apples -/
def reducedPricePerDozen (ap : ApplePricing) : ℚ :=
  6

/-- Theorem stating the reduced price per dozen apples is 6 rupees -/
theorem reduced_price_is_six (ap : ApplePricing) 
  (h1 : ap.quantityAfter = ap.quantityBefore + 50)
  (h2 : ap.quantityBefore * ap.originalPrice = 50)
  (h3 : ap.quantityAfter * (ap.originalPrice / 2) = 50) : 
  reducedPricePerDozen ap = 6 := by
  sorry

#check reduced_price_is_six

end NUMINAMATH_CALUDE_reduced_price_is_six_l2203_220319


namespace NUMINAMATH_CALUDE_right_triangle_among_given_sets_l2203_220317

-- Define a function to check if three numbers can form a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem statement
theorem right_triangle_among_given_sets :
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 5 6 7) ∧
  ¬(is_right_triangle 5 (-11) 12) ∧
  is_right_triangle 5 12 13 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_among_given_sets_l2203_220317


namespace NUMINAMATH_CALUDE_boat_speed_is_twelve_l2203_220387

/-- Represents the speed of a boat and current in a river --/
structure RiverJourney where
  boat_speed : ℝ
  current_speed : ℝ

/-- Represents the time taken for upstream and downstream journeys --/
structure JourneyTimes where
  upstream_time : ℝ
  downstream_time : ℝ

/-- Checks if the given boat speed is consistent with the journey times --/
def is_consistent_speed (journey : RiverJourney) (times : JourneyTimes) : Prop :=
  (journey.boat_speed - journey.current_speed) * times.upstream_time =
  (journey.boat_speed + journey.current_speed) * times.downstream_time

/-- The main theorem to prove --/
theorem boat_speed_is_twelve (times : JourneyTimes) 
    (h1 : times.upstream_time = 5)
    (h2 : times.downstream_time = 3) :
    ∃ (journey : RiverJourney), 
      journey.boat_speed = 12 ∧ 
      is_consistent_speed journey times := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_is_twelve_l2203_220387


namespace NUMINAMATH_CALUDE_unique_solution_l2203_220342

-- Define the equation
def equation (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ 4 ∧ (3 * x^2 - 12 * x) / (x^2 - 4 * x) = x - 2

-- Theorem statement
theorem unique_solution : ∃! x : ℝ, equation x := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2203_220342


namespace NUMINAMATH_CALUDE_tan_alpha_equals_negative_one_l2203_220310

theorem tan_alpha_equals_negative_one (α : Real) 
  (h1 : |Real.sin α| = |Real.cos α|) 
  (h2 : α > Real.pi / 2 ∧ α < Real.pi) : 
  Real.tan α = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_negative_one_l2203_220310


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_l2203_220320

theorem triangle_cosine_sum (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (b + c = 12) →  -- Given condition
  (C = 2 * π / 3) →  -- 120° in radians
  (Real.sin B = 5 * Real.sqrt 3 / 14) →
  (Real.cos A + Real.cos B = 12 / 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_l2203_220320


namespace NUMINAMATH_CALUDE_quadratic_equations_solution_l2203_220393

theorem quadratic_equations_solution :
  -- Part 1
  (∀ x, 1969 * x^2 - 1974 * x + 5 = 0 ↔ x = 1 ∨ x = 5/1969) ∧
  -- Part 2
  (∀ a b c x,
    -- Case 1
    (a + b - 2*c = 0 ∧ b + c - 2*a ≠ 0 →
      (a + b - 2*c) * x^2 + (b + c - 2*a) * x + (c + a - 2*b) = 0 ↔
      x = -(c + a - 2*b) / (b + c - 2*a)) ∧
    (a + b - 2*c = 0 ∧ b + c - 2*a = 0 →
      (a + b - 2*c) * x^2 + (b + c - 2*a) * x + (c + a - 2*b) = 0 ↔
      True) ∧
    -- Case 2
    (a + b - 2*c ≠ 0 →
      (a + b - 2*c) * x^2 + (b + c - 2*a) * x + (c + a - 2*b) = 0 ↔
      x = 1 ∨ x = (c + a - 2*b) / (a + b - 2*c))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solution_l2203_220393


namespace NUMINAMATH_CALUDE_thabo_hardcover_books_l2203_220383

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (books : BookCollection) : Prop :=
  books.hardcover_nonfiction + books.paperback_nonfiction + books.paperback_fiction = 200 ∧
  books.paperback_nonfiction = books.hardcover_nonfiction + 20 ∧
  books.paperback_fiction = 2 * books.paperback_nonfiction

theorem thabo_hardcover_books :
  ∀ (books : BookCollection), is_valid_collection books → books.hardcover_nonfiction = 35 := by
  sorry

end NUMINAMATH_CALUDE_thabo_hardcover_books_l2203_220383


namespace NUMINAMATH_CALUDE_f_properties_l2203_220305

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + (1 + Real.cos (2 * x)) / 4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (k : ℤ), ∀ (x : ℝ), 
    k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 →
    ∀ (y : ℝ), k * Real.pi - Real.pi / 3 ≤ y ∧ y ≤ x → f y ≤ f x) ∧
  (∀ (A B C : ℝ) (a b c : ℝ),
    f A = 1/2 → b + c = 3 →
    a = Real.sqrt (b^2 + c^2 - 2*b*c*Real.cos A) →
    a ≥ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2203_220305


namespace NUMINAMATH_CALUDE_rectangular_field_dimension_l2203_220370

theorem rectangular_field_dimension (m : ℝ) : ∃! m : ℝ, (3*m + 5)*(m - 1) = 104 ∧ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_dimension_l2203_220370


namespace NUMINAMATH_CALUDE_square_area_increase_l2203_220307

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.25 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.5625 := by
sorry

end NUMINAMATH_CALUDE_square_area_increase_l2203_220307


namespace NUMINAMATH_CALUDE_sum_and_transformations_l2203_220335

theorem sum_and_transformations (x y z M : ℚ) : 
  x + y + z = 72 ∧ 
  x - 9 = M ∧ 
  y + 9 = M ∧ 
  9 * z = M → 
  M = 34 := by
sorry

end NUMINAMATH_CALUDE_sum_and_transformations_l2203_220335


namespace NUMINAMATH_CALUDE_prob_at_least_one_woman_l2203_220303

/-- The probability of selecting at least one woman when randomly choosing 3 people from a group of 5 men and 5 women is 5/6. -/
theorem prob_at_least_one_woman (n_men n_women n_select : ℕ) (h_men : n_men = 5) (h_women : n_women = 5) (h_select : n_select = 3) :
  let total := n_men + n_women
  let prob_no_women := (n_men.choose n_select : ℚ) / (total.choose n_select : ℚ)
  1 - prob_no_women = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_woman_l2203_220303


namespace NUMINAMATH_CALUDE_dice_probability_l2203_220327

/-- The number of sides on each die -/
def num_sides : ℕ := 12

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of one-digit outcomes on each die -/
def one_digit_outcomes : ℕ := 9

/-- The number of two-digit outcomes on each die -/
def two_digit_outcomes : ℕ := num_sides - one_digit_outcomes

/-- The probability of rolling a one-digit number on a single die -/
def prob_one_digit : ℚ := one_digit_outcomes / num_sides

/-- The probability of rolling a two-digit number on a single die -/
def prob_two_digit : ℚ := two_digit_outcomes / num_sides

/-- The number of dice showing one-digit numbers -/
def num_one_digit : ℕ := 3

/-- The number of dice showing two-digit numbers -/
def num_two_digit : ℕ := num_dice - num_one_digit

theorem dice_probability :
  (Nat.choose num_dice num_one_digit : ℚ) *
  (prob_one_digit ^ num_one_digit) *
  (prob_two_digit ^ num_two_digit) =
  135 / 512 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l2203_220327


namespace NUMINAMATH_CALUDE_age_of_replaced_man_is_44_l2203_220325

/-- The age of the other replaced man given the conditions of the problem -/
def age_of_replaced_man (initial_men_count : ℕ) (age_increase : ℕ) (known_man_age : ℕ) (women_avg_age : ℕ) : ℕ :=
  44

/-- Theorem stating that the age of the other replaced man is 44 years old -/
theorem age_of_replaced_man_is_44 
  (initial_men_count : ℕ) 
  (age_increase : ℕ) 
  (known_man_age : ℕ) 
  (women_avg_age : ℕ) 
  (h1 : initial_men_count = 6)
  (h2 : age_increase = 3)
  (h3 : known_man_age = 24)
  (h4 : women_avg_age = 34) :
  age_of_replaced_man initial_men_count age_increase known_man_age women_avg_age = 44 := by
  sorry

end NUMINAMATH_CALUDE_age_of_replaced_man_is_44_l2203_220325


namespace NUMINAMATH_CALUDE_earliest_100_degrees_l2203_220377

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 15*t + 40

-- State the theorem
theorem earliest_100_degrees :
  ∃ t : ℝ, t ≥ 0 ∧ temperature t = 100 ∧ ∀ s, s ≥ 0 ∧ temperature s = 100 → s ≥ t :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_earliest_100_degrees_l2203_220377


namespace NUMINAMATH_CALUDE_partnership_profit_theorem_l2203_220352

/-- Represents the profit distribution in a partnership business -/
structure PartnershipProfit where
  investment_A : ℕ
  investment_B : ℕ
  investment_C : ℕ
  profit_share_C : ℕ

/-- Calculates the total profit of the partnership -/
def total_profit (p : PartnershipProfit) : ℕ :=
  (p.profit_share_C * (p.investment_A + p.investment_B + p.investment_C)) / p.investment_C

/-- Theorem stating that given the investments and C's profit share, the total profit is 80000 -/
theorem partnership_profit_theorem (p : PartnershipProfit) 
  (h1 : p.investment_A = 27000)
  (h2 : p.investment_B = 72000)
  (h3 : p.investment_C = 81000)
  (h4 : p.profit_share_C = 36000) :
  total_profit p = 80000 := by
  sorry

#eval total_profit { investment_A := 27000, investment_B := 72000, investment_C := 81000, profit_share_C := 36000 }

end NUMINAMATH_CALUDE_partnership_profit_theorem_l2203_220352


namespace NUMINAMATH_CALUDE_smallest_integer_y_smallest_solution_l2203_220309

theorem smallest_integer_y (y : ℤ) : (7 - 3 * y ≥ 22) ↔ (y ≤ -5) := by sorry

theorem smallest_solution : ∃ (y : ℤ), (7 - 3 * y ≥ 22) ∧ ∀ (z : ℤ), (7 - 3 * z ≥ 22) → (y ≤ z) := by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_smallest_solution_l2203_220309


namespace NUMINAMATH_CALUDE_count_divisors_5940_mult_6_l2203_220384

/-- The number of positive divisors of 5940 that are multiples of 6 -/
def divisors_5940_mult_6 : ℕ := 24

/-- 5940 expressed as a product of prime factors -/
def factorization_5940 : ℕ := 2^2 * 3^3 * 5 * 11

theorem count_divisors_5940_mult_6 :
  (∀ d : ℕ, d > 0 ∧ d ∣ factorization_5940 ∧ 6 ∣ d) →
  (∃! n : ℕ, n = divisors_5940_mult_6) :=
sorry

end NUMINAMATH_CALUDE_count_divisors_5940_mult_6_l2203_220384


namespace NUMINAMATH_CALUDE_insurance_premium_calculation_l2203_220323

/-- Calculates the new insurance premium after accidents and tickets. -/
theorem insurance_premium_calculation
  (initial_premium : ℝ)
  (accident_increase_percent : ℝ)
  (ticket_increase : ℝ)
  (num_accidents : ℕ)
  (num_tickets : ℕ)
  (h1 : initial_premium = 50)
  (h2 : accident_increase_percent = 0.1)
  (h3 : ticket_increase = 5)
  (h4 : num_accidents = 1)
  (h5 : num_tickets = 3) :
  initial_premium * (1 + num_accidents * accident_increase_percent) + num_tickets * ticket_increase = 70 :=
by sorry


end NUMINAMATH_CALUDE_insurance_premium_calculation_l2203_220323


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l2203_220364

theorem x_squared_plus_reciprocal (x : ℝ) (h : 49 = x^6 + 1/x^6) :
  x^2 + 1/x^2 = (51 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l2203_220364


namespace NUMINAMATH_CALUDE_sotka_not_divisible_by_nine_l2203_220350

/-- Represents a mapping of letters to digits -/
def LetterMapping := Char → Nat

/-- Checks if a number represented by a string is divisible by a given number -/
def isDivisible (s : String) (n : Nat) (mapping : LetterMapping) : Prop :=
  (s.toList.map mapping).sum % n = 0

/-- Ensures that each letter maps to a unique digit between 0 and 9 -/
def isValidMapping (mapping : LetterMapping) : Prop :=
  ∀ c₁ c₂, c₁ ≠ c₂ → mapping c₁ ≠ mapping c₂ ∧ mapping c₁ < 10 ∧ mapping c₂ < 10

theorem sotka_not_divisible_by_nine :
  ∀ mapping : LetterMapping,
    isValidMapping mapping →
    isDivisible "ДЕВЯНОСТО" 90 mapping →
    isDivisible "ДЕВЯТКА" 9 mapping →
    mapping 'О' = 0 →
    ¬ isDivisible "СОТКА" 9 mapping :=
by
  sorry

end NUMINAMATH_CALUDE_sotka_not_divisible_by_nine_l2203_220350


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l2203_220360

/-- Given an ellipse with equation 16(x+2)^2 + 4y^2 = 64, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance :
  ∀ (x y : ℝ), 16 * (x + 2)^2 + 4 * y^2 = 64 →
  ∃ (C D : ℝ × ℝ),
    (C.1 + 2)^2 / 4 + C.2^2 / 16 = 1 ∧
    (D.1 + 2)^2 / 4 + D.2^2 / 16 = 1 ∧
    (C.2 = 0 ∨ C.2 = 0) ∧
    (D.1 = -2 ∨ D.1 = -2) ∧
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l2203_220360


namespace NUMINAMATH_CALUDE_complex_modulus_sum_l2203_220308

theorem complex_modulus_sum : Complex.abs (3 - 5*Complex.I) + Complex.abs (3 + 5*Complex.I) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sum_l2203_220308


namespace NUMINAMATH_CALUDE_rectangle_length_l2203_220315

theorem rectangle_length (s : ℝ) (l : ℝ) : 
  s > 0 → l > 0 →
  s^2 = 5 * (l * 10) →
  4 * s = 200 →
  l = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l2203_220315


namespace NUMINAMATH_CALUDE_lcm_24_36_45_l2203_220358

theorem lcm_24_36_45 : Nat.lcm 24 (Nat.lcm 36 45) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_36_45_l2203_220358


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2203_220324

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  1 / (x + 1) + 4 / (y + 2) ≥ 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2203_220324


namespace NUMINAMATH_CALUDE_prime_square_difference_divisibility_l2203_220330

theorem prime_square_difference_divisibility (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧
    1 < p - a^2 ∧
    p - a^2 < p - b^2 ∧
    (p - b^2) % (p - a^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_difference_divisibility_l2203_220330


namespace NUMINAMATH_CALUDE_tangent_line_at_two_condition_equivalent_to_range_l2203_220388

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + (2 * a - 1) / x + 1 - 3 * a

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y - 4 = 0

-- Theorem for part I
theorem tangent_line_at_two (a : ℝ) (h : a = 1) :
  ∃ y, f a 2 = y ∧ tangent_line 2 y :=
sorry

-- Theorem for part II
theorem condition_equivalent_to_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ (1 - a) * Real.log x) ↔ a ≥ 1/3 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_two_condition_equivalent_to_range_l2203_220388


namespace NUMINAMATH_CALUDE_impossible_cover_all_endings_l2203_220385

theorem impossible_cover_all_endings (a : Fin 14 → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  ¬(∀ d : Fin 100, ∃ k l : Fin 14, (a k + a l) % 100 = d) := by
  sorry

end NUMINAMATH_CALUDE_impossible_cover_all_endings_l2203_220385


namespace NUMINAMATH_CALUDE_percentage_between_55_and_65_l2203_220392

/-- Represents the percentage of students who scored at least 55% on the test -/
def scored_at_least_55 : ℝ := 55

/-- Represents the percentage of students who scored at most 65% on the test -/
def scored_at_most_65 : ℝ := 65

/-- Represents the percentage of students who scored between 55% and 65% (inclusive) on the test -/
def scored_between_55_and_65 : ℝ := scored_at_most_65 - (100 - scored_at_least_55)

theorem percentage_between_55_and_65 : scored_between_55_and_65 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_between_55_and_65_l2203_220392


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_is_one_l2203_220314

theorem fraction_to_zero_power_is_one (a b : ℤ) (hb : b ≠ 0) : (a / b : ℚ) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_is_one_l2203_220314


namespace NUMINAMATH_CALUDE_speed_ratio_after_meeting_l2203_220373

/-- Represents a car with a speed -/
structure Car where
  speed : ℝ

/-- Represents the scenario of two cars meeting -/
structure CarMeeting where
  carA : Car
  carB : Car
  totalDistance : ℝ
  timeToMeet : ℝ
  timeAAfterMeet : ℝ
  timeBAfterMeet : ℝ

/-- The theorem stating the ratio of speeds given the conditions -/
theorem speed_ratio_after_meeting (m : CarMeeting) 
  (h1 : m.timeAAfterMeet = 4)
  (h2 : m.timeBAfterMeet = 1)
  (h3 : m.totalDistance = m.carA.speed * m.timeToMeet + m.carB.speed * m.timeToMeet)
  (h4 : m.carA.speed * m.timeAAfterMeet = m.totalDistance - m.carA.speed * m.timeToMeet)
  (h5 : m.carB.speed * m.timeBAfterMeet = m.totalDistance - m.carB.speed * m.timeToMeet) :
  m.carA.speed / m.carB.speed = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_after_meeting_l2203_220373


namespace NUMINAMATH_CALUDE_class_size_l2203_220391

/-- The number of girls in Tom's class -/
def girls : ℕ := 22

/-- The difference between the number of girls and boys in Tom's class -/
def difference : ℕ := 3

/-- The total number of students in Tom's class -/
def total_students : ℕ := girls + (girls - difference)

theorem class_size : total_students = 41 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2203_220391


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2203_220372

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 4, 5}
def N : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ N) ∩ M = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2203_220372


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2203_220367

/-- Given a geometric sequence {a_n} with first term a and common ratio q,
    if a_2 * a_5 = 2 * a_3 and (a_4 + 2 * a_7) / 2 = 5/4,
    then the sum of the first 5 terms (S_5) is equal to 31. -/
theorem geometric_sequence_sum (a q : ℝ) : 
  (a * q * (a * q^4) = 2 * (a * q^2)) →
  ((a * q^3 + 2 * (a * q^6)) / 2 = 5/4) →
  (a * (1 - q^5)) / (1 - q) = 31 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2203_220367


namespace NUMINAMATH_CALUDE_calculation_proofs_l2203_220337

theorem calculation_proofs :
  (2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / Real.sqrt 2 = (3 * Real.sqrt 2) / 2) ∧
  ((Real.sqrt 3 - Real.sqrt 2)^2 + (Real.sqrt 8 - Real.sqrt 3) * (2 * Real.sqrt 2 + Real.sqrt 3) = 10 - 2 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l2203_220337


namespace NUMINAMATH_CALUDE_digit_sum_unbounded_l2203_220344

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sequence of sum of digits of a^n -/
def digitSumSequence (a : ℕ) (n : ℕ) : ℕ := sumOfDigits (a^n)

theorem digit_sum_unbounded (a : ℕ) (h1 : Even a) (h2 : ¬(5 ∣ a)) :
  ∀ M : ℕ, ∃ N : ℕ, ∀ n ≥ N, digitSumSequence a n > M :=
sorry

end NUMINAMATH_CALUDE_digit_sum_unbounded_l2203_220344


namespace NUMINAMATH_CALUDE_symmetric_point_of_A_l2203_220354

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin in 3D Cartesian space -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Point A with coordinates (1, 1, 2) -/
def pointA : Point3D := ⟨1, 1, 2⟩

/-- A point is symmetric to another point with respect to the origin if the origin is the midpoint of the line segment connecting the two points -/
def isSymmetricWrtOrigin (p q : Point3D) : Prop :=
  origin.x = (p.x + q.x) / 2 ∧
  origin.y = (p.y + q.y) / 2 ∧
  origin.z = (p.z + q.z) / 2

/-- The theorem stating that the point symmetric to A(1, 1, 2) with respect to the origin has coordinates (-1, -1, -2) -/
theorem symmetric_point_of_A :
  ∃ (B : Point3D), isSymmetricWrtOrigin pointA B ∧ B = ⟨-1, -1, -2⟩ :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_of_A_l2203_220354


namespace NUMINAMATH_CALUDE_sum_of_ratios_geq_six_l2203_220316

theorem sum_of_ratios_geq_six {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x / y + y / z + z / x + y / x + z / y + x / z ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_geq_six_l2203_220316


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l2203_220311

theorem cubic_inequality_solution (x : ℝ) :
  x^3 - 9*x^2 + 23*x - 15 < 0 ↔ x ∈ Set.Iio 1 ∪ Set.Ioo 3 5 :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l2203_220311


namespace NUMINAMATH_CALUDE_not_enough_unique_names_l2203_220353

/-- Represents the number of possible occurrences of each letter (a, o, u) in a standardized word -/
def letter_choices : ℕ := 7

/-- Represents the total number of tribe members -/
def tribe_members : ℕ := 400

/-- Represents the number of unique standardized words in the Mumbo-Jumbo language -/
def unique_words : ℕ := letter_choices ^ 3

theorem not_enough_unique_names : unique_words < tribe_members := by
  sorry

end NUMINAMATH_CALUDE_not_enough_unique_names_l2203_220353


namespace NUMINAMATH_CALUDE_farmer_land_area_l2203_220376

theorem farmer_land_area (A : ℝ) 
  (h1 : 0.9 * A * 0.1 = 360) 
  (h2 : 0.9 * A * 0.6 + 0.9 * A * 0.3 + 360 = 0.9 * A) : 
  A = 4000 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_area_l2203_220376


namespace NUMINAMATH_CALUDE_tree_spacing_l2203_220321

/-- Given a yard of length 225 meters with 26 trees planted at equal distances,
    including one tree at each end, the distance between two consecutive trees is 9 meters. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (tree_spacing : ℝ) : 
  yard_length = 225 →
  num_trees = 26 →
  tree_spacing * (num_trees - 1) = yard_length →
  tree_spacing = 9 := by sorry

end NUMINAMATH_CALUDE_tree_spacing_l2203_220321
