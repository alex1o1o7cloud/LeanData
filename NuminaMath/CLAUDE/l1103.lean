import Mathlib

namespace complete_square_quadratic_l1103_110392

/-- Given a quadratic equation x^2 + 8x - 1 = 0, when written in the form (x + a)^2 = b, b equals 17 -/
theorem complete_square_quadratic : 
  ∃ a : ℝ, ∀ x : ℝ, (x^2 + 8*x - 1 = 0) ↔ ((x + a)^2 = 17) :=
by sorry

end complete_square_quadratic_l1103_110392


namespace identify_counterfeit_l1103_110335

/-- Represents a coin with its denomination and weight -/
structure Coin where
  denomination : Nat
  weight : Nat

/-- Represents the state of a balance scale -/
inductive Balance
  | Left
  | Right
  | Equal

/-- Represents a weighing operation on the balance scale -/
def weigh (left right : List Coin) : Balance :=
  sorry

/-- Represents the set of coins -/
def coins : List Coin :=
  [⟨1, 1⟩, ⟨2, 2⟩, ⟨3, 3⟩, ⟨5, 5⟩]

/-- Represents the counterfeit coin -/
def counterfeit : Coin :=
  sorry

/-- The main theorem stating that the counterfeit coin can be identified in two weighings -/
theorem identify_counterfeit :
  ∃ (weighing1 weighing2 : List Coin × List Coin),
    let result1 := weigh weighing1.1 weighing1.2
    let result2 := weigh weighing2.1 weighing2.2
    ∃ (identified : Coin), identified = counterfeit :=
  sorry

end identify_counterfeit_l1103_110335


namespace chairs_to_remove_l1103_110306

/-- Given a conference hall setup with the following conditions:
  - Each row has 15 chairs
  - Initially, there are 195 chairs
  - 120 attendees are expected
  - All rows must be complete
  - The number of remaining chairs must be the smallest multiple of 15 that is greater than or equal to 120
  
  This theorem proves that the number of chairs to be removed is 60. -/
theorem chairs_to_remove (chairs_per_row : ℕ) (initial_chairs : ℕ) (expected_attendees : ℕ)
  (h1 : chairs_per_row = 15)
  (h2 : initial_chairs = 195)
  (h3 : expected_attendees = 120)
  (h4 : ∃ (n : ℕ), n * chairs_per_row ≥ expected_attendees ∧
        ∀ (m : ℕ), m * chairs_per_row ≥ expected_attendees → n ≤ m) :
  initial_chairs - (chairs_per_row * (initial_chairs / chairs_per_row)) = 60 :=
sorry

end chairs_to_remove_l1103_110306


namespace bella_stamps_count_l1103_110331

/-- Represents the number of stamps of each type Bella bought -/
structure StampCounts where
  snowflake : ℕ
  truck : ℕ
  rose : ℕ
  butterfly : ℕ

/-- Calculates the total number of stamps bought -/
def totalStamps (counts : StampCounts) : ℕ :=
  counts.snowflake + counts.truck + counts.rose + counts.butterfly

/-- Theorem stating the total number of stamps Bella bought -/
theorem bella_stamps_count : ∃ (counts : StampCounts),
  (counts.snowflake : ℚ) * (105 / 100) = 1575 / 100 ∧
  counts.truck = counts.snowflake + 11 ∧
  counts.rose = counts.truck - 17 ∧
  (counts.butterfly : ℚ) = (3 / 2) * counts.rose ∧
  totalStamps counts = 64 := by
  sorry

#check bella_stamps_count

end bella_stamps_count_l1103_110331


namespace rectangular_field_area_l1103_110330

theorem rectangular_field_area (perimeter width length : ℝ) : 
  perimeter = 100 → 
  2 * (length + width) = perimeter → 
  length = 3 * width → 
  length * width = 468.75 := by
  sorry

end rectangular_field_area_l1103_110330


namespace probability_calculations_l1103_110394

/-- Represents the number of students choosing each subject -/
structure SubjectCounts where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  politics : ℕ
  history : ℕ
  geography : ℕ

/-- The total number of students -/
def totalStudents : ℕ := 1000

/-- The actual distribution of students across subjects -/
def actualCounts : SubjectCounts :=
  { physics := 300
  , chemistry := 200
  , biology := 100
  , politics := 200
  , history := 100
  , geography := 100 }

/-- Calculates the probability of an event given the number of favorable outcomes -/
def probability (favorableOutcomes : ℕ) : ℚ :=
  favorableOutcomes / totalStudents

/-- Theorem stating the probabilities of various events -/
theorem probability_calculations (counts : SubjectCounts) 
    (h : counts = actualCounts) : 
    probability counts.chemistry = 1/5 ∧ 
    probability (counts.biology + counts.history) = 1/5 ∧
    probability (counts.chemistry + counts.geography) = 3/10 := by
  sorry


end probability_calculations_l1103_110394


namespace equation_solution_l1103_110321

theorem equation_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 = 2020 * x) ↔
  x = 1 :=
by sorry

end equation_solution_l1103_110321


namespace harvest_difference_l1103_110361

theorem harvest_difference (apples peaches pears : ℕ) : 
  apples = 60 →
  peaches = 3 * apples →
  pears = apples / 2 →
  (apples + peaches) - pears = 210 := by
  sorry

end harvest_difference_l1103_110361


namespace triangle_trig_identity_l1103_110378

theorem triangle_trig_identity (D E F : Real) (DE DF EF : Real) : 
  DE = 7 → DF = 8 → EF = 5 → 
  (Real.cos ((D - E) / 2) / Real.sin (F / 2)) - 
  (Real.sin ((D - E) / 2) / Real.cos (F / 2)) = 16 / 7 := by
  sorry

end triangle_trig_identity_l1103_110378


namespace rectangle_area_increase_l1103_110332

theorem rectangle_area_increase (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let original_area := L * W
  let new_length := 1.2 * L
  let new_width := 1.2 * W
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.44 := by
sorry

end rectangle_area_increase_l1103_110332


namespace quadratic_roots_difference_l1103_110343

theorem quadratic_roots_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * X^2 + b * X + c = 0 → |r₁ - r₂| = 3 :=
by
  sorry

#check quadratic_roots_difference 1 (-7) 10

end quadratic_roots_difference_l1103_110343


namespace z_pure_imaginary_iff_m_eq_2013_l1103_110307

/-- A complex number z is pure imaginary if and only if its real part is zero and its imaginary part is non-zero. -/
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of real number m. -/
def z (m : ℝ) : ℂ :=
  Complex.mk (m - 2013) (m - 1)

/-- Theorem stating that z is pure imaginary if and only if m = 2013. -/
theorem z_pure_imaginary_iff_m_eq_2013 :
    ∀ m : ℝ, is_pure_imaginary (z m) ↔ m = 2013 := by
  sorry

end z_pure_imaginary_iff_m_eq_2013_l1103_110307


namespace sum_with_radical_conjugate_l1103_110377

theorem sum_with_radical_conjugate :
  let x : ℝ := 5 - Real.sqrt 500
  let y : ℝ := 5 + Real.sqrt 500
  x + y = 10 := by sorry

end sum_with_radical_conjugate_l1103_110377


namespace vector_sum_proof_l1103_110344

/-- Given points A, B, and C in ℝ², prove that AC + (1/3)BA = (2, -3) -/
theorem vector_sum_proof (A B C : ℝ × ℝ) 
  (hA : A = (2, 4)) 
  (hB : B = (-1, -5)) 
  (hC : C = (3, -2)) : 
  (C.1 - A.1, C.2 - A.2) + (1/3 * (A.1 - B.1), 1/3 * (A.2 - B.2)) = (2, -3) := by
  sorry

end vector_sum_proof_l1103_110344


namespace sum_of_squares_l1103_110367

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 9)
  (eq2 : y^2 + 5*z = -9)
  (eq3 : z^2 + 7*x = -18) :
  x^2 + y^2 + z^2 = 20.75 := by
sorry

end sum_of_squares_l1103_110367


namespace simplify_expression_l1103_110351

theorem simplify_expression (y : ℝ) : (3*y)^3 - (4*y)*(y^2) = 23*y^3 := by
  sorry

end simplify_expression_l1103_110351


namespace cubic_floor_equation_solution_l1103_110327

theorem cubic_floor_equation_solution :
  ∃! x : ℝ, 3 * x^3 - ⌊x⌋ = 3 :=
by
  -- The unique solution is x = ∛(4/3)
  use Real.rpow (4/3) (1/3)
  sorry

end cubic_floor_equation_solution_l1103_110327


namespace sqrt_81_div_3_l1103_110391

theorem sqrt_81_div_3 : Real.sqrt 81 / 3 = 3 := by
  sorry

end sqrt_81_div_3_l1103_110391


namespace linear_equation_not_proportional_l1103_110317

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0

/-- Direct proportionality between x and y -/
def DirectlyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t, y t = k * x t

/-- Inverse proportionality between x and y -/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t, x t * y t = k

/-- 
For a linear equation ax + by = c, where a ≠ 0 and b ≠ 0,
y is neither directly nor inversely proportional to x
-/
theorem linear_equation_not_proportional (eq : LinearEquation) :
  let x : ℝ → ℝ := λ t => t
  let y : ℝ → ℝ := λ t => (eq.c - eq.a * t) / eq.b
  ¬(DirectlyProportional x y ∨ InverselyProportional x y) := by
  sorry


end linear_equation_not_proportional_l1103_110317


namespace hotel_room_occupancy_l1103_110350

theorem hotel_room_occupancy (num_rooms : ℕ) (towels_per_person : ℕ) (total_towels : ℕ) 
  (h1 : num_rooms = 10)
  (h2 : towels_per_person = 2)
  (h3 : total_towels = 60) :
  total_towels / towels_per_person / num_rooms = 3 := by
  sorry

end hotel_room_occupancy_l1103_110350


namespace line_canonical_equations_l1103_110358

/-- The canonical equations of a line given by the intersection of two planes -/
theorem line_canonical_equations (x y z : ℝ) : 
  (x + 5*y - z + 11 = 0 ∧ x - y + 2*z - 1 = 0) → 
  ((x + 1)/9 = (y + 2)/(-3) ∧ (y + 2)/(-3) = z/(-6)) :=
by sorry

end line_canonical_equations_l1103_110358


namespace bathroom_square_footage_l1103_110352

/-- Calculates the square footage of a bathroom given the number of tiles and tile size. -/
theorem bathroom_square_footage 
  (width_tiles : ℕ) 
  (length_tiles : ℕ) 
  (tile_size_inches : ℕ) 
  (h1 : width_tiles = 10) 
  (h2 : length_tiles = 20) 
  (h3 : tile_size_inches = 6) : 
  (width_tiles * length_tiles * tile_size_inches^2) / 144 = 50 := by
  sorry

#check bathroom_square_footage

end bathroom_square_footage_l1103_110352


namespace rectangle_area_reduction_l1103_110313

theorem rectangle_area_reduction (initial_length initial_width : ℝ)
  (reduced_length reduced_width : ℝ) :
  initial_length = 5 →
  initial_width = 7 →
  reduced_length = initial_length - 2 →
  reduced_width = initial_width - 1 →
  reduced_length * initial_width = 21 →
  reduced_length * reduced_width = 18 :=
by sorry

end rectangle_area_reduction_l1103_110313


namespace inequality_proof_l1103_110365

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≤ 1) : x^6 - y^6 + 2*y^3 < π/2 := by
  sorry

end inequality_proof_l1103_110365


namespace ellipse_triangle_area_l1103_110388

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point on the ellipse
def P : ℝ × ℝ := sorry

-- Assume P is on the ellipse
axiom P_on_ellipse : is_on_ellipse P.1 P.2

-- Define the distances from P to the foci
def PF₁ : ℝ := sorry
def PF₂ : ℝ := sorry

-- Assume the ratio of PF₁ to PF₂ is 2:1
axiom distance_ratio : PF₁ = 2 * PF₂

-- Define the area of the triangle
def triangle_area : ℝ := sorry

-- State the theorem
theorem ellipse_triangle_area : triangle_area = 4 := sorry

end ellipse_triangle_area_l1103_110388


namespace integer_solutions_l1103_110345

def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n

def expression_a (n : ℤ) : ℚ := (n^4 + 3) / (n^2 + n + 1)
def expression_b (n : ℤ) : ℚ := (n^3 + n + 1) / (n^2 - n + 1)

theorem integer_solutions :
  (∀ n : ℤ, is_integer (expression_a n) ↔ n = -3 ∨ n = -1 ∨ n = 0) ∧
  (∀ n : ℤ, is_integer (expression_b n) ↔ n = 0 ∨ n = 1) :=
sorry

end integer_solutions_l1103_110345


namespace total_profit_is_45000_l1103_110349

/-- Represents the total profit earned by Tom and Jose given their investments and Jose's share of profit. -/
def total_profit (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : ℕ :=
  let tom_ratio : ℕ := tom_investment * tom_months
  let jose_ratio : ℕ := jose_investment * jose_months
  let total_ratio : ℕ := tom_ratio + jose_ratio
  (jose_profit * total_ratio) / jose_ratio

/-- Theorem stating that the total profit is 45000 given the specified conditions. -/
theorem total_profit_is_45000 :
  total_profit 30000 12 45000 10 25000 = 45000 := by
  sorry

end total_profit_is_45000_l1103_110349


namespace no_four_primes_product_11_times_sum_l1103_110362

theorem no_four_primes_product_11_times_sum : 
  ¬ ∃ (a b c d : ℕ), 
    Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧
    (a * b * c * d = 11 * (a + b + c + d)) ∧
    ((a + b + c + d = 46) ∨ (a + b + c + d = 47) ∨ (a + b + c + d = 48)) :=
sorry

end no_four_primes_product_11_times_sum_l1103_110362


namespace closest_point_l1103_110396

def v (t : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 2 + 7*t
  | 1 => -3 + 5*t
  | 2 => -3 - t

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 4
  | 1 => 4
  | 2 => 5

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 7
  | 1 => 5
  | 2 => -1

theorem closest_point (t : ℝ) : 
  (∀ s : ℝ, ‖v t - a‖ ≤ ‖v s - a‖) ↔ t = 41/75 :=
sorry

end closest_point_l1103_110396


namespace geometric_sequence_general_term_l1103_110389

/-- A geometric sequence {a_n} satisfying the given conditions has the specified general term. -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence condition
  a 1 + a 3 = 10 →                        -- First given condition
  a 2 + a 4 = 5 →                         -- Second given condition
  ∀ n, a n = 8 * (1/2)^(n - 1) :=         -- Conclusion: general term
by sorry

end geometric_sequence_general_term_l1103_110389


namespace shirt_tie_combinations_l1103_110353

/-- The number of possible shirt-and-tie combinations given a set of shirts and ties with restrictions -/
theorem shirt_tie_combinations (total_shirts : ℕ) (total_ties : ℕ) (restricted_shirts : ℕ) (restricted_ties : ℕ) :
  total_shirts = 8 →
  total_ties = 7 →
  restricted_shirts = 3 →
  restricted_ties = 2 →
  total_shirts * total_ties - restricted_shirts * restricted_ties = 50 := by
sorry

end shirt_tie_combinations_l1103_110353


namespace triangle_area_ratio_origami_triangle_area_ratio_l1103_110301

/-- The ratio of the areas of two triangles with the same base and different heights -/
theorem triangle_area_ratio (base : ℝ) (height1 height2 : ℝ) (h_base : base > 0) 
  (h_height1 : height1 > 0) (h_height2 : height2 > 0) :
  (1 / 2 * base * height1) / (1 / 2 * base * height2) = height1 / height2 := by
  sorry

/-- The specific ratio of triangle areas for the given problem -/
theorem origami_triangle_area_ratio :
  (1 / 2 * 3 * 6.02) / (1 / 2 * 3 * 2) = 3.01 := by
  sorry

end triangle_area_ratio_origami_triangle_area_ratio_l1103_110301


namespace dipolia_puzzle_solution_l1103_110354

-- Define the types of people in Dipolia
inductive PersonType
| Knight
| Liar

-- Define the possible meanings of "Irgo"
inductive IrgoMeaning
| Yes
| No

-- Define the properties of knights and liars
def always_truthful (p : PersonType) : Prop :=
  p = PersonType.Knight

def always_lies (p : PersonType) : Prop :=
  p = PersonType.Liar

-- Define the scenario
structure DipoliaScenario where
  inhabitant_type : PersonType
  irgo_meaning : IrgoMeaning
  guide_truthful : Prop

-- Theorem statement
theorem dipolia_puzzle_solution (scenario : DipoliaScenario) :
  scenario.guide_truthful →
  (scenario.irgo_meaning = IrgoMeaning.Yes ∧ scenario.inhabitant_type = PersonType.Liar) :=
by sorry

end dipolia_puzzle_solution_l1103_110354


namespace simplify_expression_l1103_110382

theorem simplify_expression (x : ℝ) : (2*x + 20) + (150*x + 20) = 152*x + 40 := by
  sorry

end simplify_expression_l1103_110382


namespace ellipse_properties_l1103_110384

/-- Given an ellipse defined by the equation 25x^2 + 9y^2 = 225, 
    this theorem proves its major axis length, minor axis length, and eccentricity. -/
theorem ellipse_properties : ∃ (a b c : ℝ),
  (∀ (x y : ℝ), 25 * x^2 + 9 * y^2 = 225 → x^2 / a^2 + y^2 / b^2 = 1) ∧
  2 * a = 10 ∧
  2 * b = 6 ∧
  c^2 = a^2 - b^2 ∧
  c / a = 0.8 := by
sorry

end ellipse_properties_l1103_110384


namespace games_for_512_players_l1103_110333

/-- A single-elimination tournament with a given number of initial players. -/
structure SingleEliminationTournament where
  initial_players : ℕ
  initial_players_pos : initial_players > 0

/-- The number of games played in a single-elimination tournament. -/
def games_played (t : SingleEliminationTournament) : ℕ :=
  t.initial_players - 1

/-- Theorem stating that a single-elimination tournament with 512 initial players
    requires 511 games to determine the champion. -/
theorem games_for_512_players :
  ∀ (t : SingleEliminationTournament), t.initial_players = 512 → games_played t = 511 := by
  sorry

end games_for_512_players_l1103_110333


namespace repeating_base_k_representation_l1103_110385

/-- Given positive integers m and k, if the repeating base-k representation of 3/28 is 0.121212...₍ₖ₎, then k = 10 -/
theorem repeating_base_k_representation (m k : ℕ+) :
  (∃ (a : ℕ → ℕ), (∀ n, a n < k) ∧
    (∀ n, a (2*n) = 1 ∧ a (2*n+1) = 2) ∧
    (3 : ℚ) / 28 = ∑' n, (a n : ℚ) / k^(n+1)) →
  k = 10 := by sorry

end repeating_base_k_representation_l1103_110385


namespace smallest_mustang_length_l1103_110338

/-- Proves that the smallest model Mustang is 12 inches long given the specified conditions -/
theorem smallest_mustang_length :
  let full_size : ℝ := 240
  let mid_size_ratio : ℝ := 1 / 10
  let smallest_ratio : ℝ := 1 / 2
  let mid_size : ℝ := full_size * mid_size_ratio
  let smallest_size : ℝ := mid_size * smallest_ratio
  smallest_size = 12 := by sorry

end smallest_mustang_length_l1103_110338


namespace plant_growth_probability_l1103_110357

theorem plant_growth_probability (p_1m : ℝ) (p_2m : ℝ) 
  (h1 : p_1m = 0.8) 
  (h2 : p_2m = 0.4) : 
  p_2m / p_1m = 0.5 := by
  sorry

end plant_growth_probability_l1103_110357


namespace congruent_triangles_sum_l1103_110366

/-- A triangle represented by its three side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two triangles are congruent if their corresponding sides are equal -/
def congruent (t1 t2 : Triangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

theorem congruent_triangles_sum (x y : ℝ) :
  let t1 : Triangle := ⟨2, 5, x⟩
  let t2 : Triangle := ⟨y, 2, 6⟩
  congruent t1 t2 → x + y = 11 := by
  sorry

end congruent_triangles_sum_l1103_110366


namespace intersection_of_A_and_B_l1103_110347

def set_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

def set_B : Set ℤ := {x | x^2 - 3*x - 4 < 0}

theorem intersection_of_A_and_B : set_A ∩ (set_B.image (coe : ℤ → ℝ)) = {0, 1, 2} := by
  sorry

end intersection_of_A_and_B_l1103_110347


namespace girl_scout_cookie_sales_l1103_110314

theorem girl_scout_cookie_sales
  (total_boxes : ℕ)
  (total_value : ℚ)
  (choc_chip_price : ℚ)
  (plain_price : ℚ)
  (h1 : total_boxes = 1585)
  (h2 : total_value = 1586.75)
  (h3 : choc_chip_price = 1.25)
  (h4 : plain_price = 0.75) :
  ∃ (plain_boxes : ℕ) (choc_chip_boxes : ℕ),
    plain_boxes + choc_chip_boxes = total_boxes ∧
    plain_price * plain_boxes + choc_chip_price * choc_chip_boxes = total_value ∧
    plain_boxes = 789 :=
by sorry

end girl_scout_cookie_sales_l1103_110314


namespace modular_arithmetic_problem_l1103_110334

theorem modular_arithmetic_problem :
  ∃ (a b : ℕ), 
    (7 * a) % 60 = 1 ∧ 
    (13 * b) % 60 = 1 ∧ 
    ((3 * a + 9 * b) % 60 : ℕ) = 42 := by
  sorry

end modular_arithmetic_problem_l1103_110334


namespace randy_initial_amount_l1103_110320

/-- Represents Randy's piggy bank finances over a year -/
structure PiggyBank where
  initial_amount : ℕ
  monthly_deposit : ℕ
  store_visits : ℕ
  min_cost_per_visit : ℕ
  max_cost_per_visit : ℕ
  final_balance : ℕ

/-- Theorem stating that Randy's initial amount was $104 -/
theorem randy_initial_amount (pb : PiggyBank) 
  (h1 : pb.monthly_deposit = 50)
  (h2 : pb.store_visits = 200)
  (h3 : pb.min_cost_per_visit = 2)
  (h4 : pb.max_cost_per_visit = 3)
  (h5 : pb.final_balance = 104) :
  pb.initial_amount = 104 := by
  sorry

#check randy_initial_amount

end randy_initial_amount_l1103_110320


namespace trigonometric_calculations_l1103_110346

theorem trigonometric_calculations :
  (((Real.pi - 2) ^ 0 - |1 - Real.tan (60 * Real.pi / 180)| - (1/2)⁻¹ + 6 / Real.sqrt 3) = Real.sqrt 3) ∧
  ((Real.sin (45 * Real.pi / 180) - Real.cos (30 * Real.pi / 180) * Real.tan (60 * Real.pi / 180)) = (Real.sqrt 2 - 3) / 2) :=
by sorry

end trigonometric_calculations_l1103_110346


namespace only_solutions_for_equation_l1103_110310

theorem only_solutions_for_equation (x p n : ℕ) : 
  Prime p → 2 * x * (x + 5) = p^n + 3 * (x - 1) → 
  ((x = 2 ∧ p = 5 ∧ n = 2) ∨ (x = 0 ∧ p = 3 ∧ n = 1)) := by
  sorry

end only_solutions_for_equation_l1103_110310


namespace train_length_l1103_110368

/-- The length of a train given its crossing times over a bridge and a lamp post -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (lamp_time : ℝ) 
  (h1 : bridge_length = 1500)
  (h2 : bridge_time = 70)
  (h3 : lamp_time = 20) :
  ∃ (train_length : ℝ), 
    train_length / lamp_time = (train_length + bridge_length) / bridge_time ∧ 
    train_length = 600 := by
  sorry

end train_length_l1103_110368


namespace circumcircle_radius_of_specific_triangle_l1103_110371

/-- The radius of the circumcircle of a triangle with side lengths 8, 15, and 17 is 8.5. -/
theorem circumcircle_radius_of_specific_triangle : 
  ∀ (a b c : ℝ) (r : ℝ),
    a = 8 → b = 15 → c = 17 →
    (a^2 + b^2 = c^2) →  -- right triangle condition
    r = c / 2 →          -- radius is half the hypotenuse
    r = 8.5 := by
  sorry

end circumcircle_radius_of_specific_triangle_l1103_110371


namespace solutions_are_correct_l1103_110329

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 = 49
def equation2 (x : ℝ) : Prop := (2*x + 3)^2 = 4*(2*x + 3)
def equation3 (x : ℝ) : Prop := 2*x^2 + 4*x - 3 = 0
def equation4 (x : ℝ) : Prop := (x + 8)*(x + 1) = -12

-- Theorem stating the solutions are correct
theorem solutions_are_correct :
  (equation1 7 ∧ equation1 (-7)) ∧
  (equation2 (-3/2) ∧ equation2 (1/2)) ∧
  (equation3 ((-2 + Real.sqrt 10) / 2) ∧ equation3 ((-2 - Real.sqrt 10) / 2)) ∧
  (equation4 (-4) ∧ equation4 (-5)) := by sorry

end solutions_are_correct_l1103_110329


namespace inequality_range_l1103_110304

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0) ↔ 
  m > -1/5 ∧ m ≤ 3 := by sorry

end inequality_range_l1103_110304


namespace unit_vector_parallel_l1103_110397

/-- Given two vectors a and b in ℝ², prove that the unit vector parallel to 2a - 3b
    is either (√5/5, 2√5/5) or (-√5/5, -2√5/5) -/
theorem unit_vector_parallel (a b : ℝ × ℝ) (ha : a = (5, 4)) (hb : b = (3, 2)) :
  let v := (2 • a.1 - 3 • b.1, 2 • a.2 - 3 • b.2)
  (v.1 / Real.sqrt (v.1^2 + v.2^2), v.2 / Real.sqrt (v.1^2 + v.2^2)) = (Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5) ∨
  (v.1 / Real.sqrt (v.1^2 + v.2^2), v.2 / Real.sqrt (v.1^2 + v.2^2)) = (-Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5) :=
by sorry

end unit_vector_parallel_l1103_110397


namespace arithmetic_sequence_100th_term_l1103_110373

/-- For an arithmetic sequence {a_n} with first term 1 and common difference 3,
    prove that the 100th term is 298. -/
theorem arithmetic_sequence_100th_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) = a n + 3) →  -- arithmetic sequence with common difference 3
  a 1 = 1 →                    -- first term is 1
  a 100 = 298 := by             -- 100th term is 298
sorry

end arithmetic_sequence_100th_term_l1103_110373


namespace set_inclusion_implies_a_bound_l1103_110315

-- Define set A
def A : Set ℝ := {x : ℝ | (4 : ℝ) / (x + 1) > 1}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*x - a^2 + 2*a < 0}

-- Theorem statement
theorem set_inclusion_implies_a_bound (a : ℝ) (h1 : a < 1) :
  (∀ x : ℝ, x ∈ A → x ∈ B a) → a ≤ -3 := by
  sorry

end set_inclusion_implies_a_bound_l1103_110315


namespace total_pencils_l1103_110369

/-- Given an initial number of pencils and a number of pencils added,
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end total_pencils_l1103_110369


namespace function_value_negation_l1103_110374

/-- Given a function f(x) = a * sin(πx + α) + b * cos(πx + β) where f(2002) = 3,
    prove that f(2003) = -f(2002). -/
theorem function_value_negation (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 2002 = 3 → f 2003 = -f 2002 := by
  sorry

end function_value_negation_l1103_110374


namespace jamie_remaining_capacity_l1103_110303

/-- The maximum amount of liquid Jamie can consume before needing the bathroom -/
def max_liquid : ℕ := 32

/-- The amount of liquid Jamie has already consumed -/
def consumed_liquid : ℕ := 24

/-- The amount of additional liquid Jamie can consume -/
def remaining_capacity : ℕ := max_liquid - consumed_liquid

theorem jamie_remaining_capacity :
  remaining_capacity = 8 := by
  sorry

end jamie_remaining_capacity_l1103_110303


namespace power_mod_thirteen_l1103_110355

theorem power_mod_thirteen : 6^2040 ≡ 1 [ZMOD 13] := by sorry

end power_mod_thirteen_l1103_110355


namespace power_equality_l1103_110393

theorem power_equality (k m : ℕ) 
  (h1 : 3 ^ (k - 1) = 9) 
  (h2 : 4 ^ (m + 2) = 64) : 
  2 ^ (3 * k + 2 * m) = 2 ^ 11 := by
  sorry

end power_equality_l1103_110393


namespace toy_store_revenue_l1103_110380

theorem toy_store_revenue (december : ℝ) (november january : ℝ) 
  (h1 : november = (2/5) * december) 
  (h2 : january = (1/3) * november) : 
  december = 5 * ((november + january) / 2) := by
  sorry

end toy_store_revenue_l1103_110380


namespace factor_of_polynomial_l1103_110386

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^4 + 4 : ℝ) = (x^2 - 2*x + 2) * q x :=
sorry

end factor_of_polynomial_l1103_110386


namespace negation_of_forall_positive_negation_of_greater_than_zero_l1103_110370

theorem negation_of_forall_positive (p : ℝ → Prop) :
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬(p x)) :=
by sorry

theorem negation_of_greater_than_zero :
  (¬∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) :=
by sorry

end negation_of_forall_positive_negation_of_greater_than_zero_l1103_110370


namespace x_cubed_coefficient_equation_l1103_110376

theorem x_cubed_coefficient_equation (a : ℝ) : 
  (∃ k : ℝ, k = 56 ∧ k = 6 * a^2 - 15 * a + 20) ↔ (a = 6 ∨ a = -1) :=
by sorry

end x_cubed_coefficient_equation_l1103_110376


namespace farmer_profit_percentage_l1103_110383

/-- Calculate the profit percentage for a farmer's corn harvest --/
theorem farmer_profit_percentage
  (corn_seeds_cost : ℝ)
  (fertilizers_pesticides_cost : ℝ)
  (labor_cost : ℝ)
  (num_corn_bags : ℕ)
  (price_per_bag : ℝ)
  (h1 : corn_seeds_cost = 50)
  (h2 : fertilizers_pesticides_cost = 35)
  (h3 : labor_cost = 15)
  (h4 : num_corn_bags = 10)
  (h5 : price_per_bag = 11) :
  let total_cost := corn_seeds_cost + fertilizers_pesticides_cost + labor_cost
  let total_revenue := (num_corn_bags : ℝ) * price_per_bag
  let profit := total_revenue - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 10 := by
sorry

end farmer_profit_percentage_l1103_110383


namespace probability_above_curve_l1103_110308

-- Define the set of single-digit positive integers
def SingleDigitPos : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

-- Define the condition for (a,c) to be above the curve
def AboveCurve (a c : ℕ) : Prop := ∀ x : ℝ, c > a * x^3 - c * x^2

-- Define the count of valid points
def ValidPointsCount : ℕ := 16

-- Define the total number of possible points
def TotalPointsCount : ℕ := 81

-- State the theorem
theorem probability_above_curve :
  (↑ValidPointsCount / ↑TotalPointsCount : ℚ) = 16/81 :=
sorry

end probability_above_curve_l1103_110308


namespace red_marbles_count_l1103_110342

theorem red_marbles_count (total : ℕ) (blue : ℕ) (orange : ℕ) (red : ℕ) : 
  total = 24 →
  blue = total / 2 →
  orange = 6 →
  total = blue + orange + red →
  red = 6 := by
sorry

end red_marbles_count_l1103_110342


namespace license_plate_difference_l1103_110348

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The number of possible license plates for State A -/
def state_a_plates : ℕ := num_letters^5 * num_digits

/-- The number of possible license plates for State B -/
def state_b_plates : ℕ := num_letters^3 * num_digits^3

theorem license_plate_difference :
  state_a_plates - state_b_plates = 10123776 := by
  sorry

end license_plate_difference_l1103_110348


namespace no_infinite_prime_sequence_l1103_110324

theorem no_infinite_prime_sequence :
  ¬ ∃ (p : ℕ → ℕ), (∀ k, p (k + 1) = 5 * p k + 4) ∧ (∀ n, Nat.Prime (p n)) :=
by sorry

end no_infinite_prime_sequence_l1103_110324


namespace pencil_packs_l1103_110375

theorem pencil_packs (pencils_per_pack : ℕ) (pencils_per_row : ℕ) (num_rows : ℕ) : 
  pencils_per_pack = 24 →
  pencils_per_row = 16 →
  num_rows = 42 →
  (num_rows * pencils_per_row) / pencils_per_pack = 28 := by
sorry

end pencil_packs_l1103_110375


namespace total_crayons_l1103_110360

/-- Given that each child has 8 crayons and there are 7 children, prove that the total number of crayons is 56. -/
theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) 
  (h1 : crayons_per_child = 8) (h2 : num_children = 7) : 
  crayons_per_child * num_children = 56 := by
  sorry


end total_crayons_l1103_110360


namespace man_speed_against_stream_l1103_110398

/-- Calculates the speed against the stream given the rate in still water and speed with the stream -/
def speed_against_stream (rate_still : ℝ) (speed_with_stream : ℝ) : ℝ :=
  |rate_still - (speed_with_stream - rate_still)|

/-- Theorem: Given a man's rate in still water of 2 km/h and speed with the stream of 6 km/h,
    his speed against the stream is 2 km/h -/
theorem man_speed_against_stream :
  speed_against_stream 2 6 = 2 := by
  sorry

end man_speed_against_stream_l1103_110398


namespace parallelogram_side_length_l1103_110302

theorem parallelogram_side_length 
  (s : ℝ) 
  (angle : ℝ) 
  (area : ℝ) 
  (h₁ : angle = π / 3) -- 60 degrees in radians
  (h₂ : area = 27 * Real.sqrt 3)
  (h₃ : area = 3 * s * s * Real.sin angle) :
  s = 3 * Real.sqrt 2 := by
sorry

end parallelogram_side_length_l1103_110302


namespace average_of_quadratic_roots_l1103_110322

theorem average_of_quadratic_roots (c : ℝ) 
  (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 3 * x₁^2 - 6 * x₁ + c = 0 ∧ 3 * x₂^2 - 6 * x₂ + c = 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    3 * x₁^2 - 6 * x₁ + c = 0 ∧ 
    3 * x₂^2 - 6 * x₂ + c = 0 ∧
    (x₁ + x₂) / 2 = 1 :=
by sorry

end average_of_quadratic_roots_l1103_110322


namespace force_balance_l1103_110319

/-- A force in 2D space represented by its x and y components -/
structure Force where
  x : ℝ
  y : ℝ

/-- The sum of two forces -/
def Force.add (f g : Force) : Force :=
  ⟨f.x + g.x, f.y + g.y⟩

/-- The negation of a force -/
def Force.neg (f : Force) : Force :=
  ⟨-f.x, -f.y⟩

/-- Given two forces F₁ and F₂, prove that F₃ balances the system -/
theorem force_balance (F₁ F₂ F₃ : Force) 
    (h₁ : F₁ = ⟨1, 1⟩) 
    (h₂ : F₂ = ⟨2, 3⟩) 
    (h₃ : F₃ = ⟨-3, -4⟩) : 
  F₃.add (F₁.add F₂) = ⟨0, 0⟩ := by
  sorry


end force_balance_l1103_110319


namespace average_weight_increase_l1103_110328

/-- Proves that the increase in average weight is 2.5 kg when a person weighing 65 kg
    in a group of 6 is replaced by a person weighing 80 kg. -/
theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 6 →
  old_weight = 65 →
  new_weight = 80 →
  (new_weight - old_weight) / initial_count = 2.5 := by
  sorry

end average_weight_increase_l1103_110328


namespace smallest_k_property_l1103_110341

theorem smallest_k_property : ∃ k : ℝ, k = 2 ∧
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → 
    (a ≤ k ∨ b ≤ k ∨ (5 / a^2 + 6 / b^3) ≤ k)) ∧
  (∀ k' : ℝ, k' < k →
    ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧
      a > k' ∧ b > k' ∧ (5 / a^2 + 6 / b^3) > k') :=
by sorry

end smallest_k_property_l1103_110341


namespace plant_height_after_two_years_l1103_110300

/-- The height of a plant after a given number of years -/
def plant_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (4 ^ years)

/-- Theorem: A plant that quadruples its height every year and reaches 256 feet
    after 4 years will be 16 feet tall after 2 years -/
theorem plant_height_after_two_years
  (h : plant_height (plant_height 1 0) 4 = 256) :
  plant_height (plant_height 1 0) 2 = 16 := by
  sorry

#check plant_height_after_two_years

end plant_height_after_two_years_l1103_110300


namespace hillarys_descending_rate_l1103_110325

/-- Proof of Hillary's descending rate on Mt. Everest --/
theorem hillarys_descending_rate 
  (total_distance : ℝ) 
  (hillary_climbing_rate : ℝ) 
  (eddy_climbing_rate : ℝ) 
  (hillary_stop_short : ℝ) 
  (total_time : ℝ) :
  total_distance = 4700 →
  hillary_climbing_rate = 800 →
  eddy_climbing_rate = 500 →
  hillary_stop_short = 700 →
  total_time = 6 →
  ∃ (hillary_descending_rate : ℝ),
    hillary_descending_rate = 1000 ∧
    hillary_descending_rate * (total_time - (total_distance - hillary_stop_short) / hillary_climbing_rate) = 
    (total_distance - hillary_stop_short) - (eddy_climbing_rate * total_time) :=
by sorry

end hillarys_descending_rate_l1103_110325


namespace triangle_abc_properties_l1103_110316

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  (2 * c = Real.sqrt 3 * a + 2 * b * Real.cos A) →
  (c = 1) →
  (1 / 2 * a * c * Real.sin B = Real.sqrt 3 / 2) →
  -- Conclusions
  (B = π / 6) ∧ (b = Real.sqrt 7) :=
by sorry

end triangle_abc_properties_l1103_110316


namespace profit_decrease_l1103_110318

theorem profit_decrease (march_profit : ℝ) (h1 : march_profit > 0) : 
  let april_profit := march_profit * 1.4
  let june_profit := march_profit * 1.68
  ∃ (may_profit : ℝ), 
    may_profit = april_profit * 0.8 ∧ 
    june_profit = may_profit * 1.5 := by
  sorry

end profit_decrease_l1103_110318


namespace circus_ticket_cost_l1103_110372

theorem circus_ticket_cost (ticket_price : ℝ) (num_tickets : ℕ) (total_cost : ℝ) : 
  ticket_price = 44 ∧ num_tickets = 7 ∧ total_cost = ticket_price * num_tickets → total_cost = 308 :=
by sorry

end circus_ticket_cost_l1103_110372


namespace solve_system_l1103_110337

theorem solve_system (w u y z x : ℤ) 
  (hw : w = 100)
  (hz : z = w + 25)
  (hy : y = z + 12)
  (hu : u = y + 5)
  (hx : x = u + 7) : x = 149 := by
  sorry

end solve_system_l1103_110337


namespace five_sixths_of_twelve_fifths_l1103_110359

theorem five_sixths_of_twelve_fifths (a b c d : ℚ) : 
  a = 5 ∧ b = 6 ∧ c = 12 ∧ d = 5 → (a / b) * (c / d) = 2 := by
  sorry

end five_sixths_of_twelve_fifths_l1103_110359


namespace f_range_implies_a_range_l1103_110309

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + a else -a * (x - 2)^2 + 1

/-- The range of f(x) is (-∞, +∞) -/
def has_full_range (a : ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, f a x = y

/-- The range of a is (0, 2] -/
def a_range : Set ℝ := Set.Ioo 0 2 ∪ {2}

theorem f_range_implies_a_range :
  ∀ a : ℝ, has_full_range a → a ∈ a_range :=
sorry

end f_range_implies_a_range_l1103_110309


namespace lottery_savings_calculation_l1103_110399

theorem lottery_savings_calculation (lottery_winnings : ℚ) 
  (tax_rate : ℚ) (student_loan_rate : ℚ) (investment_rate : ℚ) (fun_money : ℚ) :
  lottery_winnings = 12006 →
  tax_rate = 1/2 →
  student_loan_rate = 1/3 →
  investment_rate = 1/5 →
  fun_money = 2802 →
  ∃ (savings : ℚ),
    savings = 1000 ∧
    lottery_winnings * (1 - tax_rate) * (1 - student_loan_rate) - fun_money = savings * (1 + investment_rate) :=
by sorry

end lottery_savings_calculation_l1103_110399


namespace geometric_sum_problem_l1103_110339

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_problem : 
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 3280/6561 := by
  sorry

end geometric_sum_problem_l1103_110339


namespace mice_on_bottom_path_l1103_110311

/-- Represents the number of mice in each house --/
structure MouseDistribution where
  left : ℕ
  top : ℕ
  right : ℕ

/-- The problem setup --/
def initial_distribution : MouseDistribution := ⟨8, 3, 7⟩
def final_distribution : MouseDistribution := ⟨5, 4, 9⟩

/-- The theorem to prove --/
theorem mice_on_bottom_path :
  let bottom_path_mice := 
    (initial_distribution.left + initial_distribution.right) -
    (final_distribution.left + final_distribution.right)
  bottom_path_mice = 11 := by
  sorry


end mice_on_bottom_path_l1103_110311


namespace ellipse_equation_l1103_110326

/-- 
Given an ellipse with center at the origin, one focus at (0, √50), 
and a chord intersecting the line y = 3x - 2 with midpoint x-coordinate 1/2, 
prove that the standard equation of the ellipse is x²/25 + y²/75 = 1.
-/
theorem ellipse_equation (F : ℝ × ℝ) (midpoint_x : ℝ) : 
  F = (0, Real.sqrt 50) →
  midpoint_x = 1/2 →
  ∃ (x y : ℝ), x^2/25 + y^2/75 = 1 ∧
    ∃ (x1 y1 x2 y2 : ℝ), 
      (x1^2/25 + y1^2/75 = 1) ∧
      (x2^2/25 + y2^2/75 = 1) ∧
      (y1 = 3*x1 - 2) ∧
      (y2 = 3*x2 - 2) ∧
      ((x1 + x2)/2 = midpoint_x) ∧
      ((y1 + y2)/2 = 3*midpoint_x - 2) :=
by sorry


end ellipse_equation_l1103_110326


namespace johns_pens_l1103_110395

/-- The number of pens John has -/
def total_pens (blue black red : ℕ) : ℕ := blue + black + red

theorem johns_pens :
  ∀ (blue black red : ℕ),
  blue = 18 →
  blue = 2 * black →
  black = red + 5 →
  total_pens blue black red = 31 := by
sorry

end johns_pens_l1103_110395


namespace equal_profit_loss_price_correct_l1103_110381

/-- The selling price that results in equal profit and loss -/
def equalProfitLossPrice (costPrice : ℕ) (lossPrice : ℕ) : ℕ :=
  costPrice + (costPrice - lossPrice)

theorem equal_profit_loss_price_correct (costPrice lossPrice : ℕ) :
  let sellingPrice := equalProfitLossPrice costPrice lossPrice
  (sellingPrice - costPrice) = (costPrice - lossPrice) → sellingPrice = 57 :=
by
  intro sellingPrice h
  sorry

#eval equalProfitLossPrice 50 43

end equal_profit_loss_price_correct_l1103_110381


namespace square_ratio_side_length_l1103_110379

theorem square_ratio_side_length (area_ratio : ℚ) :
  area_ratio = 250 / 98 →
  ∃ (a b c : ℕ), 
    (a = 25 ∧ b = 5 ∧ c = 7) ∧
    (Real.sqrt area_ratio * c = a * Real.sqrt b) :=
by sorry

end square_ratio_side_length_l1103_110379


namespace distinct_prime_factors_of_252_l1103_110336

theorem distinct_prime_factors_of_252 : Nat.card (Nat.factors 252).toFinset = 3 := by
  sorry

end distinct_prime_factors_of_252_l1103_110336


namespace cricket_game_overs_l1103_110312

theorem cricket_game_overs (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) (remaining_overs : ℝ) :
  target = 282 →
  initial_rate = 3.2 →
  required_rate = 6.25 →
  remaining_overs = 40 →
  ∃ x : ℝ, x = 10 ∧ initial_rate * x + required_rate * remaining_overs = target :=
by sorry

end cricket_game_overs_l1103_110312


namespace intersection_distance_l1103_110323

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 16 = 1

-- Define the parabola (using the derived equation from the solution)
def parabola (x y : ℝ) : Prop := x = y^2 / (4 * Real.sqrt 5) + Real.sqrt 5

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ parabola p.1 p.2}

-- Theorem statement
theorem intersection_distance :
  ∃ p1 p2 : ℝ × ℝ, p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 2 * Real.sqrt 10 :=
sorry

end intersection_distance_l1103_110323


namespace father_son_age_ratio_l1103_110387

/-- Represents the ages of a father and son -/
structure Ages where
  son : ℕ
  father : ℕ

/-- The current ages of the father and son -/
def currentAges : Ages :=
  { son := 24, father := 72 }

/-- The ages of the father and son 8 years ago -/
def pastAges : Ages :=
  { son := currentAges.son - 8, father := currentAges.father - 8 }

/-- The ratio of the father's age to the son's age -/
def ageRatio (ages : Ages) : ℚ :=
  ages.father / ages.son

theorem father_son_age_ratio :
  (pastAges.father = 4 * pastAges.son) →
  ageRatio currentAges = 3 / 1 := by
  sorry

#eval ageRatio currentAges

end father_son_age_ratio_l1103_110387


namespace total_spears_l1103_110363

/-- The number of spears that can be made from a sapling -/
def spears_per_sapling : ℕ := 3

/-- The number of spears that can be made from a log -/
def spears_per_log : ℕ := 9

/-- The number of spears that can be made from a bundle of branches -/
def spears_per_bundle : ℕ := 7

/-- The number of spears that can be made from a large tree trunk -/
def spears_per_trunk : ℕ := 15

/-- The number of saplings Marcy has -/
def num_saplings : ℕ := 6

/-- The number of logs Marcy has -/
def num_logs : ℕ := 1

/-- The number of bundles of branches Marcy has -/
def num_bundles : ℕ := 3

/-- The number of large tree trunks Marcy has -/
def num_trunks : ℕ := 2

/-- Theorem stating the total number of spears Marcy can make -/
theorem total_spears : 
  num_saplings * spears_per_sapling + 
  num_logs * spears_per_log + 
  num_bundles * spears_per_bundle + 
  num_trunks * spears_per_trunk = 78 := by
  sorry


end total_spears_l1103_110363


namespace chairs_count_l1103_110364

/-- The number of chairs in the auditorium at Yunju's school -/
def total_chairs : ℕ := by sorry

/-- The auditorium is square-shaped -/
axiom is_square : total_chairs = (Nat.sqrt total_chairs) ^ 2

/-- Yunju's seat is 2nd from the front -/
axiom front_distance : 2 ≤ Nat.sqrt total_chairs

/-- Yunju's seat is 5th from the back -/
axiom back_distance : 5 ≤ Nat.sqrt total_chairs

/-- Yunju's seat is 3rd from the right -/
axiom right_distance : 3 ≤ Nat.sqrt total_chairs

/-- Yunju's seat is 4th from the left -/
axiom left_distance : 4 ≤ Nat.sqrt total_chairs

/-- The theorem to be proved -/
theorem chairs_count : total_chairs = 36 := by sorry

end chairs_count_l1103_110364


namespace gcd_2873_1233_l1103_110356

theorem gcd_2873_1233 : Nat.gcd 2873 1233 = 1 := by
  sorry

end gcd_2873_1233_l1103_110356


namespace triangle_formation_l1103_110390

theorem triangle_formation (a b c : ℝ) : 
  a = 4 ∧ b = 6 ∧ c = 9 →
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end triangle_formation_l1103_110390


namespace inscribed_polygon_radius_l1103_110305

/-- A 12-sided convex polygon inscribed in a circle -/
structure InscribedPolygon where
  /-- The number of sides of the polygon -/
  sides : ℕ
  /-- The number of sides with length √2 -/
  short_sides : ℕ
  /-- The number of sides with length √24 -/
  long_sides : ℕ
  /-- The length of the short sides -/
  short_length : ℝ
  /-- The length of the long sides -/
  long_length : ℝ
  /-- Condition: The polygon has 12 sides -/
  sides_eq : sides = 12
  /-- Condition: There are 6 short sides -/
  short_sides_eq : short_sides = 6
  /-- Condition: There are 6 long sides -/
  long_sides_eq : long_sides = 6
  /-- Condition: The short sides have length √2 -/
  short_length_eq : short_length = Real.sqrt 2
  /-- Condition: The long sides have length √24 -/
  long_length_eq : long_length = Real.sqrt 24

/-- The theorem stating that the radius of the circle is 4√2 -/
theorem inscribed_polygon_radius (p : InscribedPolygon) : 
  ∃ (r : ℝ), r = 4 * Real.sqrt 2 := by
  sorry

end inscribed_polygon_radius_l1103_110305


namespace sin_product_equals_one_thirty_second_l1103_110340

theorem sin_product_equals_one_thirty_second :
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * Real.sin (72 * π / 180) * Real.sin (84 * π / 180) = 1 / 32 := by
  sorry

end sin_product_equals_one_thirty_second_l1103_110340
