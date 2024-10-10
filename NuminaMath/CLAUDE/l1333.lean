import Mathlib

namespace fleet_capacity_l1333_133377

theorem fleet_capacity (num_vans : ℕ) (large_capacity : ℕ) 
  (h_num_vans : num_vans = 6)
  (h_large_capacity : large_capacity = 8000)
  (h_small_capacity : ∃ small_capacity : ℕ, small_capacity = large_capacity - (large_capacity * 30 / 100))
  (h_very_large_capacity : ∃ very_large_capacity : ℕ, very_large_capacity = large_capacity + (large_capacity * 50 / 100))
  (h_num_large : ∃ num_large : ℕ, num_large = 2)
  (h_num_small : ∃ num_small : ℕ, num_small = 1)
  (h_num_very_large : ∃ num_very_large : ℕ, num_very_large = num_vans - 2 - 1) :
  ∃ total_capacity : ℕ, total_capacity = 57600 ∧
    total_capacity = (2 * large_capacity) + 
                     (large_capacity - (large_capacity * 30 / 100)) + 
                     (3 * (large_capacity + (large_capacity * 50 / 100))) :=
by
  sorry

end fleet_capacity_l1333_133377


namespace parabola_properties_l1333_133395

/-- A parabola is defined by the equation y = -(x-3)^2 --/
def parabola (x y : ℝ) : Prop := y = -(x-3)^2

/-- The axis of symmetry of the parabola --/
def axis_of_symmetry : ℝ := 3

/-- Theorem: The parabola opens downwards and has its axis of symmetry at x=3 --/
theorem parabola_properties :
  (∀ x y : ℝ, parabola x y → y ≤ 0) ∧ 
  (∀ y : ℝ, ∃ x₁ x₂ : ℝ, x₁ < axis_of_symmetry ∧ axis_of_symmetry < x₂ ∧ parabola x₁ y ∧ parabola x₂ y) :=
sorry

end parabola_properties_l1333_133395


namespace sum_remainder_theorem_l1333_133314

theorem sum_remainder_theorem :
  (9256 + 9257 + 9258 + 9259 + 9260) % 13 = 5 := by
  sorry

end sum_remainder_theorem_l1333_133314


namespace problem_solution_l1333_133319

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := x^3 + (1-a)*x^2 - a*(a+2)*x + b

/-- The derivative of f(x) with respect to x -/
def f_derivative (a x : ℝ) : ℝ := 3*x^2 + 2*(1-a)*x - a*(a+2)

theorem problem_solution :
  (∀ a b : ℝ, f a b 0 = 0 ∧ f_derivative a 0 = -3 → (a = -3 ∨ a = 1) ∧ b = 0) ∧
  (∀ a b : ℝ, (∃ x y : ℝ, x ≠ y ∧ f_derivative a x = 0 ∧ f_derivative a y = 0) →
    a < -1/2 ∨ a > -1/2) :=
sorry

end problem_solution_l1333_133319


namespace imaginary_part_of_z_l1333_133354

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = Complex.abs (4 + 3*I)) :
  z.im = 4/5 := by
  sorry

end imaginary_part_of_z_l1333_133354


namespace eighth_root_of_390625000000000_l1333_133347

theorem eighth_root_of_390625000000000 : (390625000000000 : ℝ) ^ (1/8 : ℝ) = 101 := by
  sorry

end eighth_root_of_390625000000000_l1333_133347


namespace only_one_four_cell_piece_l1333_133338

/-- Represents a piece on the board -/
structure Piece where
  size : Nat
  deriving Repr

/-- Represents the board configuration -/
structure Board where
  size : Nat
  pieces : List Piece
  deriving Repr

/-- Checks if a board configuration is valid -/
def isValidBoard (b : Board) : Prop :=
  b.size = 7 ∧ 
  b.pieces.all (λ p => p.size = 4) ∧
  b.pieces.length ≤ 3 ∧
  (b.pieces.map (λ p => p.size)).sum = b.size * b.size

/-- Theorem: Only one four-cell piece can be used in a valid 7x7 board configuration -/
theorem only_one_four_cell_piece (b : Board) :
  isValidBoard b → (b.pieces.filter (λ p => p.size = 4)).length = 1 := by
  sorry

#check only_one_four_cell_piece

end only_one_four_cell_piece_l1333_133338


namespace rectangle_perimeter_l1333_133399

/-- The perimeter of a rectangle formed when a smaller square is cut from the corner of a larger square -/
theorem rectangle_perimeter (t s : ℝ) (h : t > s) : 2 * s + 2 * (t - s) = 2 * t := by
  sorry

end rectangle_perimeter_l1333_133399


namespace fish_fraction_removed_on_day_five_l1333_133329

/-- Represents the number of fish in Jason's aquarium on a given day -/
def fish (day : ℕ) : ℚ :=
  match day with
  | 0 => 6
  | 1 => 12
  | 2 => 16
  | 3 => 32
  | 4 => 64
  | 5 => 128
  | 6 => 256
  | _ => 0

/-- The fraction of fish removed on day 5 -/
def f : ℚ := 1/4

theorem fish_fraction_removed_on_day_five :
  fish 6 - 4 * f * fish 4 + 15 = 207 :=
sorry

end fish_fraction_removed_on_day_five_l1333_133329


namespace largest_consecutive_even_integer_l1333_133379

theorem largest_consecutive_even_integer (a b c d : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- positive integers
  Even a ∧ Even b ∧ Even c ∧ Even d →  -- even integers
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2 →  -- consecutive
  a * b * c * d = 5040 →  -- product is 5040
  d = 20 :=  -- largest is 20
by sorry

end largest_consecutive_even_integer_l1333_133379


namespace problem_statement_l1333_133342

theorem problem_statement (w x y : ℝ) 
  (h1 : 7 / w + 7 / x = 7 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  x = 0.5 := by
sorry

end problem_statement_l1333_133342


namespace valleyball_hockey_league_players_l1333_133331

/-- The cost of a pair of gloves in dollars -/
def glove_cost : ℕ := 7

/-- The additional cost of a helmet compared to gloves in dollars -/
def helmet_additional_cost : ℕ := 8

/-- The total cost to equip all players in the league in dollars -/
def total_league_cost : ℕ := 3570

/-- The number of sets of equipment each player needs -/
def sets_per_player : ℕ := 2

/-- The number of players in the league -/
def num_players : ℕ := 81

theorem valleyball_hockey_league_players :
  num_players * sets_per_player * (glove_cost + (glove_cost + helmet_additional_cost)) = total_league_cost :=
sorry

end valleyball_hockey_league_players_l1333_133331


namespace ellipse_and_line_intersection_l1333_133336

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

/-- Definition of the line passing through (0, 2) with slope 1 -/
def line (x y : ℝ) : Prop :=
  y = x + 2

/-- Intersection points of the ellipse and the line -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2

theorem ellipse_and_line_intersection :
  ∃ (A B : ℝ × ℝ),
    intersection_points A B ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (6 * Real.sqrt 3 / 5)^2 :=
sorry

end ellipse_and_line_intersection_l1333_133336


namespace cos_three_halves_lt_sin_one_tenth_l1333_133384

theorem cos_three_halves_lt_sin_one_tenth :
  Real.cos (3/2) < Real.sin (1/10) := by
  sorry

end cos_three_halves_lt_sin_one_tenth_l1333_133384


namespace x_range_l1333_133305

-- Define the condition
def satisfies_equation (x y : ℝ) : Prop :=
  x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0

-- Define the theorem
theorem x_range (x y : ℝ) (h : satisfies_equation x y) :
  14 - 2 * Real.sqrt 13 ≤ x ∧ x ≤ 14 + 2 * Real.sqrt 13 :=
by sorry

end x_range_l1333_133305


namespace max_triangle_side_l1333_133313

theorem max_triangle_side (a b c : ℕ) : 
  a < b → b < c →  -- Ensure different side lengths
  a + b + c = 24 →  -- Perimeter condition
  a + b > c →  -- Triangle inequality
  a + c > b →  -- Triangle inequality
  b + c > a →  -- Triangle inequality
  c ≤ 11 :=
by sorry

end max_triangle_side_l1333_133313


namespace part_one_part_two_l1333_133320

noncomputable section

-- Define the functions
def f (b : ℝ) (x : ℝ) : ℝ := (2*x + b) * Real.exp x
def F (b : ℝ) (x : ℝ) : ℝ := b*x - Real.log x
def g (b : ℝ) (x : ℝ) : ℝ := b*x^2 - 2*x - F b x

-- Part 1
theorem part_one (b : ℝ) :
  b < 0 ∧ 
  (∃ (M : Set ℝ), ∀ (x y : ℝ), x ∈ M → y ∈ M → x < y → 
    ((f b x < f b y ↔ F b x < F b y) ∨ (f b x > f b y ↔ F b x > F b y))) →
  b < -2 :=
sorry

-- Part 2
theorem part_two (b : ℝ) :
  b > 0 ∧ 
  (∀ x ∈ Set.Icc 1 (Real.exp 1), g b x ≥ -2) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), g b x = -2) →
  b ≥ 1 :=
sorry

end part_one_part_two_l1333_133320


namespace m_range_l1333_133303

-- Define the propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + 2 ≥ m

def q (m : ℝ) : Prop := ∀ x : ℝ, (-(7 - 3*m))^(x+1) < (-(7 - 3*m))^x

-- State the theorem
theorem m_range (m : ℝ) : 
  (p m ∧ ¬(q m)) ∨ (¬(p m) ∧ q m) → 1 < m ∧ m < 2 :=
by sorry

end m_range_l1333_133303


namespace mountain_height_theorem_l1333_133302

-- Define the measurement points and the peak
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the measurement setup
structure MountainMeasurement where
  A : Point3D
  B : Point3D
  C : Point3D
  peak : Point3D
  AB : ℝ
  BC : ℝ
  angle_ABC : ℝ
  elevation_A : ℝ
  elevation_C : ℝ
  angle_BAT : ℝ

-- Define the theorem
theorem mountain_height_theorem (m : MountainMeasurement) 
  (h_AB : m.AB = 100)
  (h_BC : m.BC = 150)
  (h_angle_ABC : m.angle_ABC = 130 * π / 180)
  (h_elevation_A : m.elevation_A = 20 * π / 180)
  (h_elevation_C : m.elevation_C = 22 * π / 180)
  (h_angle_BAT : m.angle_BAT = 93 * π / 180) :
  ∃ (h1 h2 : ℝ), 
    (abs (h1 - 93.4) < 0.1 ∧ abs (h2 - 390.9) < 0.1) ∧
    ((m.peak.z - m.A.z = h1) ∨ (m.peak.z - m.A.z = h2)) := by
  sorry

end mountain_height_theorem_l1333_133302


namespace cubic_foot_to_cubic_inches_l1333_133359

/-- Conversion factor from feet to inches -/
def inches_per_foot : ℕ := 12

/-- Cubic inches in one cubic foot -/
def cubic_inches_per_cubic_foot : ℕ := inches_per_foot ^ 3

theorem cubic_foot_to_cubic_inches :
  cubic_inches_per_cubic_foot = 1728 :=
sorry

end cubic_foot_to_cubic_inches_l1333_133359


namespace hyperbola_equation_l1333_133369

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * Real.sqrt 2 * x

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x - 1

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop :=
  x = 2 * Real.sqrt 2

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : ∃ (x y : ℝ), asymptote x y ∧ (∃ (k : ℝ), y = (b/a) * x + k)) 
  (h4 : ∃ (x y : ℝ), hyperbola a b x y ∧ directrix x ∧ parabola x y) :
  a^2 = 2 ∧ b^2 = 6 :=
sorry

end hyperbola_equation_l1333_133369


namespace power_of_five_equality_l1333_133308

theorem power_of_five_equality (k : ℕ) : 5^k = 5 * 25^2 * 125^3 → k = 14 := by
  sorry

end power_of_five_equality_l1333_133308


namespace largest_y_value_l1333_133355

theorem largest_y_value (y : ℝ) : 
  (y / 7 + 2 / (3 * y) = 3) → y ≤ (63 + Real.sqrt 3801) / 6 :=
by sorry

end largest_y_value_l1333_133355


namespace john_typing_duration_l1333_133358

/-- The time John typed before Jack took over -/
def john_typing_time (
  john_total_time : ℝ)
  (jack_rate_ratio : ℝ)
  (jack_completion_time : ℝ) : ℝ :=
  3

/-- Theorem stating that John typed for 3 hours before Jack took over -/
theorem john_typing_duration :
  john_typing_time 5 (2/5) 4.999999999999999 = 3 := by
  sorry

end john_typing_duration_l1333_133358


namespace binomial_variance_l1333_133318

variable (p : ℝ)

-- Define the random variable X
def X : ℕ → ℝ
| 0 => 1 - p
| 1 => p
| _ => 0

-- Conditions
axiom p_range : 0 < p ∧ p < 1

-- Define the probability mass function
def pmf (k : ℕ) : ℝ := X p k

-- Define the expected value
def expectation : ℝ := p

-- Define the variance
def variance : ℝ := p * (1 - p)

-- Theorem statement
theorem binomial_variance : 
  ∀ (p : ℝ), 0 < p ∧ p < 1 → variance p = p * (1 - p) :=
by sorry

end binomial_variance_l1333_133318


namespace train_speed_l1333_133357

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 600) (h2 : time = 25) :
  length / time = 24 := by
  sorry

end train_speed_l1333_133357


namespace only_tiger_and_leopard_can_participate_l1333_133311

-- Define the animals
inductive Animal : Type
| Lion : Animal
| Tiger : Animal
| Leopard : Animal
| Elephant : Animal

-- Define a function to represent selection
def isSelected : Animal → Prop := sorry

-- Define the conditions
def conditions (isSelected : Animal → Prop) : Prop :=
  (isSelected Animal.Lion → isSelected Animal.Tiger) ∧
  (¬isSelected Animal.Leopard → ¬isSelected Animal.Tiger) ∧
  (isSelected Animal.Leopard → ¬isSelected Animal.Elephant) ∧
  (∃ (a b : Animal), a ≠ b ∧ isSelected a ∧ isSelected b ∧
    ∀ (c : Animal), c ≠ a ∧ c ≠ b → ¬isSelected c)

-- Theorem statement
theorem only_tiger_and_leopard_can_participate :
  ∀ (isSelected : Animal → Prop),
    conditions isSelected →
    isSelected Animal.Tiger ∧ isSelected Animal.Leopard ∧
    ¬isSelected Animal.Lion ∧ ¬isSelected Animal.Elephant :=
sorry

end only_tiger_and_leopard_can_participate_l1333_133311


namespace max_tetrahedron_volume_l1333_133387

noncomputable def square_pyramid_volume (base_side : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_side^2 * height

theorem max_tetrahedron_volume
  (base_side : ℝ)
  (m_distance : ℝ)
  (h : base_side = 6)
  (d : m_distance = 10) :
  ∃ (max_vol : ℝ),
    max_vol = 24 ∧
    ∀ (vol : ℝ),
      ∃ (height : ℝ),
        vol = square_pyramid_volume base_side height →
        vol ≤ max_vol :=
by sorry

end max_tetrahedron_volume_l1333_133387


namespace odd_function_properties_l1333_133378

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_properties (f : ℝ → ℝ) (h : is_odd f) :
  (f 0 = 0) ∧
  (∀ a > 0, (∀ x > 0, f x ≥ a) → (∀ y < 0, f y ≤ -a)) :=
by sorry

end odd_function_properties_l1333_133378


namespace intersection_of_parallel_planes_l1333_133380

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation for planes and lines
variable (parallelPlanes : Plane → Plane → Prop)
variable (parallelLines : Line → Line → Prop)

-- State the theorem
theorem intersection_of_parallel_planes 
  (α β γ : Plane) (m n : Line) :
  α ≠ β → α ≠ γ → β ≠ γ →
  m = intersect α γ →
  n = intersect β γ →
  parallelPlanes α β →
  parallelLines m n :=
sorry

end intersection_of_parallel_planes_l1333_133380


namespace arithmetic_sequence_length_l1333_133316

theorem arithmetic_sequence_length (a₁ aₙ d : ℤ) (h : a₁ = 165 ∧ aₙ = 45 ∧ d = -5) :
  (a₁ - aₙ) / (-d) + 1 = 25 := by
  sorry

end arithmetic_sequence_length_l1333_133316


namespace sqrt_180_simplified_l1333_133365

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end sqrt_180_simplified_l1333_133365


namespace quadratic_inequality_range_l1333_133382

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0) ↔ (1 ≤ m ∧ m < 9) :=
sorry

end quadratic_inequality_range_l1333_133382


namespace trig_function_slope_angle_l1333_133327

/-- Given a trigonometric function f(x) = a*sin(x) - b*cos(x) with the property
    that f(π/4 - x) = f(π/4 + x) for all x, prove that the slope angle of the line
    ax - by + c = 0 is 3π/4. -/
theorem trig_function_slope_angle (a b c : ℝ) :
  (∀ x, a * Real.sin x - b * Real.cos x = a * Real.sin (Real.pi/4 - x) - b * Real.cos (Real.pi/4 - x)) →
  (∃ k : ℝ, k > 0 ∧ a = k ∧ b = k) →
  Real.arctan (a / b) = 3 * Real.pi / 4 :=
sorry

end trig_function_slope_angle_l1333_133327


namespace quadratic_expression_rewrite_l1333_133373

theorem quadratic_expression_rewrite :
  ∃ (c p q : ℚ),
    (∀ k, 8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q) ∧
    q / p = -142 / 3 :=
by sorry

end quadratic_expression_rewrite_l1333_133373


namespace unique_integer_solution_l1333_133352

theorem unique_integer_solution : 
  ∃! (n : ℤ), (n^2 + 3*n + 5) / (n + 2 : ℚ) = 1 + Real.sqrt (6 - 2*n) := by
  sorry

end unique_integer_solution_l1333_133352


namespace melanie_dimes_l1333_133386

/-- The number of dimes Melanie has after receiving dimes from her parents -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Theorem: Melanie has 19 dimes after receiving dimes from her parents -/
theorem melanie_dimes : total_dimes 7 8 4 = 19 := by
  sorry

end melanie_dimes_l1333_133386


namespace inequality_solution_l1333_133330

def solution_set : Set ℝ := Set.union (Set.Icc 2 3) (Set.Ioc 3 48)

theorem inequality_solution (x : ℝ) : 
  x ∈ solution_set ↔ (x ≠ 3 ∧ (x * (x + 2)) / ((x - 3)^2) ≥ 8) :=
by sorry

end inequality_solution_l1333_133330


namespace blue_paint_cans_l1333_133368

/-- Given a paint mixture with a blue to green ratio of 4:3 and a total of 35 cans,
    prove that 20 cans of blue paint are needed. -/
theorem blue_paint_cans (total_cans : ℕ) (blue_ratio green_ratio : ℕ) : 
  total_cans = 35 → 
  blue_ratio = 4 → 
  green_ratio = 3 → 
  (blue_ratio * total_cans) / (blue_ratio + green_ratio) = 20 := by
  sorry

end blue_paint_cans_l1333_133368


namespace intersection_empty_iff_t_leq_neg_one_l1333_133394

-- Define sets A and B
def A : Set ℝ := {x | |x - 2| ≤ 3}
def B (t : ℝ) : Set ℝ := {x | x < t}

-- State the theorem
theorem intersection_empty_iff_t_leq_neg_one (t : ℝ) :
  A ∩ B t = ∅ ↔ t ≤ -1 := by sorry

end intersection_empty_iff_t_leq_neg_one_l1333_133394


namespace pension_calculation_l1333_133345

/-- Represents the pension calculation problem -/
theorem pension_calculation
  (c d r s y : ℝ)
  (h_cd : c ≠ d)
  (h_c : ∃ (t : ℝ), ∀ (x : ℝ), t * Real.sqrt (x + c - y) = t * Real.sqrt (x - y) + r)
  (h_d : ∃ (t : ℝ), ∀ (x : ℝ), t * Real.sqrt (x + d - y) = t * Real.sqrt (x - y) + s) :
  ∃ (t : ℝ), ∀ (x : ℝ), t * Real.sqrt (x - y) = (c * s^2 - d * r^2) / (2 * (d * r - c * s)) :=
sorry

end pension_calculation_l1333_133345


namespace probability_cos_geq_half_is_two_thirds_l1333_133341

noncomputable def probability_cos_geq_half : ℝ := by sorry

theorem probability_cos_geq_half_is_two_thirds :
  probability_cos_geq_half = 2/3 := by sorry

end probability_cos_geq_half_is_two_thirds_l1333_133341


namespace jackfruit_division_l1333_133351

/-- Represents the fair division of jackfruits between Renato and Leandro -/
def fair_division (renato_watermelons leandro_watermelons marcelo_jackfruits : ℕ) 
  (renato_jackfruits leandro_jackfruits : ℕ) : Prop :=
  renato_watermelons = 30 ∧
  leandro_watermelons = 18 ∧
  marcelo_jackfruits = 24 ∧
  renato_jackfruits + leandro_jackfruits = marcelo_jackfruits ∧
  (renato_watermelons + leandro_watermelons) / 3 = 16 ∧
  renato_jackfruits * 2 = renato_watermelons ∧
  leandro_jackfruits * 2 = leandro_watermelons

theorem jackfruit_division :
  ∃ (renato_jackfruits leandro_jackfruits : ℕ),
    fair_division 30 18 24 renato_jackfruits leandro_jackfruits ∧
    renato_jackfruits = 15 ∧
    leandro_jackfruits = 9 := by
  sorry

end jackfruit_division_l1333_133351


namespace gcd_lcm_sum_specific_l1333_133397

theorem gcd_lcm_sum_specific : Nat.gcd 45 4410 + Nat.lcm 45 4410 = 4455 := by sorry

end gcd_lcm_sum_specific_l1333_133397


namespace tan_product_values_l1333_133361

theorem tan_product_values (a b : Real) :
  3 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = 1 / 2 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -2 := by
  sorry

end tan_product_values_l1333_133361


namespace expression_equivalence_l1333_133349

theorem expression_equivalence : 
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) * 
  (5^32 + 7^32) * (5^64 + 7^64) * (5^128 + 7^128) = 7^256 - 5^256 := by
  sorry

end expression_equivalence_l1333_133349


namespace min_distance_and_slope_l1333_133371

-- Define the circle F
def circle_F (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the curve W (trajectory)
def curve_W (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l passing through F(1,0)
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the intersection points
def point_A (k : ℝ) : ℝ × ℝ := sorry
def point_D (k : ℝ) : ℝ × ℝ := sorry
def point_B (k : ℝ) : ℝ × ℝ := sorry
def point_C (k : ℝ) : ℝ × ℝ := sorry

-- Define the distances
def dist_AB (k : ℝ) : ℝ := sorry
def dist_CD (k : ℝ) : ℝ := sorry

-- State the theorem
theorem min_distance_and_slope :
  ∃ (k : ℝ), 
    (∀ (k' : ℝ), dist_AB k + 4 * dist_CD k ≤ dist_AB k' + 4 * dist_CD k') ∧
    dist_AB k + 4 * dist_CD k = 4 ∧
    (k = 2 * Real.sqrt 2 ∨ k = -2 * Real.sqrt 2) :=
sorry

end min_distance_and_slope_l1333_133371


namespace no_prime_roots_for_quadratic_l1333_133312

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem no_prime_roots_for_quadratic :
  ¬∃ k : ℤ, ∃ p q : ℕ, 
    is_prime p ∧ is_prime q ∧ 
    p ≠ q ∧
    (p : ℤ) * (q : ℤ) = k ∧ 
    (p : ℤ) + (q : ℤ) = 58 :=
sorry

end no_prime_roots_for_quadratic_l1333_133312


namespace min_ice_cost_l1333_133337

/-- Represents the ice purchasing options --/
inductive IcePackType
  | OnePound
  | FivePound

/-- Calculates the cost of ice for a given pack type and number of packs --/
def calculateCost (packType : IcePackType) (numPacks : ℕ) : ℚ :=
  match packType with
  | IcePackType.OnePound => 
      if numPacks > 20 
      then (6 * numPacks : ℚ) * 0.9
      else 6 * numPacks
  | IcePackType.FivePound => 
      if numPacks > 20 
      then (25 * numPacks : ℚ) * 0.9
      else 25 * numPacks

/-- Calculates the number of packs needed for a given pack type and total ice needed --/
def calculatePacks (packType : IcePackType) (totalIce : ℕ) : ℕ :=
  match packType with
  | IcePackType.OnePound => (totalIce + 9) / 10
  | IcePackType.FivePound => (totalIce + 49) / 50

/-- Theorem: The minimum cost for ice is $100.00 --/
theorem min_ice_cost : 
  let totalPeople : ℕ := 50
  let icePerPerson : ℕ := 4
  let totalIce : ℕ := totalPeople * icePerPerson
  let onePoundCost := calculateCost IcePackType.OnePound (calculatePacks IcePackType.OnePound totalIce)
  let fivePoundCost := calculateCost IcePackType.FivePound (calculatePacks IcePackType.FivePound totalIce)
  min onePoundCost fivePoundCost = 100 := by
  sorry

end min_ice_cost_l1333_133337


namespace cupcake_frosting_problem_l1333_133328

/-- Represents the number of cupcakes frosted in a given time -/
def cupcakes_frosted (rate : ℚ) (time : ℚ) : ℚ := rate * time

/-- Represents the combined rate of two people frosting cupcakes -/
def combined_rate (rate1 : ℚ) (rate2 : ℚ) : ℚ := 1 / (1 / rate1 + 1 / rate2)

theorem cupcake_frosting_problem :
  let cagney_rate : ℚ := 1 / 18  -- Cagney's frosting rate (cupcakes per second)
  let lacey_rate : ℚ := 1 / 40   -- Lacey's frosting rate (cupcakes per second)
  let total_time : ℚ := 6 * 60   -- Total time in seconds
  let lacey_delay : ℚ := 60      -- Lacey's delay in seconds

  let cagney_solo_time := lacey_delay
  let combined_time := total_time - lacey_delay
  let combined_frosting_rate := combined_rate cagney_rate lacey_rate

  let total_cupcakes := 
    cupcakes_frosted cagney_rate cagney_solo_time + 
    cupcakes_frosted combined_frosting_rate combined_time

  ⌊total_cupcakes⌋ = 27 :=
by sorry

end cupcake_frosting_problem_l1333_133328


namespace pig_count_l1333_133360

theorem pig_count (P1 P2 : ℕ) (h1 : P1 = 64) (h2 : P1 + P2 = 86) : P2 = 22 := by
  sorry

end pig_count_l1333_133360


namespace pauls_crayons_and_erasers_l1333_133350

theorem pauls_crayons_and_erasers 
  (initial_crayons : ℕ) 
  (initial_erasers : ℕ) 
  (final_crayons : ℕ) 
  (h1 : initial_crayons = 531)
  (h2 : initial_erasers = 38)
  (h3 : final_crayons = 391)
  (h4 : initial_erasers = final_erasers) :
  initial_crayons - final_crayons - initial_erasers = 102 :=
by sorry

end pauls_crayons_and_erasers_l1333_133350


namespace quadratic_roots_sum_squares_and_product_l1333_133366

theorem quadratic_roots_sum_squares_and_product (u v : ℝ) : 
  u^2 - 5*u + 3 = 0 → v^2 - 5*v + 3 = 0 → u^2 + v^2 + u*v = 22 := by
  sorry

end quadratic_roots_sum_squares_and_product_l1333_133366


namespace cyclist_speed_ratio_l1333_133325

theorem cyclist_speed_ratio : 
  ∀ (v_A v_B v_C : ℝ),
    v_A > 0 → v_B > 0 → v_C > 0 →
    (v_A - v_B) * 4 = 20 →
    (v_A + v_C) * 2 = 30 →
    v_A / v_B = 3 :=
by
  sorry

end cyclist_speed_ratio_l1333_133325


namespace fraction_value_l1333_133390

theorem fraction_value (a b : ℚ) (h1 : a = 7) (h2 : b = 2) : 3 / (a + b) = 1 / 3 := by
  sorry

end fraction_value_l1333_133390


namespace binomial_60_3_l1333_133393

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end binomial_60_3_l1333_133393


namespace circles_intersect_l1333_133376

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x-2)^2 + y^2 = 9

-- Define the distance between the centers
def distance_between_centers : ℝ := 2

-- Define the radii of the circles
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_between_centers > |radius1 - radius2| ∧
  distance_between_centers < radius1 + radius2 :=
sorry

end circles_intersect_l1333_133376


namespace polynomial_sum_coefficients_l1333_133396

theorem polynomial_sum_coefficients : 
  ∀ A B C D E : ℚ, 
  (∀ x : ℚ, (x + 2) * (x + 3) * (3*x^2 - x + 5) = A*x^4 + B*x^3 + C*x^2 + D*x + E) →
  A + B + C + D + E = 84 := by
  sorry

end polynomial_sum_coefficients_l1333_133396


namespace white_ball_count_l1333_133391

theorem white_ball_count (total : ℕ) (white blue red : ℕ) : 
  total = 1000 →
  blue = white + 14 →
  red = 3 * (blue - white) →
  total = white + blue + red →
  white = 472 := by
  sorry

end white_ball_count_l1333_133391


namespace max_distance_circle_to_line_l1333_133317

theorem max_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | 3*x + 4*y - 25 = 0}
  ∃ (p : ℝ × ℝ), p ∈ circle ∧
    (∀ (q : ℝ × ℝ), q ∈ circle →
      ∃ (r : ℝ × ℝ), r ∈ line ∧
        dist p r ≥ dist q r) ∧
    (∃ (s : ℝ × ℝ), s ∈ line ∧ dist p s = 6) :=
by sorry

end max_distance_circle_to_line_l1333_133317


namespace scientific_notation_proof_l1333_133381

theorem scientific_notation_proof : ∃ (a : ℝ) (n : ℤ), 
  0.00076 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7.6 ∧ n = -4 := by
  sorry

end scientific_notation_proof_l1333_133381


namespace least_odd_prime_factor_1234_10_plus_1_l1333_133323

theorem least_odd_prime_factor_1234_10_plus_1 : 
  (Nat.minFac (1234^10 + 1)) = 61 := by sorry

end least_odd_prime_factor_1234_10_plus_1_l1333_133323


namespace division_remainder_problem_l1333_133372

theorem division_remainder_problem 
  (P D Q R D' Q' R' C : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R')
  (h3 : R < D)
  (h4 : R' < D') :
  P % ((D + C) * D') = D' * C * R' + D * R' + C * R' + R := by
sorry

end division_remainder_problem_l1333_133372


namespace horner_method_equals_f_at_2_l1333_133334

-- Define the polynomial function
def f (x : ℝ) : ℝ := 8 * x^7 + 5 * x^6 + 3 * x^4 + 2 * x + 1

-- Define Horner's method for this specific polynomial
def horner_method (x : ℝ) : ℝ :=
  ((((((8 * x + 5) * x + 0) * x + 3) * x + 0) * x + 0) * x + 2) * x + 1

-- Theorem statement
theorem horner_method_equals_f_at_2 : 
  horner_method 2 = f 2 ∧ horner_method 2 = 1397 := by sorry

end horner_method_equals_f_at_2_l1333_133334


namespace sheila_hourly_wage_l1333_133362

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  monday_hours : ℕ
  wednesday_hours : ℕ
  friday_hours : ℕ
  tuesday_hours : ℕ
  thursday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculate Sheila's hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := 
    3 * schedule.monday_hours + 
    2 * schedule.tuesday_hours
  schedule.weekly_earnings / total_hours

/-- Sheila's actual work schedule --/
def sheila_schedule : WorkSchedule := {
  monday_hours := 8
  wednesday_hours := 8
  friday_hours := 8
  tuesday_hours := 6
  thursday_hours := 6
  weekly_earnings := 504
}

/-- Theorem: Sheila's hourly wage is $14 --/
theorem sheila_hourly_wage : 
  hourly_wage sheila_schedule = 14 := by
  sorry

end sheila_hourly_wage_l1333_133362


namespace square_product_equality_l1333_133363

theorem square_product_equality : (15 : ℕ)^2 * 9^2 * 356 = 6489300 := by
  sorry

end square_product_equality_l1333_133363


namespace vasyas_birthday_vasyas_birthday_was_thursday_l1333_133307

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the day after tomorrow
def dayAfterTomorrow (d : DayOfWeek) : DayOfWeek :=
  nextDay (nextDay d)

theorem vasyas_birthday (today : DayOfWeek) 
  (h1 : dayAfterTomorrow today = DayOfWeek.Sunday) 
  (h2 : nextDay today ≠ DayOfWeek.Sunday) : 
  nextDay (nextDay (nextDay today)) = DayOfWeek.Sunday := by
  sorry

-- The main theorem
theorem vasyas_birthday_was_thursday : 
  ∃ (today : DayOfWeek), 
    dayAfterTomorrow today = DayOfWeek.Sunday ∧ 
    nextDay today ≠ DayOfWeek.Sunday ∧
    nextDay (nextDay (nextDay today)) = DayOfWeek.Sunday := by
  sorry

end vasyas_birthday_vasyas_birthday_was_thursday_l1333_133307


namespace perfect_square_trinomial_m_values_l1333_133333

/-- A polynomial is a perfect square trinomial if it can be expressed as (x + a)^2 for some real number a. -/
def IsPerfectSquareTrinomial (p : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, p x = (x + a)^2

/-- Given that x^2 + (m-2)x + 9 is a perfect square trinomial, prove that m = 8 or m = -4. -/
theorem perfect_square_trinomial_m_values (m : ℝ) :
  IsPerfectSquareTrinomial (fun x ↦ x^2 + (m-2)*x + 9) → m = 8 ∨ m = -4 := by
  sorry


end perfect_square_trinomial_m_values_l1333_133333


namespace cube_sum_magnitude_l1333_133374

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2) 
  (h2 : Complex.abs (w^2 + z^2) = 8) : 
  Complex.abs (w^3 + z^3) = 20 := by
sorry

end cube_sum_magnitude_l1333_133374


namespace quadratic_equation_with_specific_discriminant_l1333_133392

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α

/-- Calculates the discriminant of a quadratic equation -/
def discriminant {α : Type*} [Field α] (eq : QuadraticEquation α) : α :=
  eq.b ^ 2 - 4 * eq.a * eq.c

/-- Checks if the roots of a quadratic equation are real and unequal -/
def has_real_unequal_roots {α : Type*} [LinearOrderedField α] (eq : QuadraticEquation α) : Prop :=
  discriminant eq > 0

theorem quadratic_equation_with_specific_discriminant 
  (d : ℝ) (eq : QuadraticEquation ℝ) 
  (h1 : eq.a = 3)
  (h2 : eq.b = -6 * Real.sqrt 3)
  (h3 : eq.c = d)
  (h4 : discriminant eq = 12) :
  d = 8 ∧ has_real_unequal_roots eq :=
sorry

end quadratic_equation_with_specific_discriminant_l1333_133392


namespace cricket_score_problem_l1333_133353

theorem cricket_score_problem (a b c d e : ℕ) : 
  (a + b + c + d + e) / 5 = 36 ∧  -- average score
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, a = 4 * k₁ ∧ b = 4 * k₂ ∧ c = 4 * k₃ ∧ d = 4 * k₄ ∧ e = 4 * k₅) ∧  -- scores are multiples of 4
  d = e + 12 ∧  -- D scored 12 more than E
  e = a - 8 ∧  -- E scored 8 fewer than A
  b = d + e ∧  -- B scored as many as D and E combined
  b + c = 107 ∧  -- B and C scored 107 between them
  a > b ∧ a > c ∧ a > d ∧ a > e  -- A scored the maximum runs
  →
  e = 20 := by
sorry

end cricket_score_problem_l1333_133353


namespace function_intersection_theorem_l1333_133398

noncomputable def f (a b x : ℝ) : ℝ := (a * Real.log x + b) / x

noncomputable def g (a x : ℝ) : ℝ := a + 2 - x - 2 / x

def has_extremum_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

def exactly_one_intersection (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x ≤ b ∧ f x = g x

theorem function_intersection_theorem (a b : ℝ) :
  a ≤ 2 →
  a ≠ 0 →
  has_extremum_at (f a b) (1 / Real.exp 1) →
  (a = -1 ∨ a < -2 / Real.log 2 ∨ (0 < a ∧ a ≤ 2)) ↔
  exactly_one_intersection (f a b) (g a) 0 2 :=
sorry

end function_intersection_theorem_l1333_133398


namespace rectangle_area_l1333_133335

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 6 → 
  ratio = 3 → 
  (2 * r) * (ratio * 2 * r) = 432 :=
by sorry

end rectangle_area_l1333_133335


namespace binary_101_to_decimal_l1333_133367

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101_to_decimal :
  binary_to_decimal [true, false, true] = 5 := by
  sorry

end binary_101_to_decimal_l1333_133367


namespace M_values_l1333_133364

theorem M_values (a b : ℚ) (hab : a * b ≠ 0) :
  let M := (2 * abs a) / a + (3 * b) / abs b
  M = 1 ∨ M = -1 ∨ M = 5 ∨ M = -5 := by sorry

end M_values_l1333_133364


namespace diana_box_capacity_l1333_133315

/-- Represents a box with dimensions and jellybean capacity -/
structure Box where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℕ

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℝ :=
  b.height * b.width * b.length

/-- Theorem: A box with triple height, double width, and quadruple length of Bert's box
    that holds 150 jellybeans will hold 3600 jellybeans -/
theorem diana_box_capacity (bert_box : Box)
    (h1 : bert_box.capacity = 150)
    (diana_box : Box)
    (h2 : diana_box.height = 3 * bert_box.height)
    (h3 : diana_box.width = 2 * bert_box.width)
    (h4 : diana_box.length = 4 * bert_box.length) :
    diana_box.capacity = 3600 := by
  sorry


end diana_box_capacity_l1333_133315


namespace lindas_lunchbox_total_cost_l1333_133370

/-- The cost of a sandwich at Linda's Lunchbox -/
def sandwich_cost : ℕ := 4

/-- The cost of a soda at Linda's Lunchbox -/
def soda_cost : ℕ := 2

/-- The cost of a cookie at Linda's Lunchbox -/
def cookie_cost : ℕ := 1

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 7

/-- The number of sodas purchased -/
def num_sodas : ℕ := 6

/-- The number of cookies purchased -/
def num_cookies : ℕ := 4

/-- The total cost of the purchase at Linda's Lunchbox -/
def total_cost : ℕ := num_sandwiches * sandwich_cost + num_sodas * soda_cost + num_cookies * cookie_cost

theorem lindas_lunchbox_total_cost : total_cost = 44 := by
  sorry

end lindas_lunchbox_total_cost_l1333_133370


namespace highway_project_deadline_l1333_133339

/-- Represents the initial deadline for completing the highway project --/
def initial_deadline : ℝ := 37.5

/-- The number of initial workers --/
def initial_workers : ℕ := 100

/-- The number of additional workers hired --/
def additional_workers : ℕ := 60

/-- The initial daily work hours --/
def initial_hours : ℕ := 8

/-- The new daily work hours after hiring additional workers --/
def new_hours : ℕ := 10

/-- The number of days worked before hiring additional workers --/
def days_worked : ℕ := 25

/-- The fraction of work completed before hiring additional workers --/
def work_completed : ℚ := 1/3

/-- Theorem stating that the initial deadline is correct given the conditions --/
theorem highway_project_deadline :
  ∃ (total_work : ℝ),
    total_work = initial_workers * days_worked * initial_hours ∧
    (2/3 : ℝ) * total_work = (initial_workers + additional_workers) * (initial_deadline - days_worked) * new_hours :=
by sorry

end highway_project_deadline_l1333_133339


namespace sticker_distribution_solution_l1333_133344

/-- Represents the sticker distribution problem --/
structure StickerDistribution where
  space : ℕ := 120
  cat : ℕ := 80
  dinosaur : ℕ := 150
  superhero : ℕ := 45
  space_given : ℕ := 25
  cat_given : ℕ := 13
  dinosaur_given : ℕ := 33
  superhero_given : ℕ := 29

/-- Calculates the total number of stickers left after initial distribution --/
def remaining_stickers (sd : StickerDistribution) : ℕ :=
  (sd.space - sd.space_given) + (sd.cat - sd.cat_given) + 
  (sd.dinosaur - sd.dinosaur_given) + (sd.superhero - sd.superhero_given)

/-- Theorem stating the solution to the sticker distribution problem --/
theorem sticker_distribution_solution (sd : StickerDistribution) :
  ∃ (X : ℕ), X = 3 ∧ (remaining_stickers sd - X) / 4 = 73 := by
  sorry


end sticker_distribution_solution_l1333_133344


namespace fruit_salad_cherries_l1333_133304

/-- Represents the number of fruits in a salad -/
structure FruitSalad where
  blueberries : ℕ
  raspberries : ℕ
  grapes : ℕ
  cherries : ℕ

/-- Conditions for the fruit salad problem -/
def validFruitSalad (s : FruitSalad) : Prop :=
  s.blueberries + s.raspberries + s.grapes + s.cherries = 350 ∧
  s.raspberries = 3 * s.blueberries ∧
  s.grapes = 4 * s.cherries ∧
  s.cherries = 5 * s.raspberries

/-- Theorem stating that a valid fruit salad has 66 cherries -/
theorem fruit_salad_cherries (s : FruitSalad) (h : validFruitSalad s) : s.cherries = 66 := by
  sorry

#check fruit_salad_cherries

end fruit_salad_cherries_l1333_133304


namespace point_on_x_axis_l1333_133383

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point P with coordinates (m+3, m+1) -/
def P (m : ℝ) : Point :=
  { x := m + 3, y := m + 1 }

/-- Theorem: If P(m+3, m+1) lies on the x-axis, then its coordinates are (2, 0) -/
theorem point_on_x_axis (m : ℝ) : 
  (P m).y = 0 → P m = { x := 2, y := 0 } := by
  sorry

end point_on_x_axis_l1333_133383


namespace quadrilateral_division_theorem_l1333_133388

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  /-- The sum of internal angles is 360 degrees -/
  angle_sum : ℝ
  angle_sum_eq : angle_sum = 360

/-- A diagonal of a quadrilateral -/
structure Diagonal (Q : ConvexQuadrilateral) where
  /-- The diagonal divides the quadrilateral into two triangles -/
  divides_into_triangles : Prop

/-- A triangle formed by a diagonal in a quadrilateral -/
structure Triangle (Q : ConvexQuadrilateral) (D : Diagonal Q) where
  /-- The sum of angles in the triangle is 180 degrees -/
  angle_sum : ℝ
  angle_sum_eq : angle_sum = 180

/-- Theorem: In a convex quadrilateral, it's impossible to divide it by a diagonal into two acute triangles, 
    while it's possible to divide it into two right triangles or two obtuse triangles -/
theorem quadrilateral_division_theorem (Q : ConvexQuadrilateral) :
  (∃ D : Diagonal Q, ∃ T1 T2 : Triangle Q D, 
    T1.angle_sum < 180 ∧ T2.angle_sum < 180) → False
  ∧ (∃ D : Diagonal Q, ∃ T1 T2 : Triangle Q D, 
    T1.angle_sum = 180 ∧ T2.angle_sum = 180)
  ∧ (∃ D : Diagonal Q, ∃ T1 T2 : Triangle Q D, 
    T1.angle_sum > 180 ∧ T2.angle_sum > 180) :=
by sorry

end quadrilateral_division_theorem_l1333_133388


namespace complex_arithmetic_problem_l1333_133306

theorem complex_arithmetic_problem : ((6^2 - 4^2) + 2)^3 / 2 = 5324 := by
  sorry

end complex_arithmetic_problem_l1333_133306


namespace geometric_log_arithmetic_l1333_133310

open Real

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is geometric if there exists a non-zero real number q such that
    for all n, a(n+1) = q * a(n) -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- A sequence is arithmetic if there exists a real number d such that
    for all n, a(n+1) - a(n) = d -/
def IsArithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The main theorem: If a sequence of positive terms is geometric,
    then the sequence of logarithms plus 1 is arithmetic,
    but the converse is not always true -/
theorem geometric_log_arithmetic (a : Sequence) :
  (∀ n : ℕ, a n > 0) →
  IsGeometric a →
  IsArithmetic (fun n => log (a n) + 1) ∧
  ¬(IsArithmetic (fun n => log (a n) + 1) → IsGeometric a) :=
sorry

end geometric_log_arithmetic_l1333_133310


namespace smallest_positive_integer_x_l1333_133348

theorem smallest_positive_integer_x (x : ℕ+) : (2 * (x : ℝ)^2 < 50) → x = 1 := by
  sorry

end smallest_positive_integer_x_l1333_133348


namespace last_two_digits_sum_factorials_12_l1333_133389

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |> List.sum

theorem last_two_digits_sum_factorials_12 :
  lastTwoDigits (sumFactorials 12) = 13 := by sorry

end last_two_digits_sum_factorials_12_l1333_133389


namespace midpoint_square_sum_l1333_133332

/-- Given that C = (5, 3) is the midpoint of line segment AB, where A = (3, -3) and B = (x, y),
    prove that x^2 + y^2 = 130. -/
theorem midpoint_square_sum (x y : ℝ) : 
  (5 : ℝ) = (3 + x) / 2 ∧ (3 : ℝ) = (-3 + y) / 2 → x^2 + y^2 = 130 := by
  sorry

end midpoint_square_sum_l1333_133332


namespace unique_magnitude_of_complex_roots_l1333_133321

theorem unique_magnitude_of_complex_roots : ∃! r : ℝ, ∃ z : ℂ, z^2 - 8*z + 45 = 0 ∧ Complex.abs z = r := by
  sorry

end unique_magnitude_of_complex_roots_l1333_133321


namespace parallel_segment_length_l1333_133301

/-- A trapezoid with bases a and b -/
structure Trapezoid (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- A line segment within a trapezoid -/
def ParallelSegment (t : Trapezoid a b) := ℝ

/-- The property that a line divides a trapezoid into two similar trapezoids -/
def DividesSimilarly (t : Trapezoid a b) (s : ParallelSegment t) : Prop :=
  sorry

/-- Theorem: If a line parallel to the bases divides a trapezoid into two similar trapezoids,
    then the length of the segment is the square root of the product of the bases -/
theorem parallel_segment_length (a b : ℝ) (t : Trapezoid a b) (s : ParallelSegment t) :
  DividesSimilarly t s → s = Real.sqrt (a * b) :=
sorry

end parallel_segment_length_l1333_133301


namespace arithmetic_combination_l1333_133309

theorem arithmetic_combination : (2 + 4 / 10) * 10 = 24 := by
  sorry

end arithmetic_combination_l1333_133309


namespace meet_once_l1333_133375

/-- Represents the movement of Michael and the garbage truck --/
structure Movement where
  michaelSpeed : ℝ
  truckSpeed : ℝ
  pailDistance : ℝ
  truckStopTime : ℝ

/-- Calculates the number of times Michael and the truck meet --/
def meetingCount (m : Movement) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem meet_once (m : Movement) 
  (h1 : m.michaelSpeed = 6)
  (h2 : m.truckSpeed = 10)
  (h3 : m.pailDistance = 200)
  (h4 : m.truckStopTime = 40)
  (h5 : m.pailDistance = m.truckSpeed * (m.pailDistance / m.michaelSpeed - m.truckStopTime)) :
  meetingCount m = 1 := by
  sorry

#check meet_once

end meet_once_l1333_133375


namespace closest_fraction_l1333_133322

def medals_won : ℚ := 23 / 120

def fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

theorem closest_fraction :
  ∃ (x : ℚ), x ∈ fractions ∧
  ∀ (y : ℚ), y ∈ fractions → |medals_won - x| ≤ |medals_won - y| ∧
  x = 1/5 :=
sorry

end closest_fraction_l1333_133322


namespace cubic_root_sum_l1333_133324

theorem cubic_root_sum (a b c d : ℝ) (ha : a ≠ 0) : 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 4 ∨ x = -3) →
  (b + c) / a = -13 := by
sorry

end cubic_root_sum_l1333_133324


namespace combinations_equal_twenty_l1333_133346

/-- The number of available colors -/
def num_colors : ℕ := 5

/-- The number of available painting methods -/
def num_methods : ℕ := 4

/-- The total number of combinations of colors and painting methods -/
def total_combinations : ℕ := num_colors * num_methods

/-- Theorem stating that the total number of combinations is 20 -/
theorem combinations_equal_twenty : total_combinations = 20 := by
  sorry

end combinations_equal_twenty_l1333_133346


namespace tara_ice_cream_purchase_l1333_133300

/-- The number of cartons of ice cream Tara bought -/
def ice_cream_cartons : ℕ := sorry

/-- The number of cartons of yoghurt Tara bought -/
def yoghurt_cartons : ℕ := 4

/-- The cost of one carton of ice cream in dollars -/
def ice_cream_cost : ℕ := 7

/-- The cost of one carton of yoghurt in dollars -/
def yoghurt_cost : ℕ := 1

/-- Theorem stating that Tara bought 19 cartons of ice cream -/
theorem tara_ice_cream_purchase :
  ice_cream_cartons = 19 ∧
  ice_cream_cartons * ice_cream_cost = yoghurt_cartons * yoghurt_cost + 129 :=
by sorry

end tara_ice_cream_purchase_l1333_133300


namespace area_equals_half_radius_times_pedal_perimeter_l1333_133356

open Real

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the pedal triangle of a given triangle -/
def pedalTriangle (T : Triangle) : Triangle := sorry

/-- The area of a triangle -/
def area (T : Triangle) : ℝ := sorry

/-- The perimeter of a triangle -/
def perimeter (T : Triangle) : ℝ := sorry

/-- The circumradius of a triangle -/
def circumradius (T : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is acute-angled -/
def isAcute (T : Triangle) : Prop := sorry

theorem area_equals_half_radius_times_pedal_perimeter (T : Triangle) 
  (h : isAcute T) : 
  area T = (circumradius T / 2) * perimeter (pedalTriangle T) := by
  sorry

end area_equals_half_radius_times_pedal_perimeter_l1333_133356


namespace sphere_hemisphere_volume_ratio_l1333_133385

theorem sphere_hemisphere_volume_ratio (p : ℝ) (hp : p > 0) :
  (4 / 3 * Real.pi * p^3) / (1 / 2 * 4 / 3 * Real.pi * (2*p)^3) = 1 / 4 := by
  sorry

end sphere_hemisphere_volume_ratio_l1333_133385


namespace common_chord_equation_l1333_133326

/-- Given two circles in the xy-plane, this theorem states the equation of the line
    on which their common chord lies. -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 - 10*x - 10*y = 0) →
  (x^2 + y^2 + 6*x - 2*y - 40 = 0) →
  (2*x + y - 5 = 0) :=
by sorry

end common_chord_equation_l1333_133326


namespace final_mixture_percentage_l1333_133343

/-- Percentage of material A in solution X -/
def x_percentage : ℝ := 0.20

/-- Percentage of material A in solution Y -/
def y_percentage : ℝ := 0.30

/-- Percentage of solution X in the final mixture -/
def x_mixture_percentage : ℝ := 0.80

/-- Calculate the percentage of material A in the final mixture -/
def final_percentage : ℝ := x_percentage * x_mixture_percentage + y_percentage * (1 - x_mixture_percentage)

/-- Theorem stating that the percentage of material A in the final mixture is 22% -/
theorem final_mixture_percentage : final_percentage = 0.22 := by
  sorry

end final_mixture_percentage_l1333_133343


namespace ashok_pyarelal_capital_ratio_l1333_133340

/-- Given a total loss and Pyarelal's loss, calculate the ratio of Ashok's capital to Pyarelal's capital -/
theorem ashok_pyarelal_capital_ratio 
  (total_loss : ℕ) 
  (pyarelal_loss : ℕ) 
  (h1 : total_loss = 1600) 
  (h2 : pyarelal_loss = 1440) : 
  ∃ (a p : ℕ), a ≠ 0 ∧ p ≠ 0 ∧ a / p = 1 / 9 := by
  sorry

end ashok_pyarelal_capital_ratio_l1333_133340
