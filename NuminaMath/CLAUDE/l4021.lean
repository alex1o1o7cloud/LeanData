import Mathlib

namespace NUMINAMATH_CALUDE_total_orange_purchase_l4021_402175

def initial_purchase : ℕ := 10
def additional_purchase : ℕ := 5
def num_weeks : ℕ := 3
def doubling_weeks : ℕ := 2

theorem total_orange_purchase :
  let first_week := initial_purchase + additional_purchase
  let subsequent_weeks := 2 * first_week * doubling_weeks
  first_week + subsequent_weeks = 75 := by sorry

end NUMINAMATH_CALUDE_total_orange_purchase_l4021_402175


namespace NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l4021_402150

/-- Represents an ellipse with axes of symmetry on the coordinate axes -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  h : a > 0 ∧ b > 0 ∧ a ≠ b

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation_from_conditions :
  ∀ e : Ellipse,
    e.a + e.b = 9 →  -- Sum of semi-axes is 9 (half of 18)
    e.a^2 - e.b^2 = 9 →  -- Focal distance squared is 9 (6^2 / 4)
    (∀ x y : ℝ, ellipse_equation e x y ↔ 
      (x^2 / 25 + y^2 / 16 = 1 ∨ x^2 / 16 + y^2 / 25 = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l4021_402150


namespace NUMINAMATH_CALUDE_rectangle_triangle_count_l4021_402174

-- Define the structure of our rectangle
structure DividedRectangle where
  horizontal_sections : Nat
  vertical_sections : Nat
  (h_pos : horizontal_sections > 0)
  (v_pos : vertical_sections > 0)

-- Function to calculate the number of triangles
def count_triangles (rect : DividedRectangle) : Nat :=
  sorry

-- Theorem statement
theorem rectangle_triangle_count :
  ∃ (rect : DividedRectangle),
    rect.horizontal_sections = 3 ∧
    rect.vertical_sections = 4 ∧
    count_triangles rect = 148 :=
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_count_l4021_402174


namespace NUMINAMATH_CALUDE_total_weight_BaF2_is_1051_956_l4021_402111

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 18.998

/-- The number of moles of BaF2 -/
def moles_BaF2 : ℝ := 6

/-- The molecular weight of BaF2 in g/mol -/
def molecular_weight_BaF2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_F

/-- The total weight of BaF2 in grams -/
def total_weight_BaF2 : ℝ := molecular_weight_BaF2 * moles_BaF2

/-- Theorem stating that the total weight of 6 moles of BaF2 is 1051.956 g -/
theorem total_weight_BaF2_is_1051_956 : 
  total_weight_BaF2 = 1051.956 := by sorry

end NUMINAMATH_CALUDE_total_weight_BaF2_is_1051_956_l4021_402111


namespace NUMINAMATH_CALUDE_tomatoes_count_l4021_402194

/-- The number of students who suggested adding mashed potatoes -/
def mashed_potatoes : ℕ := 144

/-- The difference between the number of students who suggested mashed potatoes
    and the number of students who suggested tomatoes -/
def difference : ℕ := 65

/-- The number of students who suggested adding tomatoes -/
def tomatoes : ℕ := mashed_potatoes - difference

theorem tomatoes_count : tomatoes = 79 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_count_l4021_402194


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l4021_402157

theorem quadratic_inequality_solutions (b : ℤ) : 
  (∃ (x₁ x₂ x₃ x₄ : ℤ), 
    (∀ x : ℤ, x^2 + b*x + 1 ≤ 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)) →
  b = 4 ∨ b = -4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l4021_402157


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l4021_402170

theorem consecutive_integers_problem (x y z : ℤ) : 
  x = y + 1 → 
  y = z + 1 → 
  x > y → 
  y > z → 
  2*x + 3*y + 3*z = 5*y + 8 → 
  z = 2 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l4021_402170


namespace NUMINAMATH_CALUDE_emilee_earnings_l4021_402187

/-- Given the total earnings and individual earnings of Jermaine and Terrence, 
    calculate Emilee's earnings. -/
theorem emilee_earnings 
  (total : ℕ) 
  (terrence_earnings : ℕ) 
  (jermaine_terrence_diff : ℕ) 
  (h1 : total = 90) 
  (h2 : terrence_earnings = 30) 
  (h3 : jermaine_terrence_diff = 5) : 
  total - (terrence_earnings + (terrence_earnings + jermaine_terrence_diff)) = 25 :=
by
  sorry

#check emilee_earnings

end NUMINAMATH_CALUDE_emilee_earnings_l4021_402187


namespace NUMINAMATH_CALUDE_prob_white_glow_pop_is_12_21_l4021_402143

/-- Represents the color of a kernel -/
inductive KernelColor
| White
| Yellow

/-- Represents the properties of kernels in the bag -/
structure KernelProperties where
  totalWhite : Rat
  totalYellow : Rat
  whiteGlow : Rat
  yellowGlow : Rat
  whiteGlowPop : Rat
  yellowGlowPop : Rat

/-- The given properties of the kernels in the bag -/
def bagProperties : KernelProperties :=
  { totalWhite := 3/4
  , totalYellow := 1/4
  , whiteGlow := 1/2
  , yellowGlow := 3/4
  , whiteGlowPop := 1/2
  , yellowGlowPop := 3/4
  }

/-- The probability that a randomly selected kernel that glows and pops is white -/
def probWhiteGlowPop (props : KernelProperties) : Rat :=
  let whiteGlowPop := props.totalWhite * props.whiteGlow * props.whiteGlowPop
  let yellowGlowPop := props.totalYellow * props.yellowGlow * props.yellowGlowPop
  whiteGlowPop / (whiteGlowPop + yellowGlowPop)

/-- Theorem stating that the probability of selecting a white kernel that glows and pops is 12/21 -/
theorem prob_white_glow_pop_is_12_21 :
  probWhiteGlowPop bagProperties = 12/21 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_glow_pop_is_12_21_l4021_402143


namespace NUMINAMATH_CALUDE_volume_of_sphere_wedge_l4021_402131

/-- The volume of a wedge of a sphere -/
theorem volume_of_sphere_wedge (c : ℝ) (h : c = 12 * Real.pi) :
  let r := c / (2 * Real.pi)
  let sphere_volume := (4 / 3) * Real.pi * r^3
  let wedge_volume := sphere_volume / 4
  wedge_volume = 72 * Real.pi := by
sorry


end NUMINAMATH_CALUDE_volume_of_sphere_wedge_l4021_402131


namespace NUMINAMATH_CALUDE_child_ticket_price_soccer_match_l4021_402188

/-- The price of a child's ticket at a soccer match -/
def child_ticket_price (num_adults num_children : ℕ) (adult_ticket_price total_bill : ℚ) : ℚ :=
  (total_bill - num_adults * adult_ticket_price) / num_children

theorem child_ticket_price_soccer_match :
  child_ticket_price 25 32 12 450 = 469/100 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_price_soccer_match_l4021_402188


namespace NUMINAMATH_CALUDE_donut_area_l4021_402136

/-- The area of a donut shape formed by two concentric circles -/
theorem donut_area (r₁ r₂ : ℝ) (h₁ : r₁ = 7) (h₂ : r₂ = 10) :
  (r₂^2 - r₁^2) * π = 51 * π := by
  sorry

#check donut_area

end NUMINAMATH_CALUDE_donut_area_l4021_402136


namespace NUMINAMATH_CALUDE_matchstick_100th_stage_l4021_402154

/-- Represents the number of matchsticks in each stage of the geometric shape construction -/
def matchstick_sequence : ℕ → ℕ
  | 0 => 4  -- First stage (index 0) has 4 matchsticks
  | n + 1 => matchstick_sequence n + 5  -- Each subsequent stage adds 5 matchsticks

/-- Theorem stating that the 100th stage (index 99) requires 499 matchsticks -/
theorem matchstick_100th_stage : matchstick_sequence 99 = 499 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_100th_stage_l4021_402154


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_m_range_l4021_402144

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The main theorem -/
theorem point_in_fourth_quadrant_m_range (m : ℝ) :
  in_fourth_quadrant ⟨m + 3, m - 5⟩ ↔ -3 < m ∧ m < 5 := by
  sorry


end NUMINAMATH_CALUDE_point_in_fourth_quadrant_m_range_l4021_402144


namespace NUMINAMATH_CALUDE_twelve_percent_greater_than_80_l4021_402186

theorem twelve_percent_greater_than_80 (x : ℝ) : 
  x = 80 * (1 + 12 / 100) → x = 89.6 := by
sorry

end NUMINAMATH_CALUDE_twelve_percent_greater_than_80_l4021_402186


namespace NUMINAMATH_CALUDE_symmetric_distribution_within_one_std_dev_l4021_402196

/-- A symmetric distribution about a mean -/
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percent_less_than_mean_plus_std_dev : ℝ

/-- The percentage of a symmetric distribution within one standard deviation of the mean -/
def percent_within_one_std_dev (dist : SymmetricDistribution) : ℝ :=
  2 * (dist.percent_less_than_mean_plus_std_dev - 50)

theorem symmetric_distribution_within_one_std_dev 
  (dist : SymmetricDistribution) 
  (h1 : dist.is_symmetric = true) 
  (h2 : dist.percent_less_than_mean_plus_std_dev = 82) :
  percent_within_one_std_dev dist = 64 := by
  sorry

#check symmetric_distribution_within_one_std_dev

end NUMINAMATH_CALUDE_symmetric_distribution_within_one_std_dev_l4021_402196


namespace NUMINAMATH_CALUDE_race_distance_l4021_402115

/-- The race problem -/
theorem race_distance (t_A t_B : ℕ) (lead : ℕ) (h1 : t_A = 36) (h2 : t_B = 45) (h3 : lead = 24) :
  ∃ D : ℕ, D = 24 ∧ (D : ℚ) / t_A * t_B = D + lead :=
by sorry

end NUMINAMATH_CALUDE_race_distance_l4021_402115


namespace NUMINAMATH_CALUDE_circle_center_proof_l4021_402118

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation x² - 8x + y² - 4y = 4, prove that its center is (4, 2) -/
theorem circle_center_proof (eq : CircleEquation) 
    (h1 : eq.a = 1)
    (h2 : eq.b = -8)
    (h3 : eq.c = 1)
    (h4 : eq.d = -4)
    (h5 : eq.e = -4) :
    CircleCenter.mk 4 2 = CircleCenter.mk (-eq.b / (2 * eq.a)) (-eq.d / (2 * eq.c)) :=
  sorry

end NUMINAMATH_CALUDE_circle_center_proof_l4021_402118


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l4021_402151

theorem partial_fraction_decomposition_product (x : ℝ) 
  (A B C : ℝ) : 
  (x^2 - 25) / (x^3 - 4*x^2 + x + 6) = 
  A / (x - 3) + B / (x + 1) + C / (x - 2) →
  A * B * C = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l4021_402151


namespace NUMINAMATH_CALUDE_school_students_count_l4021_402125

theorem school_students_count (below_8_percent : ℝ) (age_8_count : ℕ) (above_8_ratio : ℝ) :
  below_8_percent = 0.2 →
  age_8_count = 60 →
  above_8_ratio = 2/3 →
  ∃ (total : ℕ), total = 125 ∧ 
    (total : ℝ) * below_8_percent + (age_8_count : ℝ) + (age_8_count : ℝ) * above_8_ratio = total := by
  sorry

end NUMINAMATH_CALUDE_school_students_count_l4021_402125


namespace NUMINAMATH_CALUDE_student_selection_problem_l4021_402183

theorem student_selection_problem (n m k : ℕ) (hn : n = 10) (hm : m = 2) (hk : k = 4) :
  (Nat.choose n k) - (Nat.choose (n - m) k) = 140 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_problem_l4021_402183


namespace NUMINAMATH_CALUDE_domain_of_shifted_sum_l4021_402162

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc 0 4

-- State the theorem
theorem domain_of_shifted_sum (hf : Set.range f = dom_f) :
  {x : ℝ | ∃ y, y ∈ dom_f ∧ x + 1 = y} ∩ {x : ℝ | ∃ y, y ∈ dom_f ∧ x - 1 = y} = Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_domain_of_shifted_sum_l4021_402162


namespace NUMINAMATH_CALUDE_no_solution_squared_equals_negative_one_l4021_402126

theorem no_solution_squared_equals_negative_one :
  ¬ ∃ x : ℝ, (3*x - 2)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_squared_equals_negative_one_l4021_402126


namespace NUMINAMATH_CALUDE_bankruptcy_division_l4021_402121

/-- Represents the weight of an item -/
structure Weight (α : Type) where
  value : ℕ

/-- Represents the collection of items -/
structure Inventory where
  horns : ℕ
  hooves : ℕ
  weight : ℕ

/-- Represents a person's share of the inventory -/
structure Share where
  horns : ℕ
  hooves : ℕ
  hasWeight : Bool

def totalWeight (inv : Inventory) (w : Weight ℕ) (δ : ℕ) : ℕ :=
  inv.horns * (w.value + δ) + inv.hooves * w.value + inv.weight * (w.value + 2 * δ)

def shareWeight (s : Share) (w : Weight ℕ) (δ : ℕ) : ℕ :=
  s.horns * (w.value + δ) + s.hooves * w.value + (if s.hasWeight then w.value + 2 * δ else 0)

theorem bankruptcy_division (w : Weight ℕ) (δ : ℕ) :
  ∃ (panikovsky balaganov : Share),
    panikovsky.horns + balaganov.horns = 17 ∧
    panikovsky.hooves + balaganov.hooves = 2 ∧
    panikovsky.hasWeight = false ∧
    balaganov.hasWeight = true ∧
    shareWeight panikovsky w δ = shareWeight balaganov w δ ∧
    (panikovsky.horns = 9 ∧ panikovsky.hooves = 2 ∧
     balaganov.horns = 8 ∧ balaganov.hooves = 0) :=
  sorry

end NUMINAMATH_CALUDE_bankruptcy_division_l4021_402121


namespace NUMINAMATH_CALUDE_min_distance_to_line_l4021_402159

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

def line (x y : ℝ) : Prop := x - y - 2 = 0

theorem min_distance_to_line : 
  ∃ (d : ℝ), d = Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), f x = y → 
  d ≤ (|x - y - 2|) / Real.sqrt (1^2 + (-1)^2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l4021_402159


namespace NUMINAMATH_CALUDE_max_height_is_three_l4021_402148

/-- Represents a rectangular prism formed by unit cubes -/
structure RectangularPrism where
  base_area : ℕ
  height : ℕ

/-- The volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℕ :=
  prism.base_area * prism.height

/-- The set of all possible rectangular prisms with a given base area -/
def possible_prisms (base_area : ℕ) (total_cubes : ℕ) : Set RectangularPrism :=
  {prism | prism.base_area = base_area ∧ volume prism ≤ total_cubes}

/-- The theorem stating that the maximum height of a rectangular prism
    with base area 4 and 12 total cubes is 3 -/
theorem max_height_is_three :
  ∀ (prism : RectangularPrism),
    prism ∈ possible_prisms 4 12 →
    prism.height ≤ 3 ∧
    ∃ (max_prism : RectangularPrism),
      max_prism ∈ possible_prisms 4 12 ∧
      max_prism.height = 3 :=
sorry

end NUMINAMATH_CALUDE_max_height_is_three_l4021_402148


namespace NUMINAMATH_CALUDE_gcf_of_180_and_270_l4021_402152

theorem gcf_of_180_and_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_and_270_l4021_402152


namespace NUMINAMATH_CALUDE_oil_tank_explosion_theorem_l4021_402110

/-- The number of bullets available -/
def num_bullets : ℕ := 5

/-- The probability of hitting the target on each shot -/
def hit_probability : ℚ := 2/3

/-- The probability of the oil tank exploding -/
def explosion_probability : ℚ := 232/243

/-- The probability that the number of shots is not less than 4 -/
def shots_ge_4_probability : ℚ := 7/27

/-- Each shot is independent and the probability of hitting each time is 2/3.
    The first hit causes oil to flow out, and the second hit causes an explosion.
    Shooting stops when the oil tank explodes or bullets run out. -/
theorem oil_tank_explosion_theorem :
  (∀ (n : ℕ), n ≤ num_bullets → (hit_probability^n * (1 - hit_probability)^(num_bullets - n) : ℚ) = (2/3)^n * (1/3)^(num_bullets - n)) →
  explosion_probability = 232/243 ∧
  shots_ge_4_probability = 7/27 :=
sorry

end NUMINAMATH_CALUDE_oil_tank_explosion_theorem_l4021_402110


namespace NUMINAMATH_CALUDE_green_bay_high_relay_race_length_l4021_402127

/-- Calculates the total length of a relay race given the number of team members and the distance each member runs. -/
def relay_race_length (team_members : ℕ) (distance_per_member : ℕ) : ℕ :=
  team_members * distance_per_member

/-- Theorem stating that a relay race with 5 team members, each running 30 meters, has a total length of 150 meters. -/
theorem green_bay_high_relay_race_length :
  relay_race_length 5 30 = 150 := by
  sorry

#eval relay_race_length 5 30

end NUMINAMATH_CALUDE_green_bay_high_relay_race_length_l4021_402127


namespace NUMINAMATH_CALUDE_middle_managers_sample_size_l4021_402197

/-- Calculates the number of individuals to be selected from a stratum in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (total_sample : ℕ) (stratum_size : ℕ) : ℕ :=
  (total_sample * stratum_size) / total_population

/-- Proves that the number of middle managers to be selected is 6 -/
theorem middle_managers_sample_size :
  stratified_sample_size 160 32 30 = 6 := by
  sorry

#eval stratified_sample_size 160 32 30

end NUMINAMATH_CALUDE_middle_managers_sample_size_l4021_402197


namespace NUMINAMATH_CALUDE_divisor_sum_difference_bound_l4021_402161

/-- Sum of counts of positive even divisors of numbers from 1 to n -/
def D1 (n : ℕ) : ℕ := sorry

/-- Sum of counts of positive odd divisors of numbers from 1 to n -/
def D2 (n : ℕ) : ℕ := sorry

/-- The difference between D2 and D1 is no greater than n -/
theorem divisor_sum_difference_bound (n : ℕ) : D2 n - D1 n ≤ n := by sorry

end NUMINAMATH_CALUDE_divisor_sum_difference_bound_l4021_402161


namespace NUMINAMATH_CALUDE_area_equality_l4021_402192

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the shapes
def is_convex_quadrilateral (A C D F : ℝ × ℝ) : Prop := sorry
def is_equilateral_triangle (A B E : ℝ × ℝ) : Prop := sorry
def is_square (A C D F : ℝ × ℝ) : Prop := sorry
def is_rectangle (A C D F : ℝ × ℝ) : Prop := sorry

-- Define the point on side condition
def point_on_side (P Q R : ℝ × ℝ) : Prop := sorry

-- Define the area calculation function
def area_triangle (P Q R : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem area_equality 
  (h_quad : is_convex_quadrilateral A C D F)
  (h_tri : is_equilateral_triangle A B E)
  (h_common : A = A)  -- Common vertex
  (h_B_on_CF : point_on_side B C F)
  (h_E_on_FD : point_on_side E F D)
  (h_shape : is_square A C D F ∨ is_rectangle A C D F) :
  area_triangle A D E + area_triangle A B C = area_triangle B E F := by
  sorry

end NUMINAMATH_CALUDE_area_equality_l4021_402192


namespace NUMINAMATH_CALUDE_nearest_integer_to_power_l4021_402137

theorem nearest_integer_to_power : 
  ∃ n : ℤ, |n - (3 + Real.sqrt 5)^6| < 1/2 ∧ n = 2744 :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_power_l4021_402137


namespace NUMINAMATH_CALUDE_min_removals_for_no_products_l4021_402160

theorem min_removals_for_no_products (n : ℕ) (hn : n = 1982) :
  ∃ (S : Finset ℕ),
    S.card = 43 ∧ 
    (∀ k ∈ Finset.range (n + 1) \ S, k = 1 ∨ k ≥ 45) ∧
    (∀ a b k, a ∈ Finset.range (n + 1) \ S → b ∈ Finset.range (n + 1) \ S → 
      k ∈ Finset.range (n + 1) \ S → a ≠ b → a * b ≠ k) ∧
    (∀ T : Finset ℕ, T.card < 43 → 
      ∃ a b k, a ∈ Finset.range (n + 1) \ T → b ∈ Finset.range (n + 1) \ T → 
        k ∈ Finset.range (n + 1) \ T → a ≠ b → a * b = k) :=
by sorry

end NUMINAMATH_CALUDE_min_removals_for_no_products_l4021_402160


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l4021_402167

theorem complex_fraction_equality : 2 * (1 + 1 / (1 - 1 / (2 + 2))) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l4021_402167


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_four_l4021_402124

/-- Given acute angles x and y satisfying specific trigonometric equations,
    prove that their combination 2x + y equals π/4 radians. -/
theorem angle_sum_is_pi_over_four (x y : Real) (hx : 0 < x ∧ x < π/2) (hy : 0 < y ∧ y < π/2)
  (h1 : 2 * Real.cos x ^ 2 + 3 * Real.cos y ^ 2 = 1)
  (h2 : 2 * Real.sin (2 * x) + 3 * Real.sin (2 * y) = 0) :
  2 * x + y = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_four_l4021_402124


namespace NUMINAMATH_CALUDE_shoe_probability_l4021_402173

def total_pairs : ℕ := 20
def black_pairs : ℕ := 8
def brown_pairs : ℕ := 5
def gray_pairs : ℕ := 3
def white_pairs : ℕ := 4

theorem shoe_probability :
  let total_shoes := total_pairs * 2
  let prob_black := (black_pairs * 2 / total_shoes) * (black_pairs / (total_shoes - 1))
  let prob_brown := (brown_pairs * 2 / total_shoes) * (brown_pairs / (total_shoes - 1))
  let prob_gray := (gray_pairs * 2 / total_shoes) * (gray_pairs / (total_shoes - 1))
  let prob_white := (white_pairs * 2 / total_shoes) * (white_pairs / (total_shoes - 1))
  prob_black + prob_brown + prob_gray + prob_white = 19 / 130 := by
  sorry

end NUMINAMATH_CALUDE_shoe_probability_l4021_402173


namespace NUMINAMATH_CALUDE_lion_path_angles_l4021_402139

theorem lion_path_angles (r : ℝ) (path_length : ℝ) (turn_angles : List ℝ) : 
  r = 10 →
  path_length = 30000 →
  path_length ≤ 2 * r + r * (turn_angles.sum) →
  turn_angles.sum ≥ 2998 := by
sorry

end NUMINAMATH_CALUDE_lion_path_angles_l4021_402139


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_first_vessel_l4021_402190

/-- Proves that the percentage of alcohol in the first vessel is 25% --/
theorem alcohol_percentage_in_first_vessel : 
  ∀ (x : ℝ),
  -- Vessel capacities and total liquid
  let vessel1_capacity : ℝ := 2
  let vessel2_capacity : ℝ := 6
  let total_liquid : ℝ := 8
  let final_vessel_capacity : ℝ := 10
  -- Alcohol percentages
  let vessel2_alcohol_percentage : ℝ := 50
  let final_mixture_percentage : ℝ := 35
  -- Condition: total alcohol in final mixture
  (x / 100) * vessel1_capacity + (vessel2_alcohol_percentage / 100) * vessel2_capacity = 
    (final_mixture_percentage / 100) * final_vessel_capacity →
  -- Conclusion: alcohol percentage in first vessel is 25%
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_first_vessel_l4021_402190


namespace NUMINAMATH_CALUDE_stationery_shop_sales_l4021_402171

theorem stationery_shop_sales (total_sales percent_pens percent_pencils : ℝ) 
  (h_total : total_sales = 100)
  (h_pens : percent_pens = 38)
  (h_pencils : percent_pencils = 35) :
  total_sales - percent_pens - percent_pencils = 27 := by
  sorry

end NUMINAMATH_CALUDE_stationery_shop_sales_l4021_402171


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l4021_402176

/-- Converts a binary number (represented as a list of bits) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The given binary numbers -/
def b1 : List Bool := [true, false, true, true]    -- 1101₂
def b2 : List Bool := [true, true, true]           -- 111₂
def b3 : List Bool := [false, true, true, true]    -- 1110₂
def b4 : List Bool := [true, false, false, true]   -- 1001₂
def b5 : List Bool := [false, true, false, true]   -- 1010₂

/-- The result binary number -/
def result : List Bool := [true, false, false, true, true]  -- 11001₂

theorem binary_addition_subtraction :
  binary_to_decimal b1 + binary_to_decimal b2 - binary_to_decimal b3 + 
  binary_to_decimal b4 + binary_to_decimal b5 = binary_to_decimal result := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l4021_402176


namespace NUMINAMATH_CALUDE_room_length_is_five_l4021_402181

/-- Given a rectangular room with known width, total paving cost, and paving rate per square meter,
    prove that the length of the room is 5 meters. -/
theorem room_length_is_five (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  width = 4.75 →
  total_cost = 21375 →
  rate_per_sqm = 900 →
  (total_cost / rate_per_sqm) / width = 5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_is_five_l4021_402181


namespace NUMINAMATH_CALUDE_greatest_common_multiple_under_120_l4021_402101

theorem greatest_common_multiple_under_120 : ∃ (n : ℕ), n = 90 ∧ 
  (∀ m : ℕ, m < 120 → m % 10 = 0 → m % 15 = 0 → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_under_120_l4021_402101


namespace NUMINAMATH_CALUDE_sequence_gcd_property_l4021_402106

theorem sequence_gcd_property (a : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) →
  ∀ i : ℕ, a i = i := by
sorry

end NUMINAMATH_CALUDE_sequence_gcd_property_l4021_402106


namespace NUMINAMATH_CALUDE_second_player_wins_l4021_402184

/-- Represents a position on an 8x8 chessboard -/
def Position := Fin 8 × Fin 8

/-- Represents a knight's move -/
def KnightMove := List (Int × Int)

/-- The list of possible knight moves -/
def knightMoves : KnightMove :=
  [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]

/-- Checks if a position is valid on the 8x8 board -/
def isValidPosition (p : Position) : Bool := true

/-- Checks if two positions are a knight's move apart -/
def isKnightMove (p1 p2 : Position) : Bool := sorry

/-- Represents the state of the game -/
structure GameState where
  placedKnights : List Position
  currentPlayer : Nat

/-- Checks if a move is legal given the current game state -/
def isLegalMove (state : GameState) (move : Position) : Bool := sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Option Position

/-- Checks if a strategy is winning for a player -/
def isWinningStrategy (player : Nat) (strat : Strategy) : Prop := sorry

/-- The main theorem: the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (strat : Strategy), isWinningStrategy 1 strat := sorry

end NUMINAMATH_CALUDE_second_player_wins_l4021_402184


namespace NUMINAMATH_CALUDE_odd_function_zero_l4021_402117

/-- Definition of an odd function -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Theorem: For an odd function f defined at 0, f(0) = 0 -/
theorem odd_function_zero (f : ℝ → ℝ) (h : IsOdd f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_zero_l4021_402117


namespace NUMINAMATH_CALUDE_tan_alpha_neg_half_implies_expression_eq_neg_third_l4021_402120

theorem tan_alpha_neg_half_implies_expression_eq_neg_third (α : Real) 
  (h : Real.tan α = -1/2) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_neg_half_implies_expression_eq_neg_third_l4021_402120


namespace NUMINAMATH_CALUDE_f_symmetry_l4021_402191

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^7 + a*x^5 + b*x - 5

-- State the theorem
theorem f_symmetry (a b : ℝ) : f a b (-3) = 5 → f a b 3 = -15 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l4021_402191


namespace NUMINAMATH_CALUDE_range_of_k_l4021_402108

-- Define the condition function
def condition (x : ℝ) : Prop := 3 / (x + 1) < 1

-- Define the sufficient condition
def sufficient_condition (k : ℝ) : Prop := ∀ x, x > k → condition x

-- Define the not necessary condition
def not_necessary_condition (k : ℝ) : Prop := ∃ x, condition x ∧ x ≤ k

-- State the theorem
theorem range_of_k :
  ∀ k, (sufficient_condition k ∧ not_necessary_condition k) ↔ k ∈ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l4021_402108


namespace NUMINAMATH_CALUDE_lemonade_sales_difference_l4021_402165

/-- Anna's lemonade sales problem -/
theorem lemonade_sales_difference :
  let plain_glasses : ℕ := 36
  let plain_price : ℚ := 3/4  -- $0.75 represented as a rational number
  let strawberry_earnings : ℚ := 16
  let plain_earnings := plain_glasses * plain_price
  plain_earnings - strawberry_earnings = 11 := by sorry

end NUMINAMATH_CALUDE_lemonade_sales_difference_l4021_402165


namespace NUMINAMATH_CALUDE_mary_lamb_count_l4021_402132

/-- Calculates the final number of lambs Mary has given the initial conditions. -/
def final_lamb_count (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) 
  (traded_lambs : ℕ) (extra_lambs : ℕ) : ℕ :=
  initial_lambs + lambs_with_babies * babies_per_lamb - traded_lambs + extra_lambs

/-- Proves that Mary ends up with 14 lambs given the initial conditions. -/
theorem mary_lamb_count : 
  final_lamb_count 6 2 2 3 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_mary_lamb_count_l4021_402132


namespace NUMINAMATH_CALUDE_prob_two_ones_twelve_dice_l4021_402128

/-- The number of dice rolled -/
def n : ℕ := 12

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice we want to show a specific result -/
def k : ℕ := 2

/-- The probability of rolling exactly k ones out of n dice -/
def prob_k_ones (n k : ℕ) : ℚ :=
  (n.choose k : ℚ) * (1 / sides)^k * (1 - 1 / sides)^(n - k)

theorem prob_two_ones_twelve_dice : 
  prob_k_ones n k = (66 * 5^10 : ℚ) / 6^12 := by sorry

end NUMINAMATH_CALUDE_prob_two_ones_twelve_dice_l4021_402128


namespace NUMINAMATH_CALUDE_book_cost_calculation_l4021_402164

theorem book_cost_calculation (num_books : ℕ) (money_have : ℕ) (money_save : ℕ) :
  num_books = 8 ∧ money_have = 13 ∧ money_save = 27 →
  (money_have + money_save) / num_books = 5 := by
sorry

end NUMINAMATH_CALUDE_book_cost_calculation_l4021_402164


namespace NUMINAMATH_CALUDE_article_price_before_discount_l4021_402182

/-- 
Given an article whose price after a 24% decrease is 988 rupees, 
prove that its original price was 1300 rupees.
-/
theorem article_price_before_discount (price_after_discount : ℝ) 
  (h1 : price_after_discount = 988) 
  (h2 : price_after_discount = 0.76 * (original_price : ℝ)) : 
  original_price = 1300 := by
  sorry

end NUMINAMATH_CALUDE_article_price_before_discount_l4021_402182


namespace NUMINAMATH_CALUDE_number_categorization_l4021_402178

/-- Define the set of numbers we're working with -/
def numbers : Set ℚ := {-3.14, 22/7, 0, 2023}

/-- Define the set of negative rational numbers -/
def negative_rationals : Set ℚ := {x : ℚ | x < 0}

/-- Define the set of positive fractions -/
def positive_fractions : Set ℚ := {x : ℚ | x > 0 ∧ x ≠ ⌊x⌋}

/-- Define the set of non-negative integers -/
def non_negative_integers : Set ℤ := {x : ℤ | x ≥ 0}

/-- Define the set of natural numbers (including 0) -/
def natural_numbers : Set ℕ := Set.univ

/-- Theorem stating the categorization of the given numbers -/
theorem number_categorization :
  (-3.14 ∈ negative_rationals) ∧
  (22/7 ∈ positive_fractions) ∧
  (0 ∈ non_negative_integers) ∧
  (2023 ∈ non_negative_integers) ∧
  (0 ∈ natural_numbers) ∧
  (2023 ∈ natural_numbers) :=
by sorry

end NUMINAMATH_CALUDE_number_categorization_l4021_402178


namespace NUMINAMATH_CALUDE_negation_equivalence_l4021_402168

theorem negation_equivalence (a b : ℝ) :
  ¬(a ≤ 2 ∧ b ≤ 2) ↔ (a > 2 ∨ b > 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4021_402168


namespace NUMINAMATH_CALUDE_possible_values_of_P_zero_l4021_402140

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The property that the polynomial P must satisfy -/
def SatisfiesProperty (P : RealPolynomial) : Prop :=
  ∀ x y : ℝ, |y^2 - P x| ≤ 2 * |x| ↔ |x^2 - P y| ≤ 2 * |y|

/-- The theorem stating the possible values of P(0) -/
theorem possible_values_of_P_zero (P : RealPolynomial) 
  (h : SatisfiesProperty P) : 
  P 0 < 0 ∨ P 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_P_zero_l4021_402140


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l4021_402146

theorem diophantine_equation_solutions : 
  ∀ a b c : ℕ+, 
  (8 * a.val - 5 * b.val)^2 + (3 * b.val - 2 * c.val)^2 + (3 * c.val - 7 * a.val)^2 = 2 ↔ 
  ((a.val = 3 ∧ b.val = 5 ∧ c.val = 7) ∨ (a.val = 12 ∧ b.val = 19 ∧ c.val = 28)) :=
by sorry

#check diophantine_equation_solutions

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l4021_402146


namespace NUMINAMATH_CALUDE_exists_digit_sum_div_11_l4021_402129

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Theorem: Among 39 consecutive natural numbers, there is always at least one number
    whose sum of digits is divisible by 11. -/
theorem exists_digit_sum_div_11 (n : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (digit_sum (n + k) % 11 = 0) := by sorry

end NUMINAMATH_CALUDE_exists_digit_sum_div_11_l4021_402129


namespace NUMINAMATH_CALUDE_triangle_parallelogram_altitude_l4021_402113

theorem triangle_parallelogram_altitude (b h_t h_p : ℝ) : 
  b > 0 →  -- Ensure base is positive
  h_t = 200 →  -- Given altitude of triangle
  (1 / 2) * b * h_t = b * h_p →  -- Equal areas
  h_p = 100 := by
sorry

end NUMINAMATH_CALUDE_triangle_parallelogram_altitude_l4021_402113


namespace NUMINAMATH_CALUDE_power_of_product_l4021_402185

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l4021_402185


namespace NUMINAMATH_CALUDE_f_maximum_properties_l4021_402198

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem f_maximum_properties (x₀ : ℝ) 
  (h₁ : ∀ x > 0, f x ≤ f x₀) 
  (h₂ : x₀ > 0) : 
  f x₀ = x₀ ∧ f x₀ < (1/2) := by
  sorry

end NUMINAMATH_CALUDE_f_maximum_properties_l4021_402198


namespace NUMINAMATH_CALUDE_fraction_equality_proof_l4021_402102

theorem fraction_equality_proof (x : ℝ) : 
  x ≠ 4 ∧ x ≠ 8 → ((x - 3) / (x - 4) = (x - 5) / (x - 8) ↔ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_proof_l4021_402102


namespace NUMINAMATH_CALUDE_geometric_with_arithmetic_subsequence_l4021_402180

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define an arithmetic subsequence
def arithmetic_subsequence (a : ℕ → ℝ) (sub : ℕ → ℕ) (d : ℝ) : Prop :=
  ∀ k : ℕ, a (sub (k + 1)) = a (sub k) + d

-- Main theorem
theorem geometric_with_arithmetic_subsequence
  (a : ℕ → ℝ) (q : ℝ) (sub : ℕ → ℕ) (d : ℝ) :
  geometric_sequence a q →
  q ≠ 1 →
  arithmetic_subsequence a sub d →
  (∀ k : ℕ, sub (k + 1) > sub k) →
  q = -1 :=
sorry

end NUMINAMATH_CALUDE_geometric_with_arithmetic_subsequence_l4021_402180


namespace NUMINAMATH_CALUDE_reading_pattern_equation_l4021_402172

/-- Represents the total number of words in the book "Mencius" -/
def total_words : ℕ := 34685

/-- Represents the number of days taken to read the book -/
def days : ℕ := 3

/-- Represents the relationship between words read on consecutive days -/
def daily_increase_factor : ℕ := 2

/-- Theorem stating the correct equation for the reading pattern -/
theorem reading_pattern_equation (x : ℕ) :
  x + daily_increase_factor * x + daily_increase_factor^2 * x = total_words →
  x + 2*x + 4*x = total_words :=
by sorry

end NUMINAMATH_CALUDE_reading_pattern_equation_l4021_402172


namespace NUMINAMATH_CALUDE_photo_arrangements_l4021_402163

/-- The number of male students -/
def num_male : ℕ := 5

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- Calculates the number of arrangements with female student A at one end -/
def arrangements_a_at_end : ℕ := sorry

/-- Calculates the number of arrangements with both female students not at the ends -/
def arrangements_females_not_at_ends : ℕ := sorry

/-- Calculates the number of arrangements with the two female students not adjacent -/
def arrangements_females_not_adjacent : ℕ := sorry

/-- Calculates the number of arrangements with female student A on the right side of female student B -/
def arrangements_a_right_of_b : ℕ := sorry

theorem photo_arrangements :
  arrangements_a_at_end = 1440 ∧
  arrangements_females_not_at_ends = 2400 ∧
  arrangements_females_not_adjacent = 3600 ∧
  arrangements_a_right_of_b = 2520 := by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l4021_402163


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4021_402100

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum of the first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h1 : seq.a 2 + seq.a 6 = 2)
    (h2 : seq.S 9 = -18) :
    (∀ n, seq.a n = 13 - 3*n) ∧
    (∀ n, |seq.S n| ≥ |seq.S 8|) ∧
    (|seq.S 8| = 4) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4021_402100


namespace NUMINAMATH_CALUDE_phoenix_temperature_l4021_402138

theorem phoenix_temperature (t : ℝ) : 
  (∀ s, -s^2 + 14*s + 40 = 77 → s ≤ t) → -t^2 + 14*t + 40 = 77 → t = 11 := by
  sorry

end NUMINAMATH_CALUDE_phoenix_temperature_l4021_402138


namespace NUMINAMATH_CALUDE_line_symmetry_l4021_402119

-- Define the original line
def original_line (x y : ℝ) : Prop := 2 * x - y = 0

-- Define symmetry with respect to x-axis
def symmetric_wrt_x_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, g x = -f x

-- Define the proposed symmetric line
def proposed_symmetric_line (x y : ℝ) : Prop := 2 * x + y = 0

-- Theorem statement
theorem line_symmetry :
  ∃ (f g : ℝ → ℝ),
    (∀ x y, original_line x y ↔ y = f x) ∧
    (∀ x y, proposed_symmetric_line x y ↔ y = g x) ∧
    symmetric_wrt_x_axis f g :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l4021_402119


namespace NUMINAMATH_CALUDE_congruence_problem_l4021_402133

theorem congruence_problem (x : ℤ) : 
  x ≡ 1 [ZMOD 27] ∧ x ≡ 6 [ZMOD 37] → x ≡ 110 [ZMOD 999] := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l4021_402133


namespace NUMINAMATH_CALUDE_cloth_meters_sold_l4021_402134

/-- Proves that the number of meters of cloth sold is 66, given the selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_meters_sold
  (selling_price : ℕ)
  (profit_per_meter : ℕ)
  (cost_per_meter : ℕ)
  (h1 : selling_price = 660)
  (h2 : profit_per_meter = 5)
  (h3 : cost_per_meter = 5) :
  selling_price / (profit_per_meter + cost_per_meter) = 66 := by
  sorry

#check cloth_meters_sold

end NUMINAMATH_CALUDE_cloth_meters_sold_l4021_402134


namespace NUMINAMATH_CALUDE_six_digit_number_theorem_l4021_402103

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ (a b c d e f : ℕ),
    n = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧
    a ≠ 0 ∧ 
    10000 * b + 1000 * c + 100 * d + 10 * e + f + 100000 * a = 3 * n

theorem six_digit_number_theorem :
  ∀ n : ℕ, is_valid_number n → (n = 142857 ∨ n = 285714) :=
sorry

end NUMINAMATH_CALUDE_six_digit_number_theorem_l4021_402103


namespace NUMINAMATH_CALUDE_merchant_profit_l4021_402195

theorem merchant_profit (C S : ℝ) (h : C > 0) (h1 : 18 * C = 16 * S) : 
  (S - C) / C * 100 = 12.5 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_l4021_402195


namespace NUMINAMATH_CALUDE_greatest_integer_2pi_minus_6_l4021_402179

theorem greatest_integer_2pi_minus_6 :
  Int.floor (2 * Real.pi - 6) = 0 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_2pi_minus_6_l4021_402179


namespace NUMINAMATH_CALUDE_special_triangle_sides_l4021_402199

/-- A triangle with specific properties -/
structure SpecialTriangle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The altitudes (heights) of the triangle
  ha : ℕ
  hb : ℕ
  hc : ℕ
  -- The radius of the inscribed circle
  r : ℝ
  -- Conditions
  radius_condition : r = 4/3
  altitudes_sum : ha + hb + hc = 13
  altitude_relation : 1/ha + 1/hb + 1/hc = 3/4

/-- Theorem about the side lengths of the special triangle -/
theorem special_triangle_sides (t : SpecialTriangle) : 
  t.a = 32 / Real.sqrt 15 ∧ 
  t.b = 24 / Real.sqrt 15 ∧ 
  t.c = 16 / Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_sides_l4021_402199


namespace NUMINAMATH_CALUDE_inner_circle_distance_l4021_402114

/-- A right triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_lengths : a = 9 ∧ b = 12 ∧ c = 15

/-- The path of the center of a circle rolling inside the triangle -/
def inner_circle_path (t : RightTriangle) (r : ℝ) : ℝ := 
  (t.a - 2*r) + (t.b - 2*r) + (t.c - 2*r)

/-- The theorem to be proved -/
theorem inner_circle_distance (t : RightTriangle) : 
  inner_circle_path t 2 = 24 := by sorry

end NUMINAMATH_CALUDE_inner_circle_distance_l4021_402114


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4021_402107

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence a_n where a_3 + a_5 = 20 and a_4 = 8, prove that a_2 + a_6 = 34 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geo : IsGeometricSequence a)
    (h_sum : a 3 + a 5 = 20) (h_fourth : a 4 = 8) : a 2 + a 6 = 34 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4021_402107


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l4021_402104

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes needed to fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem stating that the smallest number of identical cubes needed to fill the given box is 90 -/
theorem smallest_number_of_cubes_for_given_box :
  smallestNumberOfCubes ⟨27, 15, 6⟩ = 90 := by
  sorry

#eval smallestNumberOfCubes ⟨27, 15, 6⟩

end NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l4021_402104


namespace NUMINAMATH_CALUDE_three_sevenths_decomposition_l4021_402149

theorem three_sevenths_decomposition :
  3 / 7 = 1 / 8 + 1 / 56 + 1 / 9 + 1 / 72 := by
  sorry

#check three_sevenths_decomposition

end NUMINAMATH_CALUDE_three_sevenths_decomposition_l4021_402149


namespace NUMINAMATH_CALUDE_book_dimensions_and_area_l4021_402177

/-- Represents the dimensions and surface area of a book. -/
structure Book where
  L : ℝ  -- Length
  W : ℝ  -- Width
  T : ℝ  -- Thickness
  A1 : ℝ  -- Area of front cover
  A2 : ℝ  -- Area of spine
  S : ℝ  -- Total surface area

/-- Theorem stating the width and total surface area of a book with given dimensions. -/
theorem book_dimensions_and_area (b : Book) 
  (hL : b.L = 5)
  (hT : b.T = 2)
  (hA1 : b.A1 = 50)
  (hA1_eq : b.A1 = b.L * b.W)
  (hA2_eq : b.A2 = b.T * b.W)
  (hS_eq : b.S = 2 * b.A1 + b.A2 + 2 * (b.L * b.T)) :
  b.W = 10 ∧ b.S = 140 := by
  sorry

#check book_dimensions_and_area

end NUMINAMATH_CALUDE_book_dimensions_and_area_l4021_402177


namespace NUMINAMATH_CALUDE_passengers_from_other_continents_l4021_402105

theorem passengers_from_other_continents 
  (total : ℕ) 
  (h_total : total = 240) 
  (h_na : total / 3 = 80) 
  (h_eu : total / 8 = 30) 
  (h_af : total / 5 = 48) 
  (h_as : total / 6 = 40) : 
  total - (total / 3 + total / 8 + total / 5 + total / 6) = 42 := by
  sorry

end NUMINAMATH_CALUDE_passengers_from_other_continents_l4021_402105


namespace NUMINAMATH_CALUDE_limit_of_sequence_l4021_402135

/-- The sequence a_n defined as (1 + 3n) / (6 - n) converges to -3 as n approaches infinity. -/
theorem limit_of_sequence (ε : ℝ) (hε : ε > 0) : 
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → |((1 : ℝ) + 3 * n) / (6 - n) + 3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l4021_402135


namespace NUMINAMATH_CALUDE_balloon_permutations_l4021_402169

def balloon_letters : Nat := 7
def balloon_l_count : Nat := 2
def balloon_o_count : Nat := 2

theorem balloon_permutations :
  (balloon_letters.factorial) / (balloon_l_count.factorial * balloon_o_count.factorial) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_permutations_l4021_402169


namespace NUMINAMATH_CALUDE_system_solvability_l4021_402153

-- Define the system of equations
def system (x y a b : ℝ) : Prop :=
  x * Real.cos a + y * Real.sin a + 4 ≤ 0 ∧
  x^2 + y^2 + 10*x + 2*y - b^2 - 8*b + 10 = 0

-- Define the set of valid b values
def valid_b_set (b : ℝ) : Prop :=
  b ≤ -8 - Real.sqrt 26 ∨ b ≥ Real.sqrt 26

-- Theorem statement
theorem system_solvability (b : ℝ) :
  (∀ a, ∃ x y, system x y a b) ↔ valid_b_set b :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l4021_402153


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4021_402109

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (0 ≤ x ∧ x < 1) ↔ (x + 2*a ≥ 4 ∧ (2*x - b) / 3 < 1)) →
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4021_402109


namespace NUMINAMATH_CALUDE_carousel_horses_l4021_402130

theorem carousel_horses (blue purple green gold : ℕ) : 
  purple = 3 * blue →
  green = 2 * purple →
  gold = green / 6 →
  blue + purple + green + gold = 33 →
  blue = 3 := by
sorry

end NUMINAMATH_CALUDE_carousel_horses_l4021_402130


namespace NUMINAMATH_CALUDE_rook_paths_eq_catalan_l4021_402156

/-- The number of valid paths for a rook on an n × n chessboard -/
def rookPaths (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (Nat.choose (2 * n - 2) (n - 1)) / n

/-- The Catalan number C_n -/
def catalanNumber (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (Nat.choose (2 * n) n) / (n + 1)

/-- Theorem: The number of valid rook paths on an n × n chessboard
    is equal to the (n-1)th Catalan number -/
theorem rook_paths_eq_catalan (n : ℕ) :
  rookPaths n = catalanNumber (n - 1) :=
sorry

end NUMINAMATH_CALUDE_rook_paths_eq_catalan_l4021_402156


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l4021_402158

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (f x + f y + f z) = f (f x - f y) + f (2 * x * y + f z) + 2 * f (x * z - y * z)

/-- The main theorem stating that functions satisfying the equation are either constant zero or square. -/
theorem functional_equation_solutions (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    (∀ x, f x = 0) ∨ (∀ x, f x = x^2) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l4021_402158


namespace NUMINAMATH_CALUDE_star_eight_four_l4021_402141

-- Define the & operation
def amp (a b : ℝ) : ℝ := (a + b) * (a - b)

-- Define the ★ operation
def star (c d : ℝ) : ℝ := amp c d + 2 * (c + d)

-- Theorem to prove
theorem star_eight_four : star 8 4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_star_eight_four_l4021_402141


namespace NUMINAMATH_CALUDE_correct_calculation_l4021_402189

def correct_sum (mistaken_sum : ℕ) (original_tens : ℕ) (mistaken_tens : ℕ) 
                (original_units : ℕ) (mistaken_units : ℕ) : ℕ :=
  mistaken_sum - (mistaken_units - original_units) + (original_tens - mistaken_tens) * 10

theorem correct_calculation (mistaken_sum : ℕ) (original_tens : ℕ) (mistaken_tens : ℕ) 
                            (original_units : ℕ) (mistaken_units : ℕ) : 
  mistaken_sum = 111 ∧ 
  original_tens = 7 ∧ 
  mistaken_tens = 4 ∧ 
  original_units = 5 ∧ 
  mistaken_units = 8 → 
  correct_sum mistaken_sum original_tens mistaken_tens original_units mistaken_units = 138 := by
  sorry

#eval correct_sum 111 7 4 5 8

end NUMINAMATH_CALUDE_correct_calculation_l4021_402189


namespace NUMINAMATH_CALUDE_average_of_data_set_l4021_402166

def data_set : List ℤ := [7, 5, -2, 5, 10]

theorem average_of_data_set :
  (data_set.sum : ℚ) / data_set.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_data_set_l4021_402166


namespace NUMINAMATH_CALUDE_area_ratio_midpoint_quadrilateral_l4021_402145

/-- A quadrilateral in a 2D plane --/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The area of a quadrilateral --/
def area (q : Quadrilateral) : ℝ := sorry

/-- The quadrilateral formed by the midpoints of another quadrilateral's sides --/
def midpointQuadrilateral (q : Quadrilateral) : Quadrilateral := sorry

/-- Theorem: The area of a quadrilateral is twice the area of its midpoint quadrilateral --/
theorem area_ratio_midpoint_quadrilateral (q : Quadrilateral) : 
  area q = 2 * area (midpointQuadrilateral q) := by sorry

end NUMINAMATH_CALUDE_area_ratio_midpoint_quadrilateral_l4021_402145


namespace NUMINAMATH_CALUDE_f_inequality_l4021_402123

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + a * x

theorem f_inequality (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) :
  (f a x₁ - f a x₂) / (x₂ - x₁) > 
  (Real.log ((x₁ + x₂) / 2) - a * ((x₁ + x₂) / 2) + a) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_l4021_402123


namespace NUMINAMATH_CALUDE_product_of_fractions_l4021_402155

theorem product_of_fractions : (2 : ℚ) / 3 * (5 : ℚ) / 11 = 10 / 33 := by sorry

end NUMINAMATH_CALUDE_product_of_fractions_l4021_402155


namespace NUMINAMATH_CALUDE_plane_Q_satisfies_conditions_l4021_402142

def plane1 (x y z : ℝ) : ℝ := 2*x - y + 2*z - 4
def plane2 (x y z : ℝ) : ℝ := 3*x + y - z - 6
def planeQ (x y z : ℝ) : ℝ := 19*x - 67*y + 109*z - 362

def point : ℝ × ℝ × ℝ := (2, 0, 3)

theorem plane_Q_satisfies_conditions :
  (∀ x y z : ℝ, plane1 x y z = 0 ∧ plane2 x y z = 0 → planeQ x y z = 0) ∧ 
  (planeQ ≠ plane1 ∧ planeQ ≠ plane2) ∧
  (let (x₀, y₀, z₀) := point
   abs (19*x₀ - 67*y₀ + 109*z₀ - 362) / Real.sqrt (19^2 + (-67)^2 + 109^2) = 3 / Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_plane_Q_satisfies_conditions_l4021_402142


namespace NUMINAMATH_CALUDE_johns_house_nails_l4021_402112

/-- Calculates the total number of nails needed for a house wall -/
def total_nails (large_planks : ℕ) (nails_per_plank : ℕ) (additional_nails : ℕ) : ℕ :=
  large_planks * nails_per_plank + additional_nails

/-- Proves that John needs 229 nails for the house wall -/
theorem johns_house_nails :
  total_nails 13 17 8 = 229 := by
  sorry

end NUMINAMATH_CALUDE_johns_house_nails_l4021_402112


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_attained_l4021_402122

theorem min_value_of_function (θ : ℝ) (h : 1 - Real.cos θ ≠ 0) :
  (2 - Real.sin θ) / (1 - Real.cos θ) ≥ 3/4 :=
sorry

theorem min_value_attained (θ : ℝ) (h : 1 - Real.cos θ ≠ 0) :
  ∃ θ₀, (2 - Real.sin θ₀) / (1 - Real.cos θ₀) = 3/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_attained_l4021_402122


namespace NUMINAMATH_CALUDE_equal_distribution_l4021_402147

theorem equal_distribution (total_amount : ℕ) (num_persons : ℕ) (amount_per_person : ℕ) :
  total_amount = 42900 →
  num_persons = 22 →
  amount_per_person = total_amount / num_persons →
  amount_per_person = 1950 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_l4021_402147


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4021_402193

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- Theorem statement
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4021_402193


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l4021_402116

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the modified quadratic function
def g (a b c : ℝ) (x : ℝ) := a * (x^2 + 1) + b * (x - 1) + c - 2 * a * x

theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) :
  (∀ x : ℝ, f a b c x > 0 ↔ -1 < x ∧ x < 2) →
  (∀ x : ℝ, g a b c x > 0 ↔ 0 < x ∧ x < 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l4021_402116
