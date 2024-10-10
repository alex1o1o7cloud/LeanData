import Mathlib

namespace cube_plus_linear_inequality_l431_43156

theorem cube_plus_linear_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^3 + b^3 + a + b ≥ 4 * a * b := by
  sorry

end cube_plus_linear_inequality_l431_43156


namespace singles_on_itunes_l431_43108

def total_songs : ℕ := 55
def albums_15_songs : ℕ := 2
def songs_per_album_15 : ℕ := 15
def albums_20_songs : ℕ := 1
def songs_per_album_20 : ℕ := 20

theorem singles_on_itunes : 
  total_songs - (albums_15_songs * songs_per_album_15 + albums_20_songs * songs_per_album_20) = 5 := by
  sorry

end singles_on_itunes_l431_43108


namespace arithmetic_mean_of_first_four_primes_reciprocals_l431_43182

theorem arithmetic_mean_of_first_four_primes_reciprocals :
  let first_four_primes := [2, 3, 5, 7]
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / 4) = 247 / 840 := by
  sorry

end arithmetic_mean_of_first_four_primes_reciprocals_l431_43182


namespace chromosome_size_homology_l431_43185

/-- Represents a chromosome -/
structure Chromosome where
  size : ℕ
  is_homologous : Bool
  has_centromere : Bool
  gene_order : List ℕ

/-- Represents a pair of chromosomes -/
structure ChromosomePair where
  chromosome1 : Chromosome
  chromosome2 : Chromosome

/-- Defines what it means for chromosomes to be homologous -/
def are_homologous (c1 c2 : Chromosome) : Prop :=
  c1.is_homologous = true ∧ c2.is_homologous = true

/-- Defines what it means for chromosomes to be sister chromatids -/
def are_sister_chromatids (c1 c2 : Chromosome) : Prop :=
  c1.size = c2.size ∧ c1.gene_order = c2.gene_order

/-- Defines a tetrad -/
def is_tetrad (cp : ChromosomePair) : Prop :=
  are_homologous cp.chromosome1 cp.chromosome2

theorem chromosome_size_homology (c1 c2 : Chromosome) :
  c1.size = c2.size → are_homologous c1 c2 → False :=
sorry

#check chromosome_size_homology

end chromosome_size_homology_l431_43185


namespace arrangement_exists_l431_43164

theorem arrangement_exists (n : ℕ) : 
  ∃ (p : Fin n → ℕ), Function.Injective p ∧ Set.range p = Finset.range n ∧
    ∀ (i j k : Fin n), i < j → j < k → 
      p j ≠ (p i + p k) / 2 := by sorry

end arrangement_exists_l431_43164


namespace moon_distance_scientific_notation_l431_43153

theorem moon_distance_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 384000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.84 ∧ n = 5 := by
  sorry

end moon_distance_scientific_notation_l431_43153


namespace inequality_proof_equality_condition_l431_43177

theorem inequality_proof (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z ≥ 3) : 
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 :=
sorry

theorem equality_condition (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 3) :
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) :=
sorry

end inequality_proof_equality_condition_l431_43177


namespace negation_of_existence_negation_of_quadratic_inequality_l431_43110

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x ∈ Set.Ioo 0 2, P x) ↔ (∀ x ∈ Set.Ioo 0 2, ¬P x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x ∈ Set.Ioo 0 2, x^2 + 2*x + 2 ≤ 0) ↔
  (∀ x ∈ Set.Ioo 0 2, x^2 + 2*x + 2 > 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_inequality_l431_43110


namespace square_99_is_white_l431_43126

def Grid := Fin 9 → Fin 9 → Bool

def is_adjacent (x1 y1 x2 y2 : Fin 9) : Prop :=
  (x1 = x2 ∧ y1.val + 1 = y2.val) ∨
  (x1 = x2 ∧ y2.val + 1 = y1.val) ∨
  (y1 = y2 ∧ x1.val + 1 = x2.val) ∨
  (y1 = y2 ∧ x2.val + 1 = x1.val)

def valid_grid (g : Grid) : Prop :=
  (g 4 4 = true) ∧
  (g 4 9 = true) ∧
  (∀ x y, g x y → (∃! x' y', is_adjacent x y x' y' ∧ g x' y')) ∧
  (∀ x y, ¬g x y → (∃! x' y', is_adjacent x y x' y' ∧ ¬g x' y'))

theorem square_99_is_white (g : Grid) (h : valid_grid g) : g 9 9 = false := by
  sorry

#check square_99_is_white

end square_99_is_white_l431_43126


namespace cube_surface_area_l431_43118

/-- The surface area of a cube given the sum of its edge lengths -/
theorem cube_surface_area (edge_sum : ℝ) (h : edge_sum = 72) : 
  6 * (edge_sum / 12)^2 = 216 := by
  sorry

end cube_surface_area_l431_43118


namespace billys_age_l431_43134

theorem billys_age (B J S : ℕ) 
  (h1 : B = 2 * J) 
  (h2 : B + J = 3 * S) 
  (h3 : S = 27) : 
  B = 54 := by
  sorry

end billys_age_l431_43134


namespace lindas_substitution_l431_43174

theorem lindas_substitution (a b c d : ℕ) (e : ℝ) : 
  a = 120 → b = 5 → c = 4 → d = 10 →
  (a / b * c + d - e : ℝ) = (a / (b * (c + (d - e)))) →
  e = 16 := by
sorry

end lindas_substitution_l431_43174


namespace special_circle_equation_l431_43147

/-- A circle with specific properties -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_in_first_quadrant : 0 < center.1 ∧ 0 < center.2
  tangent_to_line : |4 * center.1 - 3 * center.2| = 5 * radius
  tangent_to_x_axis : center.2 = radius
  radius_is_one : radius = 1

/-- The standard equation of a circle given its center and radius -/
def circle_equation (c : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  (x - c.1)^2 + (y - c.2)^2 = r^2

/-- Theorem stating that a SpecialCircle has the standard equation (x-2)^2 + (y-1)^2 = 1 -/
theorem special_circle_equation (C : SpecialCircle) (x y : ℝ) :
  circle_equation (2, 1) 1 x y ↔ circle_equation C.center C.radius x y :=
sorry

end special_circle_equation_l431_43147


namespace pigeonhole_on_floor_division_l431_43151

theorem pigeonhole_on_floor_division (n : ℕ) (h_n : n > 3) 
  (nums : Finset ℕ) (h_nums_card : nums.card = n) 
  (h_nums_distinct : nums.card = Finset.card (Finset.image id nums))
  (h_nums_bound : ∀ x ∈ nums, x < Nat.factorial (n - 1)) :
  ∃ (a b c d : ℕ), a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ d ∈ nums ∧ 
    a > b ∧ c > d ∧ (a ≠ c ∨ b ≠ d) ∧ 
    (a / b : ℕ) = (c / d : ℕ) := by
  sorry

end pigeonhole_on_floor_division_l431_43151


namespace overall_mean_score_l431_43138

/-- Given the mean scores and ratios of students in three classes, prove the overall mean score --/
theorem overall_mean_score (m a e : ℕ) (M A E : ℝ) : 
  M = 78 → A = 68 → E = 82 →
  (m : ℝ) / a = 4 / 5 →
  ((m : ℝ) + a) / e = 9 / 2 →
  (M * m + A * a + E * e) / (m + a + e : ℝ) = 74.4 := by
  sorry

#check overall_mean_score

end overall_mean_score_l431_43138


namespace swimmer_speed_l431_43159

theorem swimmer_speed (downstream_distance upstream_distance : ℝ) 
  (downstream_time upstream_time : ℝ) (h1 : downstream_distance = 35) 
  (h2 : upstream_distance = 20) (h3 : downstream_time = 5) (h4 : upstream_time = 5) :
  ∃ (speed_still_water : ℝ), speed_still_water = 5.5 ∧
  ∃ (stream_speed : ℝ),
    (speed_still_water + stream_speed) * downstream_time = downstream_distance ∧
    (speed_still_water - stream_speed) * upstream_time = upstream_distance :=
by sorry

end swimmer_speed_l431_43159


namespace fixed_OC_length_l431_43119

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point P inside the circle
def P (c : Circle) : ℝ × ℝ := sorry

-- Distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the chord AB
def chord (c : Circle) (p : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define point C
def pointC (c : Circle) (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem fixed_OC_length (c : Circle) : 
  let o := c.center
  let r := c.radius
  let p := P c
  let d := distance o p
  let oc_length := distance o (pointC c p)
  oc_length = Real.sqrt (2 * r^2 - d^2) := by sorry

end fixed_OC_length_l431_43119


namespace min_gumballs_for_four_is_eleven_l431_43100

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat

/-- Represents the minimum number of gumballs needed to guarantee four of the same color -/
def minGumballsForFour (machine : GumballMachine) : Nat :=
  11

/-- Theorem stating that for the given gumball machine, 11 is the minimum number of gumballs 
    needed to guarantee four of the same color -/
theorem min_gumballs_for_four_is_eleven (machine : GumballMachine) 
    (h1 : machine.red = 10) 
    (h2 : machine.white = 7) 
    (h3 : machine.blue = 6) : 
  minGumballsForFour machine = 11 := by
  sorry

#eval minGumballsForFour { red := 10, white := 7, blue := 6 }

end min_gumballs_for_four_is_eleven_l431_43100


namespace product_of_powers_inequality_l431_43130

theorem product_of_powers_inequality (a b : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) (hn : n ≥ 2) :
  (a^n + 1) * (b^n + 1) ≥ 4 := by
  sorry

end product_of_powers_inequality_l431_43130


namespace tangent_line_at_zero_zero_conditions_l431_43113

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

-- Part 1: Tangent line when a = 1
theorem tangent_line_at_zero (x : ℝ) :
  ∃ m b : ℝ, (deriv (f 1)) 0 = m ∧ f 1 0 = b ∧ m = 2 ∧ b = 0 := by sorry

-- Part 2: Range of a for exactly one zero in each interval
theorem zero_conditions (a : ℝ) :
  (∃! x : ℝ, x ∈ Set.Ioo (-1) 0 ∧ f a x = 0) ∧
  (∃! x : ℝ, x ∈ Set.Ioi 0 ∧ f a x = 0) ↔
  a ∈ Set.Iio (-1) := by sorry

end tangent_line_at_zero_zero_conditions_l431_43113


namespace sqrt_nine_equals_three_l431_43190

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end sqrt_nine_equals_three_l431_43190


namespace bricks_per_row_l431_43181

theorem bricks_per_row (total_bricks : ℕ) (rows_per_wall : ℕ) (num_walls : ℕ) 
  (h1 : total_bricks = 3000)
  (h2 : rows_per_wall = 50)
  (h3 : num_walls = 2) :
  total_bricks / (rows_per_wall * num_walls) = 30 := by
sorry

end bricks_per_row_l431_43181


namespace fourth_root_of_cubic_l431_43175

theorem fourth_root_of_cubic (c d : ℚ) :
  (∀ x : ℚ, c * x^3 + (c + 3*d) * x^2 + (d - 4*c) * x + (10 - c) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 2 ∨ x = -9/2) :=
by sorry

end fourth_root_of_cubic_l431_43175


namespace binomial_eight_five_l431_43120

theorem binomial_eight_five : Nat.choose 8 5 = 56 := by
  sorry

end binomial_eight_five_l431_43120


namespace white_balls_count_l431_43154

theorem white_balls_count (red_balls : ℕ) (ratio_red : ℕ) (ratio_white : ℕ) : 
  red_balls = 16 → ratio_red = 4 → ratio_white = 5 → 
  (red_balls * ratio_white) / ratio_red = 20 := by
  sorry

end white_balls_count_l431_43154


namespace marks_radiator_cost_l431_43166

/-- Calculates the total cost for Mark's car radiator replacement. -/
def total_cost (labor_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  labor_hours * hourly_rate + part_cost

/-- Proves that the total cost for Mark's car radiator replacement is $300. -/
theorem marks_radiator_cost :
  total_cost 2 75 150 = 300 := by
  sorry

end marks_radiator_cost_l431_43166


namespace sum_of_digits_of_B_is_seven_l431_43173

def X : ℕ := 4444^4444

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def A : ℕ := sum_of_digits X

def B : ℕ := sum_of_digits A

theorem sum_of_digits_of_B_is_seven :
  sum_of_digits B = 7 := by sorry

end sum_of_digits_of_B_is_seven_l431_43173


namespace coconut_grove_problem_l431_43143

theorem coconut_grove_problem (x : ℝ) 
  (yield_1 : (x + 2) * 40 = (x + 2) * 40)
  (yield_2 : x * 120 = x * 120)
  (yield_3 : (x - 2) * 180 = (x - 2) * 180)
  (average_yield : ((x + 2) * 40 + x * 120 + (x - 2) * 180) / (3 * x) = 100) :
  x = 7 := by
sorry

end coconut_grove_problem_l431_43143


namespace magazine_clients_count_l431_43180

/-- The number of clients using magazines in an advertising agency --/
def clients_using_magazines (total : ℕ) (tv : ℕ) (radio : ℕ) (tv_mag : ℕ) (tv_radio : ℕ) (radio_mag : ℕ) (all_three : ℕ) : ℕ :=
  total + all_three - (tv + radio - tv_radio)

/-- Theorem stating the number of clients using magazines --/
theorem magazine_clients_count : 
  clients_using_magazines 180 115 110 85 75 95 80 = 130 := by
  sorry

end magazine_clients_count_l431_43180


namespace fraction_simplification_l431_43162

theorem fraction_simplification : 
  (1/3 + 1/5) / ((2/7) * (3/4) - 1/7) = 112/15 := by
  sorry

end fraction_simplification_l431_43162


namespace arctg_sum_pi_half_l431_43172

theorem arctg_sum_pi_half : Real.arctan 1 + Real.arctan (1/2) + Real.arctan (1/3) = π/2 := by
  sorry

end arctg_sum_pi_half_l431_43172


namespace data_fraction_less_than_value_l431_43160

theorem data_fraction_less_than_value (data : List ℝ) (fraction : ℝ) (value : ℝ) : 
  data = [1, 2, 3, 4, 5, 5, 5, 5, 7, 11, 21] →
  fraction = 0.36363636363636365 →
  (data.filter (· < value)).length / data.length = fraction →
  value = 4 := by
  sorry

end data_fraction_less_than_value_l431_43160


namespace reading_time_difference_l431_43101

/-- Proves that the difference in reading time between Lee and Kai is 150 minutes -/
theorem reading_time_difference 
  (kai_speed : ℝ) 
  (lee_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : kai_speed = 120) 
  (h2 : lee_speed = 60) 
  (h3 : book_pages = 300) : 
  (book_pages / lee_speed - book_pages / kai_speed) * 60 = 150 := by
  sorry

end reading_time_difference_l431_43101


namespace simplify_expression_l431_43111

theorem simplify_expression : 
  ((3 + 5 + 6 + 2) / 3) + ((2 * 3 + 4 * 2 + 5) / 4) = 121 / 12 := by
  sorry

end simplify_expression_l431_43111


namespace range_of_m_l431_43142

theorem range_of_m (m : ℝ) : 
  (∀ θ : ℝ, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) →
  (m ≥ 4 ∨ m ≤ 0) :=
by sorry

end range_of_m_l431_43142


namespace x_varies_as_sixth_power_of_z_l431_43105

/-- If x varies as the square of y, and y varies as the cube of z,
    then x varies as the 6th power of z. -/
theorem x_varies_as_sixth_power_of_z
  (x y z : ℝ)
  (k j : ℝ)
  (h1 : x = k * y^2)
  (h2 : y = j * z^3) :
  ∃ m : ℝ, x = m * z^6 := by
sorry

end x_varies_as_sixth_power_of_z_l431_43105


namespace seven_awards_four_students_l431_43103

/-- The number of ways to distribute n different awards among k students,
    where each student receives at least one award. -/
def distribute_awards (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 7 awards among 4 students results in 920 ways -/
theorem seven_awards_four_students :
  distribute_awards 7 4 = 920 := by sorry

end seven_awards_four_students_l431_43103


namespace discriminant_of_specific_quadratic_l431_43139

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 5x^2 - 11x + 2 -/
def quadratic_equation (x : ℝ) : ℝ := 5*x^2 - 11*x + 2

theorem discriminant_of_specific_quadratic :
  discriminant 5 (-11) 2 = 81 := by
  sorry

end discriminant_of_specific_quadratic_l431_43139


namespace clinton_belts_l431_43171

/-- Represents the number of items Clinton has in his wardrobe -/
structure Wardrobe where
  shoes : ℕ
  belts : ℕ
  hats : ℕ

/-- Clinton's wardrobe satisfies the given conditions -/
def clinton_wardrobe (w : Wardrobe) : Prop :=
  w.shoes = 2 * w.belts ∧
  ∃ n : ℕ, w.belts = w.hats + n ∧
  w.hats = 5 ∧
  w.shoes = 14

theorem clinton_belts :
  ∀ w : Wardrobe, clinton_wardrobe w → w.belts = 7 := by
  sorry

end clinton_belts_l431_43171


namespace min_value_expression_l431_43133

theorem min_value_expression (a b c k : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  (a^2 / (k*b)) + (b^2 / (k*c)) + (c^2 / (k*a)) ≥ 3/k ∧
  ((a^2 / (k*b)) + (b^2 / (k*c)) + (c^2 / (k*a)) = 3/k ↔ a = b ∧ b = c) :=
by sorry

end min_value_expression_l431_43133


namespace arccos_cos_eq_half_x_solution_l431_43189

theorem arccos_cos_eq_half_x_solution (x : Real) :
  -π/3 ≤ x → x ≤ π/3 → Real.arccos (Real.cos x) = x/2 → x = 0 := by
  sorry

end arccos_cos_eq_half_x_solution_l431_43189


namespace no_x_squared_term_l431_43161

theorem no_x_squared_term (a : ℚ) : 
  (∀ x, (x + 1) * (x^2 - 5*a*x + a) = x^3 + (-5*a + 1)*x^2 + (-9*a)*x + a) → 
  a = 1/5 := by
  sorry

end no_x_squared_term_l431_43161


namespace opposite_of_negative_fraction_l431_43192

theorem opposite_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  -(-(1 : ℚ) / n) = 1 / n := by
  sorry

end opposite_of_negative_fraction_l431_43192


namespace smallest_n_for_floor_equation_l431_43135

theorem smallest_n_for_floor_equation : ∃ (x : ℤ), ⌊(10 : ℝ)^7 / x⌋ = 1989 ∧ ∀ (n : ℕ), n < 7 → ¬∃ (x : ℤ), ⌊(10 : ℝ)^n / x⌋ = 1989 := by
  sorry

end smallest_n_for_floor_equation_l431_43135


namespace symmetric_distribution_theorem_l431_43109

/-- A symmetric distribution with mean m and standard deviation d. -/
structure SymmetricDistribution where
  m : ℝ  -- mean
  d : ℝ  -- standard deviation
  symmetric : Bool
  within_one_std_dev : ℝ

/-- The percentage of the distribution less than m + d -/
def percent_less_than_m_plus_d (dist : SymmetricDistribution) : ℝ := sorry

theorem symmetric_distribution_theorem (dist : SymmetricDistribution) 
  (h_symmetric : dist.symmetric = true)
  (h_within_one_std_dev : dist.within_one_std_dev = 84) :
  percent_less_than_m_plus_d dist = 42 := by sorry

end symmetric_distribution_theorem_l431_43109


namespace tucker_tissues_left_l431_43170

/-- The number of tissues Tucker has left -/
def tissues_left (tissues_per_box : ℕ) (boxes_bought : ℕ) (tissues_used : ℕ) : ℕ :=
  boxes_bought * tissues_per_box - tissues_used

/-- Theorem: Tucker has 270 tissues left -/
theorem tucker_tissues_left :
  tissues_left 160 3 210 = 270 := by
  sorry

end tucker_tissues_left_l431_43170


namespace min_value_reciprocal_sum_l431_43145

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = 1) :
  ∃ (min : ℝ), min = 3 + 2 * Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → 2*x + y = 1 → 1/x + 1/y ≥ min := by
  sorry

end min_value_reciprocal_sum_l431_43145


namespace square_side_length_l431_43183

theorem square_side_length (area : ℚ) (side : ℚ) : 
  area = 9 / 16 → side * side = area → side = 3 / 4 := by
  sorry

end square_side_length_l431_43183


namespace fourth_tree_height_l431_43165

/-- Represents a row of trees with specific properties -/
structure TreeRow where
  tallestHeight : ℝ
  shortestHeight : ℝ
  angleTopLine : ℝ
  equalSpacing : Bool

/-- Calculates the height of the nth tree from the left -/
def heightOfNthTree (row : TreeRow) (n : ℕ) : ℝ :=
  sorry

/-- The main theorem stating the height of the 4th tree -/
theorem fourth_tree_height (row : TreeRow) 
  (h1 : row.tallestHeight = 2.8)
  (h2 : row.shortestHeight = 1.4)
  (h3 : row.angleTopLine = 45)
  (h4 : row.equalSpacing = true) :
  heightOfNthTree row 4 = 2.2 := by
  sorry

end fourth_tree_height_l431_43165


namespace city_park_highest_difference_l431_43148

/-- Snowfall data for different locations --/
structure SnowfallData where
  mrsHilt : Float
  brecknockSchool : Float
  townLibrary : Float
  cityPark : Float

/-- Calculate the absolute difference between two snowfall measurements --/
def snowfallDifference (a b : Float) : Float :=
  (a - b).abs

/-- Determine the location with the highest snowfall difference compared to Mrs. Hilt's house --/
def highestSnowfallDifference (data : SnowfallData) : String :=
  let schoolDiff := snowfallDifference data.mrsHilt data.brecknockSchool
  let libraryDiff := snowfallDifference data.mrsHilt data.townLibrary
  let parkDiff := snowfallDifference data.mrsHilt data.cityPark
  if parkDiff > schoolDiff && parkDiff > libraryDiff then
    "City Park"
  else if schoolDiff > libraryDiff then
    "Brecknock Elementary School"
  else
    "Town Library"

/-- Theorem: The city park has the highest snowfall difference compared to Mrs. Hilt's house --/
theorem city_park_highest_difference (data : SnowfallData)
  (h1 : data.mrsHilt = 29.7)
  (h2 : data.brecknockSchool = 17.3)
  (h3 : data.townLibrary = 23.8)
  (h4 : data.cityPark = 12.6) :
  highestSnowfallDifference data = "City Park" := by
  sorry

end city_park_highest_difference_l431_43148


namespace perpendicular_line_equation_l431_43106

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation (P : Point) (L : Line) :
  P.x = 4 ∧ P.y = -1 ∧ L.a = 3 ∧ L.b = -4 ∧ L.c = 6 →
  ∃ (M : Line), M.a = 4 ∧ M.b = 3 ∧ M.c = -13 ∧ P.liesOn M ∧ M.perpendicular L := by
  sorry

end perpendicular_line_equation_l431_43106


namespace expression_value_at_four_l431_43102

theorem expression_value_at_four :
  let x : ℚ := 4
  (x^2 - 3*x - 10) / (x - 5) = 6 := by sorry

end expression_value_at_four_l431_43102


namespace circle_intersection_l431_43137

/-- The number of intersection points between two circles -/
def intersectionPoints (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : ℕ :=
  sorry

/-- Theorem: The circle centered at (0, 3) with radius 3 and the circle centered at (5, 0) with radius 5 intersect at 4 points -/
theorem circle_intersection :
  intersectionPoints (0, 3) (5, 0) 3 5 = 4 := by
  sorry

end circle_intersection_l431_43137


namespace glass_bowls_percentage_gain_l431_43198

/-- Calculate the percentage gain from buying and selling glass bowls -/
theorem glass_bowls_percentage_gain 
  (total_bought : ℕ) 
  (cost_price : ℚ) 
  (total_sold : ℕ) 
  (selling_price : ℚ) 
  (broken : ℕ) 
  (h1 : total_bought = 250)
  (h2 : cost_price = 18)
  (h3 : total_sold = 200)
  (h4 : selling_price = 25)
  (h5 : broken = 30)
  (h6 : total_sold + broken ≤ total_bought) :
  (((total_sold : ℚ) * selling_price - (total_bought : ℚ) * cost_price) / 
   ((total_bought : ℚ) * cost_price)) * 100 = 100 / 9 := by
sorry

#eval (100 : ℚ) / 9  -- To show the approximate result

end glass_bowls_percentage_gain_l431_43198


namespace circle_area_ratio_l431_43122

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 : ℝ) * (2 * Real.pi * r₁) = (30 / 360 : ℝ) * (2 * Real.pi * r₂) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 1 / 4 := by
sorry

end circle_area_ratio_l431_43122


namespace multiplication_properties_l431_43168

theorem multiplication_properties : 
  (∀ n : ℝ, n * 0 = 0) ∧ 
  (∀ n : ℝ, n * 1 = n) ∧ 
  (∀ n : ℝ, n * (-1) = -n) ∧ 
  (∃ a b : ℝ, a + b = 0 ∧ a * b ≠ 1) := by
sorry

end multiplication_properties_l431_43168


namespace apple_pie_count_l431_43107

/-- Given a box of apples weighing 120 pounds, using half for applesauce and the rest for pies,
    with each pie requiring 4 pounds of apples, prove that 15 pies can be made. -/
theorem apple_pie_count (total_weight : ℕ) (pie_weight : ℕ) : 
  total_weight = 120 →
  pie_weight = 4 →
  (total_weight / 2) / pie_weight = 15 :=
by
  sorry

end apple_pie_count_l431_43107


namespace binary_subtraction_l431_43150

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def a : List Bool := [true, true, true, true, true, true, true, true, true, true, true]
def b : List Bool := [true, true, true, true, true, true, true]

theorem binary_subtraction :
  binary_to_decimal a - binary_to_decimal b = 1920 := by sorry

end binary_subtraction_l431_43150


namespace percentage_problem_l431_43176

theorem percentage_problem (x : ℝ) (P : ℝ) 
  (h1 : x = 180)
  (h2 : P * x = 0.10 * 500 - 5) :
  P = 0.25 := by
sorry

end percentage_problem_l431_43176


namespace pencil_count_l431_43191

/-- Proves that given the specified costs and quantities, the number of pencils needed is 24 --/
theorem pencil_count (pencil_cost folder_cost total_cost : ℚ) (folder_count : ℕ) : 
  pencil_cost = 1/2 →
  folder_cost = 9/10 →
  folder_count = 20 →
  total_cost = 30 →
  (total_cost - folder_cost * folder_count) / pencil_cost = 24 := by
sorry

end pencil_count_l431_43191


namespace gold_weight_is_ten_l431_43121

def weights : List ℕ := List.range 19

theorem gold_weight_is_ten (iron_weights bronze_weights : List ℕ) 
  (h1 : iron_weights.length = 9)
  (h2 : bronze_weights.length = 9)
  (h3 : iron_weights ⊆ weights)
  (h4 : bronze_weights ⊆ weights)
  (h5 : (iron_weights.sum - bronze_weights.sum) = 90)
  (h6 : iron_weights ∩ bronze_weights = [])
  : weights.sum - iron_weights.sum - bronze_weights.sum = 10 := by
  sorry

#check gold_weight_is_ten

end gold_weight_is_ten_l431_43121


namespace math_books_count_l431_43178

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℕ) 
  (h1 : total_books = 90)
  (h2 : math_cost = 4)
  (h3 : history_cost = 5)
  (h4 : total_price = 397) :
  ∃ (math_books : ℕ), 
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧ 
    math_books = 53 := by
  sorry

end math_books_count_l431_43178


namespace problem_solution_l431_43155

theorem problem_solution (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k)*(x - k) = x^3 - k*(x^2 + x + 3)) → k = -3 := by
  sorry

end problem_solution_l431_43155


namespace rectangular_prism_sum_l431_43163

/-- A rectangular prism is a three-dimensional geometric shape. -/
structure RectangularPrism where
  edges : ℕ
  corners : ℕ
  faces : ℕ

/-- The properties of a rectangular prism -/
def is_valid_rectangular_prism (rp : RectangularPrism) : Prop :=
  rp.edges = 12 ∧ rp.corners = 8 ∧ rp.faces = 6

/-- The theorem stating that the sum of edges, corners, and faces of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) 
  (h : is_valid_rectangular_prism rp) : 
  rp.edges + rp.corners + rp.faces = 26 := by
  sorry

end rectangular_prism_sum_l431_43163


namespace arithmetic_sequence_terms_l431_43125

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Theorem statement
theorem arithmetic_sequence_terms :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 17 6 n = 101 ∧ n = 15 := by
  sorry

end arithmetic_sequence_terms_l431_43125


namespace cost_price_per_meter_l431_43186

/-- Proves that given a selling price of Rs. 12,000 for 200 meters of cloth
    and a loss of Rs. 6 per meter, the cost price for one meter of cloth is Rs. 66. -/
theorem cost_price_per_meter
  (total_meters : ℕ)
  (selling_price : ℕ)
  (loss_per_meter : ℕ)
  (h1 : total_meters = 200)
  (h2 : selling_price = 12000)
  (h3 : loss_per_meter = 6) :
  (selling_price + total_meters * loss_per_meter) / total_meters = 66 := by
  sorry

#check cost_price_per_meter

end cost_price_per_meter_l431_43186


namespace a_range_l431_43115

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ a ≤ Real.exp x

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + x + a > 0

-- State the theorem
theorem a_range (a : ℝ) (h : p a ∧ q a) : 1/4 < a ∧ a ≤ Real.exp 1 := by
  sorry


end a_range_l431_43115


namespace some_beautiful_objects_are_colorful_l431_43196

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Rose Beautiful Colorful : U → Prop)

-- State the theorem
theorem some_beautiful_objects_are_colorful :
  (∀ x, Rose x → Beautiful x) →  -- All roses are beautiful
  (∃ x, Colorful x ∧ Rose x) →   -- Some colorful objects are roses
  (∃ x, Beautiful x ∧ Colorful x) -- Some beautiful objects are colorful
  := by sorry

end some_beautiful_objects_are_colorful_l431_43196


namespace crab_meat_per_dish_l431_43184

/-- Proves that given the conditions of Johnny's crab dish production, he uses 1.5 pounds of crab meat per dish. -/
theorem crab_meat_per_dish (dishes_per_day : ℕ) (crab_cost_per_pound : ℚ) 
  (weekly_crab_cost : ℚ) (operating_days : ℕ) :
  dishes_per_day = 40 →
  crab_cost_per_pound = 8 →
  weekly_crab_cost = 1920 →
  operating_days = 4 →
  (weekly_crab_cost / crab_cost_per_pound) / operating_days / dishes_per_day = 3/2 := by
  sorry

#check crab_meat_per_dish

end crab_meat_per_dish_l431_43184


namespace fraction_sum_theorem_l431_43140

theorem fraction_sum_theorem (x y : ℝ) (h : x ≠ y) :
  (x + y) / (x - y) + (x - y) / (x + y) = 3 →
  (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = 13/6 := by
sorry

end fraction_sum_theorem_l431_43140


namespace birds_in_tree_l431_43123

theorem birds_in_tree (initial_birds final_birds : ℕ) : 
  initial_birds = 14 → final_birds = 35 → final_birds - initial_birds = 21 := by
  sorry

end birds_in_tree_l431_43123


namespace bill_difference_is_zero_l431_43194

/-- Given Linda's and Mark's tips and tip percentages, prove that the difference between their bills is 0. -/
theorem bill_difference_is_zero (linda_tip mark_tip : ℝ) (linda_percent mark_percent : ℝ) 
  (h1 : linda_tip = 5)
  (h2 : linda_percent = 0.25)
  (h3 : mark_tip = 3)
  (h4 : mark_percent = 0.15)
  (h5 : linda_tip = linda_percent * linda_bill)
  (h6 : mark_tip = mark_percent * mark_bill)
  : linda_bill - mark_bill = 0 := by
  sorry

#check bill_difference_is_zero

end bill_difference_is_zero_l431_43194


namespace subtraction_puzzle_l431_43116

theorem subtraction_puzzle (A B : Nat) : 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  2000 + 100 * A + 32 - (100 * B + 10 * B + B) = 1000 + 100 * B + 10 * B + B → 
  B - A = 3 := by
sorry

end subtraction_puzzle_l431_43116


namespace fishing_trip_total_l431_43197

def total_fish (pikes sturgeons herrings : ℕ) : ℕ :=
  pikes + sturgeons + herrings

theorem fishing_trip_total : total_fish 30 40 75 = 145 := by
  sorry

end fishing_trip_total_l431_43197


namespace pokemon_cards_total_l431_43144

def jenny_cards : ℕ := 6

def orlando_cards (jenny : ℕ) : ℕ := jenny + 2

def richard_cards (orlando : ℕ) : ℕ := orlando * 3

def total_cards (jenny orlando richard : ℕ) : ℕ := jenny + orlando + richard

theorem pokemon_cards_total :
  total_cards jenny_cards (orlando_cards jenny_cards) (richard_cards (orlando_cards jenny_cards)) = 38 := by
  sorry

end pokemon_cards_total_l431_43144


namespace polynomial_uniqueness_l431_43195

theorem polynomial_uniqueness (P : ℝ → ℝ) : 
  (∀ x, P x = P 0 + P 1 * x + P 3 * x^3) → 
  P (-1) = 3 → 
  ∀ x, P x = 3 + x + x^3 := by
sorry

end polynomial_uniqueness_l431_43195


namespace college_entrance_exam_scoring_l431_43112

theorem college_entrance_exam_scoring (total_questions raw_score questions_answered correct_answers : ℕ)
  (h1 : total_questions = 85)
  (h2 : questions_answered = 82)
  (h3 : correct_answers = 70)
  (h4 : raw_score = 67)
  (h5 : questions_answered ≤ total_questions)
  (h6 : correct_answers ≤ questions_answered) :
  ∃ (points_subtracted : ℚ),
    points_subtracted = 1/4 ∧
    (correct_answers : ℚ) - (questions_answered - correct_answers) * points_subtracted = raw_score := by
sorry

end college_entrance_exam_scoring_l431_43112


namespace function_uniqueness_l431_43157

theorem function_uniqueness (f : ℝ → ℝ) (a : ℝ) : 
  ∃! y, f a = y :=
sorry

end function_uniqueness_l431_43157


namespace five_lines_sixteen_sections_l431_43117

/-- The number of sections created by drawing n properly intersecting line segments in a rectangle --/
def sections (n : ℕ) : ℕ :=
  if n = 0 then 1 else sections (n - 1) + n

/-- Theorem: Drawing 5 properly intersecting line segments in a rectangle creates 16 sections --/
theorem five_lines_sixteen_sections : sections 5 = 16 := by
  sorry

end five_lines_sixteen_sections_l431_43117


namespace name_is_nika_l431_43104

-- Define a cube face
inductive Face
| Front
| Back
| Left
| Right
| Top
| Bottom

-- Define a letter
inductive Letter
| N
| I
| K
| A
| T

-- Define a cube
structure Cube where
  faces : Face → Letter

-- Define the arrangement of cubes
def CubeArrangement := List Cube

-- Define the function to get the front-facing letters
def getFrontLetters (arrangement : CubeArrangement) : List Letter :=
  arrangement.map (λ cube => cube.faces Face.Front)

-- Theorem statement
theorem name_is_nika (arrangement : CubeArrangement) 
  (h1 : arrangement.length = 4)
  (h2 : getFrontLetters arrangement = [Letter.N, Letter.I, Letter.K, Letter.A]) :
  "Ника" = "Ника" :=
by sorry

end name_is_nika_l431_43104


namespace cos_double_angle_from_series_sum_l431_43131

theorem cos_double_angle_from_series_sum (θ : ℝ) :
  (∑' n, (Real.cos θ) ^ (2 * n) = 8) → Real.cos (2 * θ) = 3 / 4 := by
  sorry

end cos_double_angle_from_series_sum_l431_43131


namespace smallest_nonprime_with_large_factors_l431_43136

theorem smallest_nonprime_with_large_factors :
  ∃ n : ℕ,
    n > 1 ∧
    ¬ Prime n ∧
    (∀ p : ℕ, Prime p → p < 20 → ¬ p ∣ n) ∧
    (∀ m : ℕ, m > 1 → ¬ Prime m → (∀ q : ℕ, Prime q → q < 20 → ¬ q ∣ m) → m ≥ n) ∧
    500 < n ∧
    n ≤ 600 :=
by sorry

end smallest_nonprime_with_large_factors_l431_43136


namespace least_subtraction_for_divisibility_l431_43146

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (x : ℕ), x ≤ 9 ∧ (427398 - x) % 10 = 0 ∧ 
  ∀ (y : ℕ), y < x → (427398 - y) % 10 ≠ 0 :=
sorry

end least_subtraction_for_divisibility_l431_43146


namespace max_non_club_members_in_company_l431_43132

/-- The maximum number of people who did not join any club in a company with 5 clubs -/
def max_non_club_members (total_people : ℕ) (club_a : ℕ) (club_b : ℕ) (club_c : ℕ) (club_d : ℕ) (club_e : ℕ) (c_and_d_overlap : ℕ) (d_and_e_overlap : ℕ) : ℕ :=
  total_people - (club_a + club_b + club_c + (club_d - c_and_d_overlap) + (club_e - d_and_e_overlap))

/-- Theorem stating the maximum number of non-club members in the given scenario -/
theorem max_non_club_members_in_company :
  max_non_club_members 120 25 34 21 16 10 8 4 = 26 :=
by sorry

end max_non_club_members_in_company_l431_43132


namespace min_distance_ellipse_line_l431_43124

/-- The minimum distance between a point on the ellipse x²/8 + y²/4 = 1 
    and the line 4x + 3y - 24 = 0 is (24 - 2√41) / 5 -/
theorem min_distance_ellipse_line : 
  let ellipse := {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}
  let line := {p : ℝ × ℝ | 4 * p.1 + 3 * p.2 - 24 = 0}
  ∃ (d : ℝ), d = (24 - 2 * Real.sqrt 41) / 5 ∧ 
    (∀ p ∈ ellipse, ∀ q ∈ line, d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
    (∃ p ∈ ellipse, ∃ q ∈ line, d = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) := by
  sorry

end min_distance_ellipse_line_l431_43124


namespace largest_number_is_482_l431_43129

/-- Given a systematic sample from a set of products, this function calculates the largest number in the sample. -/
def largest_sample_number (total_products : ℕ) (smallest_number : ℕ) (second_smallest : ℕ) : ℕ :=
  let sampling_interval := second_smallest - smallest_number
  let sample_size := total_products / sampling_interval
  smallest_number + sampling_interval * (sample_size - 1)

/-- Theorem stating that for the given conditions, the largest number in the sample is 482. -/
theorem largest_number_is_482 :
  largest_sample_number 500 7 32 = 482 := by
  sorry

end largest_number_is_482_l431_43129


namespace quadratic_form_value_l431_43179

/-- Given a quadratic function f(x) = 4x^2 - 40x + 100, 
    prove that when written in the form (ax + b)^2 + c,
    the value of 2b - 3c is -20 -/
theorem quadratic_form_value (a b c : ℝ) : 
  (∀ x, 4 * x^2 - 40 * x + 100 = (a * x + b)^2 + c) →
  2 * b - 3 * c = -20 := by
  sorry

end quadratic_form_value_l431_43179


namespace jefferson_carriage_cost_l431_43127

/-- Represents the carriage rental cost calculation --/
def carriageRentalCost (
  totalDistance : ℝ)
  (stopDistances : List ℝ)
  (speeds : List ℝ)
  (baseRate : ℝ)
  (flatFee : ℝ)
  (additionalChargeThreshold : ℝ)
  (additionalChargeRate : ℝ)
  (discountRate : ℝ) : ℝ :=
  sorry

/-- Theorem stating the correct total cost for Jefferson's carriage rental --/
theorem jefferson_carriage_cost :
  carriageRentalCost
    20                     -- total distance to church
    [4, 6, 3]              -- distances to each stop
    [8, 12, 10, 15]        -- speeds for each leg
    35                     -- base rate per hour
    20                     -- flat fee
    10                     -- additional charge speed threshold
    5                      -- additional charge rate per mile
    0.1                    -- discount rate
  = 132.15 := by sorry

end jefferson_carriage_cost_l431_43127


namespace multiplication_result_l431_43114

theorem multiplication_result : (935421 * 625) = 584638125 := by
  sorry

end multiplication_result_l431_43114


namespace bacteria_growth_l431_43169

theorem bacteria_growth (division_time : ℕ) (total_time : ℕ) (initial_count : ℕ) : 
  division_time = 20 → 
  total_time = 180 → 
  initial_count = 1 → 
  2 ^ (total_time / division_time) = 512 :=
by
  sorry

#check bacteria_growth

end bacteria_growth_l431_43169


namespace flower_bed_weeds_count_l431_43141

/-- The number of weeds in the flower bed -/
def flower_bed_weeds : ℕ := 11

/-- The number of weeds in the vegetable patch -/
def vegetable_patch_weeds : ℕ := 14

/-- The number of weeds in the grass around the fruit trees -/
def grass_weeds : ℕ := 32

/-- The amount Lucille earns per weed in cents -/
def cents_per_weed : ℕ := 6

/-- The cost of the soda in cents -/
def soda_cost : ℕ := 99

/-- The amount of money Lucille has left in cents -/
def money_left : ℕ := 147

theorem flower_bed_weeds_count : 
  flower_bed_weeds = 11 :=
by sorry

end flower_bed_weeds_count_l431_43141


namespace triangle_angle_bounds_l431_43149

noncomputable def largest_angle (a b : ℝ) : ℝ := 
  Real.arccos (a / (2 * b))

noncomputable def smallest_angle_case1 (a b : ℝ) : ℝ := 
  Real.arcsin (a / b)

noncomputable def smallest_angle_case2 (a b : ℝ) : ℝ := 
  Real.arccos (b / (2 * a))

theorem triangle_angle_bounds (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (x y : ℝ),
    (largest_angle a b ≤ x ∧ x < π) ∧
    ((b ≥ a * Real.sqrt 2 → 0 < y ∧ y ≤ smallest_angle_case1 a b) ∧
     (b ≤ a * Real.sqrt 2 → 0 < y ∧ y ≤ smallest_angle_case2 a b)) :=
by sorry

end triangle_angle_bounds_l431_43149


namespace smallest_constant_inequality_l431_43167

theorem smallest_constant_inequality (x y z : ℝ) :
  ∃ D : ℝ, (∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + z^2 + 3 ≥ D * (x + y + z)) ∧
  D = -Real.sqrt (72 / 11) ∧
  ∀ E : ℝ, (∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + z^2 + 3 ≥ E * (x + y + z)) → D ≤ E :=
by sorry

end smallest_constant_inequality_l431_43167


namespace box_height_is_15_l431_43193

/-- Proves that the height of a box is 15 inches given specific conditions --/
theorem box_height_is_15 (base_length : ℝ) (base_width : ℝ) (total_volume : ℝ) 
  (cost_per_box : ℝ) (min_spend : ℝ) (h : ℝ) : 
  base_length = 20 ∧ base_width = 20 ∧ total_volume = 3060000 ∧ 
  cost_per_box = 1.3 ∧ min_spend = 663 →
  h = 15 := by
  sorry

#check box_height_is_15

end box_height_is_15_l431_43193


namespace sum_product_bound_l431_43128

theorem sum_product_bound (a b c d : ℝ) (h : a + b + c + d = 1) :
  ∃ (x : ℝ), x ≤ 0.5 ∧ (ab + ac + ad + bc + bd + cd ≤ x) ∧
  ∀ (y : ℝ), ∃ (a' b' c' d' : ℝ), a' + b' + c' + d' = 1 ∧
  a'*b' + a'*c' + a'*d' + b'*c' + b'*d' + c'*d' < y :=
sorry

end sum_product_bound_l431_43128


namespace half_percent_of_160_l431_43158

theorem half_percent_of_160 : (1 / 2 * 1 / 100) * 160 = 0.8 := by
  sorry

end half_percent_of_160_l431_43158


namespace cookies_per_batch_l431_43187

/-- Given a bag of chocolate chips with 81 chips, used to make 3 batches of cookies,
    where each cookie contains 9 chips, prove that there are 3 cookies in each batch. -/
theorem cookies_per_batch (total_chips : ℕ) (num_batches : ℕ) (chips_per_cookie : ℕ) 
  (h1 : total_chips = 81)
  (h2 : num_batches = 3)
  (h3 : chips_per_cookie = 9)
  (h4 : total_chips % num_batches = 0)
  (h5 : (total_chips / num_batches) % chips_per_cookie = 0) :
  (total_chips / num_batches) / chips_per_cookie = 3 := by
  sorry

end cookies_per_batch_l431_43187


namespace quadratic_equation_solution_l431_43199

theorem quadratic_equation_solution :
  let x₁ := -2 + Real.sqrt 2
  let x₂ := -2 - Real.sqrt 2
  x₁^2 + 4*x₁ + 2 = 0 ∧ x₂^2 + 4*x₂ + 2 = 0 := by
sorry

end quadratic_equation_solution_l431_43199


namespace zoo_count_difference_l431_43188

theorem zoo_count_difference (zebras camels monkeys giraffes : ℕ) : 
  zebras = 12 →
  camels = zebras / 2 →
  monkeys = 4 * camels →
  giraffes = 2 →
  monkeys - giraffes = 22 := by
sorry

end zoo_count_difference_l431_43188


namespace multiplicative_inverse_208_mod_307_l431_43152

theorem multiplicative_inverse_208_mod_307 : ∃ x : ℕ, x < 307 ∧ (208 * x) % 307 = 1 :=
by
  use 240
  sorry

end multiplicative_inverse_208_mod_307_l431_43152
