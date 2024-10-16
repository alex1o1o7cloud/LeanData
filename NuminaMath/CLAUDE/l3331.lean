import Mathlib

namespace NUMINAMATH_CALUDE_pet_store_count_l3331_333175

/-- Represents the count of animals in a pet store -/
structure PetStore :=
  (birds : ℕ)
  (puppies : ℕ)
  (cats : ℕ)
  (spiders : ℕ)

/-- Calculates the total number of animals in the pet store -/
def totalAnimals (store : PetStore) : ℕ :=
  store.birds + store.puppies + store.cats + store.spiders

/-- Represents the changes in animal counts -/
structure Changes :=
  (birdsSold : ℕ)
  (puppiesAdopted : ℕ)
  (spidersLoose : ℕ)

/-- Applies changes to the pet store counts -/
def applyChanges (store : PetStore) (changes : Changes) : PetStore :=
  { birds := store.birds - changes.birdsSold,
    puppies := store.puppies - changes.puppiesAdopted,
    cats := store.cats,
    spiders := store.spiders - changes.spidersLoose }

theorem pet_store_count : 
  let initialStore : PetStore := { birds := 12, puppies := 9, cats := 5, spiders := 15 }
  let changes : Changes := { birdsSold := 6, puppiesAdopted := 3, spidersLoose := 7 }
  let finalStore := applyChanges initialStore changes
  totalAnimals finalStore = 25 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_count_l3331_333175


namespace NUMINAMATH_CALUDE_expression_equals_one_l3331_333110

theorem expression_equals_one : 
  (105^2 - 8^2) / (80^2 - 13^2) * ((80 - 13) * (80 + 13)) / ((105 - 8) * (105 + 8)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3331_333110


namespace NUMINAMATH_CALUDE_det_example_and_cube_diff_sum_of_cubes_given_conditions_complex_det_sum_given_conditions_l3331_333121

-- Define the determinant operation
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Statement 1
theorem det_example_and_cube_diff : 
  det 5 4 8 9 = 13 ∧ ∀ a : ℝ, a^3 - 3*a^2 + 3*a - 1 = (a - 1)^3 := by sorry

-- Statement 2
theorem sum_of_cubes_given_conditions :
  ∀ x y : ℝ, x + y = 3 → x * y = 1 → x^3 + y^3 = 18 := by sorry

-- Statement 3
theorem complex_det_sum_given_conditions :
  ∀ x m n : ℝ, m = x - 1 → n = x + 2 → m * n = 5 →
  det m (3*m^2 + n^2) n (m^2 + 3*n^2) + det (m + n) (-2*n) n (m - n) = -8 := by sorry

end NUMINAMATH_CALUDE_det_example_and_cube_diff_sum_of_cubes_given_conditions_complex_det_sum_given_conditions_l3331_333121


namespace NUMINAMATH_CALUDE_candy_mix_proof_l3331_333129

/-- Proves that mixing 1 pound of candy A with 4 pounds of candy B
    produces 5 pounds of mixed candy that costs $2.00 per pound -/
theorem candy_mix_proof (candy_a_cost candy_b_cost mix_cost : ℝ)
                        (candy_a_weight candy_b_weight : ℝ) :
  candy_a_cost = 3.20 →
  candy_b_cost = 1.70 →
  mix_cost = 2.00 →
  candy_a_weight = 1 →
  candy_b_weight = 4 →
  (candy_a_cost * candy_a_weight + candy_b_cost * candy_b_weight) / 
    (candy_a_weight + candy_b_weight) = mix_cost :=
by sorry

end NUMINAMATH_CALUDE_candy_mix_proof_l3331_333129


namespace NUMINAMATH_CALUDE_eqLength_is_53_l3331_333100

/-- Represents a trapezoid with a circle inscribed in it. -/
structure InscribedTrapezoid where
  /-- Length of side EF -/
  ef : ℝ
  /-- Length of side FG -/
  fg : ℝ
  /-- Length of side GH -/
  gh : ℝ
  /-- Length of side HE -/
  he : ℝ
  /-- EF is parallel to GH -/
  parallel : ef > gh

/-- The length of EQ in the inscribed trapezoid. -/
def eqLength (t : InscribedTrapezoid) : ℝ := sorry

/-- Theorem stating that for the given trapezoid, EQ = 53 -/
theorem eqLength_is_53 (t : InscribedTrapezoid) 
  (h1 : t.ef = 84) 
  (h2 : t.fg = 58) 
  (h3 : t.gh = 27) 
  (h4 : t.he = 64) : 
  eqLength t = 53 := by sorry

end NUMINAMATH_CALUDE_eqLength_is_53_l3331_333100


namespace NUMINAMATH_CALUDE_alcohol_percentage_P_correct_l3331_333199

/-- The percentage of alcohol in vessel P that results in the given mixture ratio -/
def alcohol_percentage_P : ℝ := 62.5

/-- The percentage of alcohol in vessel Q -/
def alcohol_percentage_Q : ℝ := 87.5

/-- The volume of liquid taken from each vessel -/
def volume_per_vessel : ℝ := 4

/-- The ratio of alcohol to water in the resulting mixture -/
def mixture_ratio : ℝ := 3

/-- The total volume of the mixture -/
def total_volume : ℝ := 2 * volume_per_vessel

theorem alcohol_percentage_P_correct :
  (alcohol_percentage_P / 100 * volume_per_vessel +
   alcohol_percentage_Q / 100 * volume_per_vessel) / total_volume = mixture_ratio / (mixture_ratio + 1) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_P_correct_l3331_333199


namespace NUMINAMATH_CALUDE_age_ratio_in_five_years_l3331_333157

/-- Represents the ages of Sam and Dan -/
structure Ages where
  sam : ℕ
  dan : ℕ

/-- The conditions given in the problem -/
def age_conditions (a : Ages) : Prop :=
  (a.sam - 3 = 2 * (a.dan - 3)) ∧ 
  (a.sam - 7 = 3 * (a.dan - 7))

/-- The future condition we want to prove -/
def future_ratio (a : Ages) (years : ℕ) : Prop :=
  3 * (a.dan + years) = 2 * (a.sam + years)

/-- The main theorem to prove -/
theorem age_ratio_in_five_years (a : Ages) :
  age_conditions a → future_ratio a 5 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_in_five_years_l3331_333157


namespace NUMINAMATH_CALUDE_line_segment_param_product_l3331_333192

/-- Given a line segment connecting (1, -3) and (6, 9), parameterized by x = at + b and y = ct + d
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that (a+b) × (c+d) = 54. -/
theorem line_segment_param_product (a b c d : ℝ) : 
  (∀ t : ℝ, 0 ≤ t → t ≤ 1 → 
    ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (1 = b ∧ -3 = d) →
  (6 = a + b ∧ 9 = c + d) →
  (a + b) * (c + d) = 54 := by
sorry

end NUMINAMATH_CALUDE_line_segment_param_product_l3331_333192


namespace NUMINAMATH_CALUDE_concyclicity_theorem_l3331_333195

-- Define the points
variable (A B C D A' B' C' D' : Point)

-- Define the concyclicity property
def areConcyclic (P Q R S : Point) : Prop := sorry

-- State the theorem
theorem concyclicity_theorem 
  (h1 : areConcyclic A B C D)
  (h2 : areConcyclic A A' B B')
  (h3 : areConcyclic B B' C C')
  (h4 : areConcyclic C C' D D')
  (h5 : areConcyclic D D' A A') :
  areConcyclic A' B' C' D' := by sorry

end NUMINAMATH_CALUDE_concyclicity_theorem_l3331_333195


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3331_333191

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 1) (h2 : m > 0) (h3 : n > 0) :
  1/m + 1/n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3331_333191


namespace NUMINAMATH_CALUDE_pattern_solution_l3331_333112

theorem pattern_solution (n : ℕ+) (a b : ℕ+) :
  (∀ k : ℕ+, Real.sqrt (k + k / (k^2 - 1)) = k * Real.sqrt (k / (k^2 - 1))) →
  (Real.sqrt (8 + b / a) = 8 * Real.sqrt (b / a)) →
  a = 63 ∧ b = 8 := by sorry

end NUMINAMATH_CALUDE_pattern_solution_l3331_333112


namespace NUMINAMATH_CALUDE_fraction_addition_l3331_333136

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3331_333136


namespace NUMINAMATH_CALUDE_trigonometric_equality_l3331_333176

theorem trigonometric_equality : 
  Real.sqrt (1 - 2 * Real.cos (π / 2 + 3) * Real.sin (π / 2 - 3)) = Real.sin 3 + Real.cos 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l3331_333176


namespace NUMINAMATH_CALUDE_max_sum_with_constraints_l3331_333134

theorem max_sum_with_constraints (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 10) 
  (h2 : 3 * a + 6 * b ≤ 12) : 
  a + b ≤ 22 / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_constraints_l3331_333134


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l3331_333115

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i :
  1 + i + i^2 + i^3 + i^4 + i^5 + i^6 = i :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l3331_333115


namespace NUMINAMATH_CALUDE_base_eight_addition_l3331_333156

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Adds two numbers in base b -/
def addInBase (n1 n2 : List Nat) (b : Nat) : List Nat :=
  sorry

theorem base_eight_addition : ∃ b : Nat, 
  b > 1 ∧ 
  addInBase [4, 5, 2] [3, 1, 6] b = [7, 7, 0] ∧
  b = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_addition_l3331_333156


namespace NUMINAMATH_CALUDE_second_expression_proof_l3331_333168

theorem second_expression_proof (a x : ℝ) : 
  ((2 * a + 16 + x) / 2 = 84) → (a = 32) → (x = 88) := by
  sorry

end NUMINAMATH_CALUDE_second_expression_proof_l3331_333168


namespace NUMINAMATH_CALUDE_complement_B_intersect_A_union_A_M_equiv_M_l3331_333126

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x > 1}

-- Define set M with parameter a
def M (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a + 6}

-- Theorem for part (1)
theorem complement_B_intersect_A :
  (Set.univ \ B) ∩ A = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part (2)
theorem union_A_M_equiv_M (a : ℝ) :
  A ∪ M a = M a ↔ -4 < a ∧ a < -2 := by sorry

end NUMINAMATH_CALUDE_complement_B_intersect_A_union_A_M_equiv_M_l3331_333126


namespace NUMINAMATH_CALUDE_train_length_l3331_333137

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 180 → time = 9 → speed * time * (1000 / 3600) = 450 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3331_333137


namespace NUMINAMATH_CALUDE_garden_area_l3331_333167

-- Define the garden structure
structure Garden where
  side_length : ℝ
  perimeter : ℝ
  area : ℝ

-- Define the conditions
def garden_conditions (g : Garden) : Prop :=
  g.perimeter = 4 * g.side_length ∧
  g.area = g.side_length * g.side_length ∧
  1500 = 30 * g.side_length ∧
  1500 = 15 * g.perimeter

-- Theorem statement
theorem garden_area (g : Garden) (h : garden_conditions g) : g.area = 625 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l3331_333167


namespace NUMINAMATH_CALUDE_total_reptiles_count_l3331_333163

/-- The number of swamps in the sanctuary -/
def num_swamps : ℕ := 4

/-- The number of reptiles in each swamp -/
def reptiles_per_swamp : ℕ := 356

/-- The total number of reptiles in all swamp areas -/
def total_reptiles : ℕ := num_swamps * reptiles_per_swamp

theorem total_reptiles_count : total_reptiles = 1424 := by
  sorry

end NUMINAMATH_CALUDE_total_reptiles_count_l3331_333163


namespace NUMINAMATH_CALUDE_second_street_sales_l3331_333189

/-- Represents the sales data for a door-to-door salesman selling security systems. -/
structure SalesData where
  commission_per_sale : ℕ
  total_commission : ℕ
  first_street_sales : ℕ
  second_street_sales : ℕ
  fourth_street_sales : ℕ

/-- Theorem stating the number of security systems sold on the second street. -/
theorem second_street_sales (data : SalesData) : data.second_street_sales = 4 :=
  by
  have h1 : data.commission_per_sale = 25 := by sorry
  have h2 : data.total_commission = 175 := by sorry
  have h3 : data.first_street_sales = data.second_street_sales / 2 := by sorry
  have h4 : data.fourth_street_sales = 1 := by sorry
  have h5 : data.first_street_sales + data.second_street_sales + data.fourth_street_sales = 
            data.total_commission / data.commission_per_sale := by sorry
  sorry

end NUMINAMATH_CALUDE_second_street_sales_l3331_333189


namespace NUMINAMATH_CALUDE_cody_tickets_l3331_333165

theorem cody_tickets (initial : ℝ) (lost_bet : ℝ) (spent_beanie : ℝ) (won_game : ℝ) (dropped : ℝ)
  (h1 : initial = 56.5)
  (h2 : lost_bet = 6.3)
  (h3 : spent_beanie = 25.75)
  (h4 : won_game = 10.25)
  (h5 : dropped = 3.1) :
  initial - lost_bet - spent_beanie + won_game - dropped = 31.6 := by
  sorry

end NUMINAMATH_CALUDE_cody_tickets_l3331_333165


namespace NUMINAMATH_CALUDE_sports_books_count_l3331_333147

theorem sports_books_count (total_books school_books : ℕ) 
  (h1 : total_books = 58) 
  (h2 : school_books = 19) : 
  total_books - school_books = 39 := by
sorry

end NUMINAMATH_CALUDE_sports_books_count_l3331_333147


namespace NUMINAMATH_CALUDE_shortest_wire_length_l3331_333138

/-- The length of the shortest wire around two circular poles -/
theorem shortest_wire_length (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 18) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let straight_section := 2 * Real.sqrt ((r1 + r2)^2 - (r2 - r1)^2)
  let small_circle_arc := 2 * π * r1 * (1/3)
  let large_circle_arc := 2 * π * r2 * (2/3)
  straight_section + small_circle_arc + large_circle_arc = 12 * Real.sqrt 3 + 14 * π :=
by sorry

end NUMINAMATH_CALUDE_shortest_wire_length_l3331_333138


namespace NUMINAMATH_CALUDE_f_range_contains_interval_f_range_may_extend_l3331_333181

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos (x/3))^2 + 2*Real.pi * Real.arcsin (x/3) - 3*(Real.arcsin (x/3))^2 + 
  (Real.pi^2/4)*(x^3 - x^2 + 4*x - 8)

theorem f_range_contains_interval :
  ∀ y ∈ Set.Icc (Real.pi^2/4) ((9*Real.pi^2)/4),
  ∃ x ∈ Set.Icc (-3) 3, f x = y :=
by sorry

theorem f_range_may_extend :
  ∃ y, (y < Real.pi^2/4 ∨ y > (9*Real.pi^2)/4) ∧
  ∃ x ∈ Set.Icc (-3) 3, f x = y :=
by sorry

end NUMINAMATH_CALUDE_f_range_contains_interval_f_range_may_extend_l3331_333181


namespace NUMINAMATH_CALUDE_retailer_profit_is_twenty_percent_l3331_333170

/-- Calculates the percentage profit of a retailer given wholesale price, retail price, and discount percentage. -/
def calculate_percentage_profit (wholesale_price retail_price discount_percent : ℚ) : ℚ :=
  let discount := discount_percent * retail_price / 100
  let selling_price := retail_price - discount
  let profit := selling_price - wholesale_price
  (profit / wholesale_price) * 100

/-- Theorem stating that under the given conditions, the retailer's percentage profit is 20%. -/
theorem retailer_profit_is_twenty_percent :
  calculate_percentage_profit 99 132 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_is_twenty_percent_l3331_333170


namespace NUMINAMATH_CALUDE_inequality_proof_l3331_333131

theorem inequality_proof (a b c : ℝ) 
  (ha : 1 - a^2 ≥ 0) (hb : 1 - b^2 ≥ 0) (hc : 1 - c^2 ≥ 0) : 
  Real.sqrt (1 - a^2) + Real.sqrt (1 - b^2) + Real.sqrt (1 - c^2) ≤ 
  Real.sqrt (9 - (a + b + c)^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3331_333131


namespace NUMINAMATH_CALUDE_smallest_sum_x_y_l3331_333172

theorem smallest_sum_x_y (x y : ℕ+) 
  (h1 : (2010 : ℚ) / 2011 < (x : ℚ) / y)
  (h2 : (x : ℚ) / y < (2011 : ℚ) / 2012) :
  ∀ (a b : ℕ+), 
    ((2010 : ℚ) / 2011 < (a : ℚ) / b ∧ (a : ℚ) / b < (2011 : ℚ) / 2012) →
    (x + y : ℕ) ≤ (a + b : ℕ) ∧
    (x + y : ℕ) = 8044 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_x_y_l3331_333172


namespace NUMINAMATH_CALUDE_paint_one_third_square_l3331_333154

theorem paint_one_third_square (n : ℕ) (k : ℕ) : n = 18 ∧ k = 6 →
  Nat.choose n k = 18564 := by
  sorry

end NUMINAMATH_CALUDE_paint_one_third_square_l3331_333154


namespace NUMINAMATH_CALUDE_count_special_integers_l3331_333166

theorem count_special_integers : 
  ∃ (S : Finset ℤ), 
    (∀ n ∈ S, 200 < n ∧ n < 300 ∧ ∃ (r k : ℤ), n = 63 * k + r ∧ 0 ≤ r ∧ r < 5) ∧
    (∀ n : ℤ, 200 < n → n < 300 → (∃ (r k : ℤ), n = 63 * k + r ∧ 0 ≤ r ∧ r < 5) → n ∈ S) ∧
    Finset.card S = 5 :=
sorry

end NUMINAMATH_CALUDE_count_special_integers_l3331_333166


namespace NUMINAMATH_CALUDE_certain_number_proof_l3331_333120

theorem certain_number_proof : ∃ x : ℚ, 346 * x = 173 * 240 ∧ x = 120 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3331_333120


namespace NUMINAMATH_CALUDE_F_odd_and_increasing_l3331_333180

-- Define f(x) implicitly using the given condition
noncomputable def f : ℝ → ℝ := fun x => Real.exp (x * Real.log 2)

-- Define F(x) using f(x)
noncomputable def F : ℝ → ℝ := fun x => f x - 1 / f x

-- Theorem stating that F is odd and increasing
theorem F_odd_and_increasing :
  (∀ x : ℝ, F (-x) = -F x) ∧
  (∀ x y : ℝ, x < y → F x < F y) :=
by sorry

end NUMINAMATH_CALUDE_F_odd_and_increasing_l3331_333180


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3331_333122

theorem polynomial_factorization (x : ℝ) :
  4 * (x + 5) * (x + 6) * (x + 10) * (x + 12) - 3 * x^2 =
  (2 * x^2 + 35 * x + 120) * (x + 8) * (2 * x + 15) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3331_333122


namespace NUMINAMATH_CALUDE_touching_circle_exists_l3331_333116

-- Define the rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the configuration of circles
structure CircleConfiguration where
  rect : Rectangle
  circle1 : Circle
  circle2 : Circle
  circle3 : Circle

-- Define the property that circles touch each other and the rectangle sides
def circlesValidConfiguration (config : CircleConfiguration) : Prop :=
  -- Circles touch each other
  (config.circle1.center.1 + config.circle1.radius = config.circle2.center.1 - config.circle2.radius) ∧
  (config.circle2.center.1 + config.circle2.radius = config.circle3.center.1 - config.circle3.radius) ∧
  -- Circles touch the rectangle sides
  (config.circle1.center.2 = config.circle1.radius) ∧
  (config.circle2.center.2 = config.rect.height - config.circle2.radius) ∧
  (config.circle3.center.2 = config.circle3.radius)

-- Define the existence of a circle touching all three circles and one side of the rectangle
def existsTouchingCircle (config : CircleConfiguration) : Prop :=
  ∃ (x : ℝ), x > 0 ∧
    -- The new circle touches circle1 and circle2
    (x + config.circle1.radius)^2 + config.circle1.radius^2 = (x + config.circle2.radius)^2 + (config.circle2.center.2 - config.circle1.center.2)^2 ∧
    -- The new circle touches circle2 and circle3
    (x + config.circle2.radius)^2 + (config.rect.height - config.circle2.center.2 - x)^2 = (x + config.circle3.radius)^2 + (config.circle3.center.2 - x)^2

-- The theorem to be proved
theorem touching_circle_exists (config : CircleConfiguration) 
  (h1 : config.circle1.radius = 1)
  (h2 : config.circle2.radius = 3)
  (h3 : config.circle3.radius = 4)
  (h4 : circlesValidConfiguration config) :
  existsTouchingCircle config :=
sorry

end NUMINAMATH_CALUDE_touching_circle_exists_l3331_333116


namespace NUMINAMATH_CALUDE_problem_statement_l3331_333148

theorem problem_statement (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! d : ℝ, d > 0 ∧ 1 / (a + d) + 1 / (b + d) + 1 / (c + d) = 2 / d ∧
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → a * x + b * y + c * z = x * y * z →
    x + y + z ≥ (2 / d) * Real.sqrt ((a + d) * (b + d) * (c + d)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3331_333148


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l3331_333145

theorem cube_volume_ratio (q p : ℝ) (h : p = 3 * q) : q^3 / p^3 = 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l3331_333145


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l3331_333193

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 3 ∧ b = 4) ∨ (a = 3 ∧ c = 4) ∨ (b = 3 ∧ c = 4) →
  a^2 + b^2 = c^2 →
  c = 5 ∨ c = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l3331_333193


namespace NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l3331_333113

theorem least_positive_integer_with_given_remainders : 
  ∃ (d : ℕ), d > 0 ∧ d % 7 = 1 ∧ d % 5 = 2 ∧ d % 3 = 2 ∧ 
  ∀ (n : ℕ), n > 0 ∧ n % 7 = 1 ∧ n % 5 = 2 ∧ n % 3 = 2 → d ≤ n :=
by
  use 92
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l3331_333113


namespace NUMINAMATH_CALUDE_midpoint_region_area_l3331_333135

/-- A regular hexagon with area 16 -/
structure RegularHexagon :=
  (area : ℝ)
  (is_regular : Bool)
  (area_eq_16 : area = 16)

/-- The midpoint of a side of the hexagon -/
structure Midpoint :=
  (hexagon : RegularHexagon)

/-- A region formed by connecting four consecutive midpoints -/
structure MidpointRegion :=
  (hexagon : RegularHexagon)
  (midpoints : Fin 4 → Midpoint)
  (consecutive : ∀ i : Fin 3, (midpoints i).hexagon = (midpoints (i + 1)).hexagon)

/-- The theorem statement -/
theorem midpoint_region_area (region : MidpointRegion) : 
  (region.hexagon.area / 2) = 8 :=
sorry

end NUMINAMATH_CALUDE_midpoint_region_area_l3331_333135


namespace NUMINAMATH_CALUDE_walking_time_calculation_l3331_333130

/-- Given a person walking at a constant rate who covers 45 meters in 15 minutes,
    prove that it will take 30 minutes to cover an additional 90 meters. -/
theorem walking_time_calculation (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ)
    (h1 : initial_distance = 45)
    (h2 : initial_time = 15)
    (h3 : additional_distance = 90) :
    additional_distance / (initial_distance / initial_time) = 30 := by
  sorry


end NUMINAMATH_CALUDE_walking_time_calculation_l3331_333130


namespace NUMINAMATH_CALUDE_first_day_is_saturday_l3331_333123

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a month -/
structure MonthDay where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Function to get the day of the week for a given day number -/
def getDayOfWeek (dayNumber : Nat) : DayOfWeek := sorry

/-- Theorem stating that if the 25th is a Tuesday, the 1st is a Saturday -/
theorem first_day_is_saturday 
  (h : getDayOfWeek 25 = DayOfWeek.Tuesday) : 
  getDayOfWeek 1 = DayOfWeek.Saturday := by
  sorry

end NUMINAMATH_CALUDE_first_day_is_saturday_l3331_333123


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l3331_333153

/-- Represents a department in the unit -/
inductive Department
| A
| B
| C

/-- The number of employees in each department -/
def employeeCount (d : Department) : ℕ :=
  match d with
  | .A => 27
  | .B => 63
  | .C => 81

/-- The number of people drawn from department B -/
def drawnFromB : ℕ := 7

/-- The number of people drawn from a department in stratified sampling -/
def peopleDrawn (d : Department) : ℚ :=
  (employeeCount d : ℚ) * (drawnFromB : ℚ) / (employeeCount .B : ℚ)

/-- The total number of people drawn from all departments -/
def totalDrawn : ℚ :=
  peopleDrawn .A + peopleDrawn .B + peopleDrawn .C

theorem stratified_sampling_result :
  totalDrawn = 23 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_l3331_333153


namespace NUMINAMATH_CALUDE_fraction_addition_l3331_333173

theorem fraction_addition (d : ℝ) : (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3331_333173


namespace NUMINAMATH_CALUDE_inequality_solution_l3331_333197

theorem inequality_solution (a : ℝ) :
  (a = 0 → ¬∃ x, (1 - a * x)^2 < 1) ∧
  (a < 0 → ∀ x, (1 - a * x)^2 < 1 ↔ (2 / a < x ∧ x < 0)) ∧
  (a > 0 → ∀ x, (1 - a * x)^2 < 1 ↔ (0 < x ∧ x < 2 / a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3331_333197


namespace NUMINAMATH_CALUDE_complement_of_B_relative_to_A_l3331_333141

def A : Set Int := {-1, 0, 1, 2, 3}
def B : Set Int := {-1, 1}

theorem complement_of_B_relative_to_A :
  {x : Int | x ∈ A ∧ x ∉ B} = {0, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_relative_to_A_l3331_333141


namespace NUMINAMATH_CALUDE_inequalities_hold_l3331_333161

theorem inequalities_hold (a b c x y z : ℝ) 
  (h1 : x^2 < a^2) (h2 : y^2 < b^2) (h3 : z^2 < c^2) :
  (x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧ 
  (x^3 + y^3 + z^3 < a^3 + b^3 + c^3) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l3331_333161


namespace NUMINAMATH_CALUDE_faster_speed_problem_l3331_333159

/-- Proves that the faster speed is 15 km/hr given the conditions of the problem -/
theorem faster_speed_problem (actual_distance : ℝ) (original_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 10)
  (h2 : original_speed = 5)
  (h3 : additional_distance = 20) :
  let time := actual_distance / original_speed
  let faster_speed := (actual_distance + additional_distance) / time
  faster_speed = 15 := by sorry

end NUMINAMATH_CALUDE_faster_speed_problem_l3331_333159


namespace NUMINAMATH_CALUDE_square_identification_l3331_333177

-- Define the points
def A : ℝ × ℝ := (4, 1)
def B : ℝ × ℝ := (2, 4)
def C : ℝ × ℝ := (5, 6)
def D : ℝ × ℝ := (3, 5)
def E : ℝ × ℝ := (7, 3)

-- Define a function to calculate the squared distance between two points
def squaredDistance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Theorem statement
theorem square_identification :
  -- The distances AB, BC, CE, and EA are equal (forming a rhombus)
  squaredDistance A B = squaredDistance B C ∧
  squaredDistance B C = squaredDistance C E ∧
  squaredDistance C E = squaredDistance E A ∧
  -- The diagonals AC and BE are equal (making it a square)
  squaredDistance A C = squaredDistance B E ∧
  -- Point D is not part of this square
  (squaredDistance A D ≠ squaredDistance A B ∨
   squaredDistance B D ≠ squaredDistance B C ∨
   squaredDistance C D ≠ squaredDistance C E ∨
   squaredDistance E D ≠ squaredDistance E A) :=
by sorry


end NUMINAMATH_CALUDE_square_identification_l3331_333177


namespace NUMINAMATH_CALUDE_movie_collection_difference_l3331_333119

theorem movie_collection_difference (shared_movies : ℕ) (andrew_total : ℕ) (john_unique : ℕ)
  (h1 : shared_movies = 12)
  (h2 : andrew_total = 23)
  (h3 : john_unique = 8) :
  andrew_total - shared_movies + john_unique = 19 := by
sorry

end NUMINAMATH_CALUDE_movie_collection_difference_l3331_333119


namespace NUMINAMATH_CALUDE_divisible_by_thirteen_l3331_333133

theorem divisible_by_thirteen (n : ℕ) (h : n > 0) :
  ∃ m : ℤ, 4^(2*n - 1) + 3^(n + 1) = 13 * m := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_thirteen_l3331_333133


namespace NUMINAMATH_CALUDE_alphabet_value_problem_l3331_333114

theorem alphabet_value_problem (T M A H E : ℤ) : 
  T = 15 →
  M + A + T + H = 47 →
  T + E + A + M = 58 →
  M + E + E + T = 45 →
  M = 8 := by
sorry

end NUMINAMATH_CALUDE_alphabet_value_problem_l3331_333114


namespace NUMINAMATH_CALUDE_minkowski_sum_properties_l3331_333151

/-- A convex polygon with perimeter and area -/
structure ConvexPolygon where
  perimeter : ℝ
  area : ℝ

/-- The Minkowski sum of a convex polygon and a circle -/
def minkowskiSum (K : ConvexPolygon) (r : ℝ) : Set (ℝ × ℝ) := sorry

/-- The length of the curve resulting from the Minkowski sum -/
def curveLength (K : ConvexPolygon) (r : ℝ) : ℝ := sorry

/-- The area of the figure bounded by the Minkowski sum -/
def boundedArea (K : ConvexPolygon) (r : ℝ) : ℝ := sorry

/-- Main theorem about the Minkowski sum of a convex polygon and a circle -/
theorem minkowski_sum_properties (K : ConvexPolygon) (r : ℝ) :
  (curveLength K r = K.perimeter + 2 * Real.pi * r) ∧
  (boundedArea K r = K.area + K.perimeter * r + Real.pi * r^2) := by
  sorry

end NUMINAMATH_CALUDE_minkowski_sum_properties_l3331_333151


namespace NUMINAMATH_CALUDE_james_change_l3331_333198

/-- Calculates the change received when buying candy -/
def calculate_change (num_packs : ℕ) (cost_per_pack : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_packs * cost_per_pack)

/-- Proves that James received $11 in change -/
theorem james_change :
  let num_packs : ℕ := 3
  let cost_per_pack : ℕ := 3
  let amount_paid : ℕ := 20
  calculate_change num_packs cost_per_pack amount_paid = 11 := by
  sorry

end NUMINAMATH_CALUDE_james_change_l3331_333198


namespace NUMINAMATH_CALUDE_square_difference_from_sum_and_difference_l3331_333108

theorem square_difference_from_sum_and_difference (a b : ℚ) 
  (h1 : a + b = 9 / 17) (h2 : a - b = 1 / 51) : 
  a^2 - b^2 = 3 / 289 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_from_sum_and_difference_l3331_333108


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3331_333139

theorem max_value_on_circle (x y : ℝ) : 
  x^2 + y^2 = 1 → (y / (x + 2) ≤ Real.sqrt 3 / 3) ∧ 
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 1 ∧ y₀ / (x₀ + 2) = Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3331_333139


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l3331_333183

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 1)) / ((x - 3)^2)) ↔ x ≥ -1 ∧ x ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l3331_333183


namespace NUMINAMATH_CALUDE_graph_connected_probability_l3331_333109

def n : ℕ := 20
def edges_removed : ℕ := 35

theorem graph_connected_probability :
  let total_edges := n * (n - 1) / 2
  let remaining_edges := total_edges - edges_removed
  let prob_disconnected := n * (Nat.choose remaining_edges (remaining_edges - n + 1)) / (Nat.choose total_edges edges_removed)
  (1 : ℚ) - prob_disconnected = 1 - (20 * Nat.choose 171 16) / Nat.choose 190 35 := by
  sorry

end NUMINAMATH_CALUDE_graph_connected_probability_l3331_333109


namespace NUMINAMATH_CALUDE_probability_no_3x3_red_l3331_333169

/-- Represents a 4x4 grid where each cell can be colored red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a 3x3 subgrid starting at (i, j) is all red -/
def has_red_3x3 (g : Grid) (i j : Fin 2) : Prop :=
  ∀ (x y : Fin 3), g (i + x) (j + y) = true

/-- A grid is valid if it doesn't contain a 3x3 red square -/
def is_valid_grid (g : Grid) : Prop :=
  ¬ ∃ (i j : Fin 2), has_red_3x3 g i j

/-- The probability of a single cell being red -/
def p_red : ℚ := 1/2

/-- The total number of possible 4x4 grids -/
def total_grids : ℕ := 2^16

/-- The number of valid 4x4 grids (without 3x3 red squares) -/
def valid_grids : ℕ := 65152

theorem probability_no_3x3_red : 
  (valid_grids : ℚ) / total_grids = 509 / 512 :=
sorry

end NUMINAMATH_CALUDE_probability_no_3x3_red_l3331_333169


namespace NUMINAMATH_CALUDE_tv_price_reduction_l3331_333105

theorem tv_price_reduction (original_price original_quantity : ℝ) 
  (h1 : original_price > 0)
  (h2 : original_quantity > 0)
  (h3 : ∃ x : ℝ, 
    (original_price * (1 - x / 100) * (original_quantity * 1.8) = 
     original_price * original_quantity * 1.44000000000000014)) :
  ∃ x : ℝ, x = 20 ∧ 
    (original_price * (1 - x / 100) * (original_quantity * 1.8) = 
     original_price * original_quantity * 1.44000000000000014) :=
by sorry

end NUMINAMATH_CALUDE_tv_price_reduction_l3331_333105


namespace NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l3331_333118

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l3331_333118


namespace NUMINAMATH_CALUDE_cider_pints_is_180_l3331_333160

/-- Represents the number of pints of cider that can be made given the following conditions:
  * 20 golden delicious, 40 pink lady, and 30 granny smith apples make one pint of cider
  * Each farmhand can pick 120 golden delicious, 240 pink lady, and 180 granny smith apples per hour
  * There are 6 farmhands working 5 hours
  * The ratio of golden delicious : pink lady : granny smith apples gathered is 1:2:1.5
-/
def cider_pints : ℕ :=
  let golden_per_pint : ℕ := 20
  let pink_per_pint : ℕ := 40
  let granny_per_pint : ℕ := 30
  let golden_per_hour : ℕ := 120
  let pink_per_hour : ℕ := 240
  let granny_per_hour : ℕ := 180
  let farmhands : ℕ := 6
  let hours : ℕ := 5
  let golden_total : ℕ := golden_per_hour * hours * farmhands
  let pink_total : ℕ := pink_per_hour * hours * farmhands
  let granny_total : ℕ := granny_per_hour * hours * farmhands
  golden_total / golden_per_pint

theorem cider_pints_is_180 : cider_pints = 180 := by
  sorry

end NUMINAMATH_CALUDE_cider_pints_is_180_l3331_333160


namespace NUMINAMATH_CALUDE_tank_filling_proof_l3331_333179

/-- The number of buckets required to fill a tank with the original bucket size -/
def original_buckets : ℕ := 10

/-- The number of buckets required to fill the tank with reduced bucket capacity -/
def reduced_buckets : ℕ := 25

/-- The ratio of reduced bucket capacity to original bucket capacity -/
def capacity_ratio : ℚ := 2 / 5

theorem tank_filling_proof :
  original_buckets * 1 = reduced_buckets * capacity_ratio :=
by sorry

end NUMINAMATH_CALUDE_tank_filling_proof_l3331_333179


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3331_333182

theorem quadratic_inequality_solution (a b : ℝ) (h : ∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :
  (a = 1 ∧ b = 2) ∧
  (∀ m : ℝ,
    (m = 2 → ∀ x, ¬(x^2 - (m + 2) * x + 2 * m < 0)) ∧
    (m < 2 → ∀ x, x^2 - (m + 2) * x + 2 * m < 0 ↔ m < x ∧ x < 2) ∧
    (m > 2 → ∀ x, x^2 - (m + 2) * x + 2 * m < 0 ↔ 2 < x ∧ x < m)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3331_333182


namespace NUMINAMATH_CALUDE_points_on_unit_circle_l3331_333158

theorem points_on_unit_circle (t : ℝ) :
  let x := (2 - t^2) / (2 + t^2)
  let y := 3 * t / (2 + t^2)
  x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_points_on_unit_circle_l3331_333158


namespace NUMINAMATH_CALUDE_renu_work_time_l3331_333142

/-- The number of days it takes Renu to complete the work alone -/
def renu_days : ℝ := 8

/-- The number of days it takes Suma to complete the work alone -/
def suma_days : ℝ := 4.8

/-- The number of days it takes Renu and Suma to complete the work together -/
def combined_days : ℝ := 3

theorem renu_work_time :
  (1 / renu_days) + (1 / suma_days) = (1 / combined_days) :=
sorry

end NUMINAMATH_CALUDE_renu_work_time_l3331_333142


namespace NUMINAMATH_CALUDE_special_function_value_l3331_333184

/-- A function satisfying f(xy) = f(x)/y² for positive reals -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / (y ^ 2)

/-- Theorem stating that if f is a special function and f(40) = 50, then f(80) = 12.5 -/
theorem special_function_value
  (f : ℝ → ℝ)
  (h_special : special_function f)
  (h_f40 : f 40 = 50) :
  f 80 = 12.5 := by
sorry

end NUMINAMATH_CALUDE_special_function_value_l3331_333184


namespace NUMINAMATH_CALUDE_average_monthly_balance_l3331_333132

def january_balance : ℝ := 120
def february_balance : ℝ := 250
def march_balance : ℝ := 200
def april_balance : ℝ := 200
def may_balance : ℝ := 180
def num_months : ℝ := 5

theorem average_monthly_balance :
  (january_balance + february_balance + march_balance + april_balance + may_balance) / num_months = 190 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l3331_333132


namespace NUMINAMATH_CALUDE_lunch_spending_difference_l3331_333185

theorem lunch_spending_difference (total_spent friend_spent : ℕ) : 
  total_spent = 17 →
  friend_spent = 10 →
  friend_spent > total_spent - friend_spent →
  friend_spent - (total_spent - friend_spent) = 3 := by
  sorry

end NUMINAMATH_CALUDE_lunch_spending_difference_l3331_333185


namespace NUMINAMATH_CALUDE_no_common_solution_l3331_333104

theorem no_common_solution : ¬ ∃ (x y : ℝ), (x^2 - 6*x + y + 9 = 0) ∧ (x^2 + 4*y + 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l3331_333104


namespace NUMINAMATH_CALUDE_quarter_circle_radius_l3331_333146

theorem quarter_circle_radius (x y z : ℝ) (h_right_angle : x^2 + y^2 = z^2)
  (h_xy_area : π * x^2 / 4 = 2 * π) (h_xz_arc : π * y / 2 = 6 * π) :
  z / 2 = Real.sqrt 152 := by
  sorry

end NUMINAMATH_CALUDE_quarter_circle_radius_l3331_333146


namespace NUMINAMATH_CALUDE_circle_angle_sum_l3331_333140

/-- Given a circle with points X, Y, and Z, where arc XY = 50°, arc YZ = 45°, arc ZX = 90°,
    angle α = (arc XZ - arc YZ) / 2, and angle β = arc YZ / 2,
    prove that the sum of angles α and β equals 47.5°. -/
theorem circle_angle_sum (arcXY arcYZ arcZX : Real) (α β : Real) :
  arcXY = 50 ∧ arcYZ = 45 ∧ arcZX = 90 ∧
  α = (arcXY + arcYZ - arcYZ) / 2 ∧
  β = arcYZ / 2 →
  α + β = 47.5 := by
sorry


end NUMINAMATH_CALUDE_circle_angle_sum_l3331_333140


namespace NUMINAMATH_CALUDE_meeting_day_is_thursday_l3331_333190

-- Define the days of the week
inductive Day : Type
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

-- Define a function to determine if Joãozinho lies on a given day
def lies_on_day (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Thursday ∨ d = Day.Saturday

-- Define a function to get the next day
def next_day (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

-- Theorem statement
theorem meeting_day_is_thursday :
  ∀ (d : Day),
    lies_on_day d →
    (lies_on_day d → d ≠ Day.Saturday) →
    (lies_on_day d → next_day d ≠ Day.Wednesday) →
    d = Day.Thursday :=
by
  sorry


end NUMINAMATH_CALUDE_meeting_day_is_thursday_l3331_333190


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l3331_333128

/-- Defines an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Defines a line passing through two points -/
structure Line where
  m : ℝ
  c : ℝ

/-- Theorem stating the properties of the ellipse and the intersecting line -/
theorem ellipse_and_line_properties
  (M : Ellipse)
  (h_eccentricity : M.a^2 - M.b^2 = M.a^2 / 2)
  (AB : Line)
  (h_AB_points : AB.m * (-M.a) + AB.c = 0 ∧ AB.c = M.b)
  (h_AB_distance : (M.a * M.b / Real.sqrt (M.a^2 + M.b^2))^2 = 2/3)
  (l : Line)
  (h_l_point : l.c = -1)
  (h_intersection_ratio : ∃ (y₁ y₂ : ℝ), y₁ = -3 * y₂ ∧
    y₁ + y₂ = -2 * l.m / (l.m^2 + 2) ∧
    y₁ * y₂ = -1 / (l.m^2 + 2)) :
  M.a^2 = 2 ∧ M.b^2 = 1 ∧ l.m = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l3331_333128


namespace NUMINAMATH_CALUDE_quadratic_t_range_l3331_333111

/-- Represents a quadratic function of the form ax² + bx - 2 --/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Theorem statement for the range of t in the given quadratic equation --/
theorem quadratic_t_range (f : QuadraticFunction) 
  (h1 : f.a * (-1)^2 + f.b * (-1) - 2 = 0)  -- -1 is a root
  (h2 : 0 < -f.b / (2 * f.a))  -- vertex x-coordinate is positive (4th quadrant)
  (h3 : 0 < f.a)  -- parabola opens upward (4th quadrant)
  : -2 < 3 * f.a + f.b ∧ 3 * f.a + f.b < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_t_range_l3331_333111


namespace NUMINAMATH_CALUDE_tom_climbing_time_l3331_333144

/-- Given that Elizabeth takes 30 minutes to climb a hill and Tom takes four times as long,
    prove that Tom's climbing time is 2 hours. -/
theorem tom_climbing_time :
  let elizabeth_time : ℕ := 30 -- Elizabeth's climbing time in minutes
  let tom_factor : ℕ := 4 -- Tom takes four times as long as Elizabeth
  let tom_time : ℕ := elizabeth_time * tom_factor -- Tom's climbing time in minutes
  tom_time / 60 = 2 -- Tom's climbing time in hours
:= by sorry

end NUMINAMATH_CALUDE_tom_climbing_time_l3331_333144


namespace NUMINAMATH_CALUDE_factor_decomposition_l3331_333174

theorem factor_decomposition (a b : Int) : 
  a * b = 96 → a^2 + b^2 = 208 → 
  ((a = 8 ∧ b = 12) ∨ (a = -8 ∧ b = -12) ∨ (a = 12 ∧ b = 8) ∨ (a = -12 ∧ b = -8)) :=
by sorry

end NUMINAMATH_CALUDE_factor_decomposition_l3331_333174


namespace NUMINAMATH_CALUDE_number_of_proper_subsets_of_P_l3331_333171

def M : Finset ℤ := {-1, 1, 2, 3, 4, 5}
def N : Finset ℤ := {1, 2, 4}
def P : Finset ℤ := M ∩ N

theorem number_of_proper_subsets_of_P : (Finset.powerset P).card - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_proper_subsets_of_P_l3331_333171


namespace NUMINAMATH_CALUDE_equation_solution_l3331_333124

theorem equation_solution :
  ∃ x : ℝ, x + 2*x + 12 = 500 - (3*x + 4*x) → x = 48.8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3331_333124


namespace NUMINAMATH_CALUDE_class_size_l3331_333125

theorem class_size (boys girls : ℕ) : 
  (boys : ℚ) / girls = 4 / 3 →
  (boys - 8)^2 = girls - 14 →
  boys + girls = 42 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3331_333125


namespace NUMINAMATH_CALUDE_input_is_input_statement_l3331_333178

-- Define the type for programming language statements
inductive ProgrammingStatement
  | PRINT
  | INPUT
  | THEN
  | END

-- Define the function of each statement
def statementFunction (s : ProgrammingStatement) : String :=
  match s with
  | ProgrammingStatement.PRINT => "output"
  | ProgrammingStatement.INPUT => "input"
  | ProgrammingStatement.THEN => "conditional"
  | ProgrammingStatement.END => "termination"

-- Theorem: INPUT is the input statement
theorem input_is_input_statement :
  ∃ s : ProgrammingStatement, statementFunction s = "input" ∧ s = ProgrammingStatement.INPUT :=
by sorry

end NUMINAMATH_CALUDE_input_is_input_statement_l3331_333178


namespace NUMINAMATH_CALUDE_james_earnings_l3331_333103

/-- Represents the amount of water collected per inch of rain -/
def water_per_inch : ℝ := 15

/-- Represents the amount of rain on Monday in inches -/
def monday_rain : ℝ := 4

/-- Represents the amount of rain on Tuesday in inches -/
def tuesday_rain : ℝ := 3

/-- Represents the price per gallon of water in dollars -/
def price_per_gallon : ℝ := 1.2

/-- Calculates the total amount of money James made from selling all the water -/
def total_money : ℝ :=
  (monday_rain * water_per_inch + tuesday_rain * water_per_inch) * price_per_gallon

/-- Theorem stating that James made $126 from selling all the water -/
theorem james_earnings : total_money = 126 := by
  sorry

end NUMINAMATH_CALUDE_james_earnings_l3331_333103


namespace NUMINAMATH_CALUDE_michael_needs_eleven_more_l3331_333152

/-- Given Michael's current money and the total cost of items he wants to buy,
    calculate the additional money he needs. -/
def additional_money_needed (current_money total_cost : ℕ) : ℕ :=
  if total_cost > current_money then total_cost - current_money else 0

/-- Theorem stating that Michael needs $11 more to buy all items. -/
theorem michael_needs_eleven_more :
  let current_money : ℕ := 50
  let cake_cost : ℕ := 20
  let bouquet_cost : ℕ := 36
  let balloons_cost : ℕ := 5
  let total_cost : ℕ := cake_cost + bouquet_cost + balloons_cost
  additional_money_needed current_money total_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_michael_needs_eleven_more_l3331_333152


namespace NUMINAMATH_CALUDE_power_division_result_l3331_333106

theorem power_division_result : 8^15 / 64^7 = 8 := by sorry

end NUMINAMATH_CALUDE_power_division_result_l3331_333106


namespace NUMINAMATH_CALUDE_romanov_savings_l3331_333194

/-- Represents the electricity pricing and consumption data for the Romanov family --/
structure ElectricityData where
  multi_tariff_meter_cost : ℝ
  installation_cost : ℝ
  monthly_consumption : ℝ
  night_consumption : ℝ
  day_rate : ℝ
  night_rate : ℝ
  standard_rate : ℝ
  years : ℕ

/-- Calculates the savings from using a multi-tariff meter over the given period --/
def calculate_savings (data : ElectricityData) : ℝ :=
  let standard_cost := data.standard_rate * data.monthly_consumption * 12 * data.years
  let day_consumption := data.monthly_consumption - data.night_consumption
  let multi_tariff_cost := (data.day_rate * day_consumption + data.night_rate * data.night_consumption) * 12 * data.years
  let total_multi_tariff_cost := multi_tariff_cost + data.multi_tariff_meter_cost + data.installation_cost
  standard_cost - total_multi_tariff_cost

/-- Theorem stating the savings for the Romanov family --/
theorem romanov_savings :
  let data : ElectricityData := {
    multi_tariff_meter_cost := 3500,
    installation_cost := 1100,
    monthly_consumption := 300,
    night_consumption := 230,
    day_rate := 5.2,
    night_rate := 3.4,
    standard_rate := 4.6,
    years := 3
  }
  calculate_savings data = 3824 := by
  sorry

end NUMINAMATH_CALUDE_romanov_savings_l3331_333194


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l3331_333186

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals : 
  num_diagonals dodecagon_sides = 54 := by sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l3331_333186


namespace NUMINAMATH_CALUDE_remaining_money_is_130_l3331_333117

/-- Given an initial amount of money, calculate the remaining amount after spending on books and DVDs -/
def remaining_money (initial : ℚ) : ℚ :=
  let after_books := initial - (1/4 * initial + 10)
  let after_dvds := after_books - (2/5 * after_books + 8)
  after_dvds

/-- Theorem: Given $320 initially, the remaining money after buying books and DVDs is $130 -/
theorem remaining_money_is_130 : remaining_money 320 = 130 := by
  sorry

#eval remaining_money 320

end NUMINAMATH_CALUDE_remaining_money_is_130_l3331_333117


namespace NUMINAMATH_CALUDE_two_from_four_combinations_l3331_333155

theorem two_from_four_combinations : Nat.choose 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_from_four_combinations_l3331_333155


namespace NUMINAMATH_CALUDE_julies_work_hours_l3331_333150

theorem julies_work_hours (hourly_rate : ℝ) (days_per_week : ℕ) (monthly_salary : ℝ) :
  hourly_rate = 5 →
  days_per_week = 6 →
  monthly_salary = 920 →
  (monthly_salary / hourly_rate) / (days_per_week * 4 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_julies_work_hours_l3331_333150


namespace NUMINAMATH_CALUDE_train_length_l3331_333127

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) :
  speed = 72 → time = 12 → speed * time * (1000 / 3600) = 240 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3331_333127


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3331_333143

theorem smallest_n_congruence (n : ℕ+) : 
  (5 * n : ℤ) ≡ 409 [ZMOD 31] ∧ 
  ∀ m : ℕ+, (5 * m : ℤ) ≡ 409 [ZMOD 31] → n ≤ m → n = 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3331_333143


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3331_333196

theorem arithmetic_expression_equality : 2 + 3 * 5 + 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3331_333196


namespace NUMINAMATH_CALUDE_system_equation_solution_l3331_333162

theorem system_equation_solution :
  ∀ (x₁ x₂ x₃ x₄ x₅ : ℝ),
  2*x₁ + x₂ + x₃ + x₄ + x₅ = 6 →
  x₁ + 2*x₂ + x₃ + x₄ + x₅ = 12 →
  x₁ + x₂ + 2*x₃ + x₄ + x₅ = 24 →
  x₁ + x₂ + x₃ + 2*x₄ + x₅ = 48 →
  x₁ + x₂ + x₃ + x₄ + 2*x₅ = 96 →
  3*x₄ + 2*x₅ = 181 :=
by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l3331_333162


namespace NUMINAMATH_CALUDE_alyssa_cookie_count_l3331_333164

/-- The number of cookies Alyanna has -/
def aiyanna_cookies : ℕ := 140

/-- The difference between Aiyanna's and Alyssa's cookies -/
def cookie_difference : ℕ := 11

/-- The number of cookies Alyssa has -/
def alyssa_cookies : ℕ := aiyanna_cookies - cookie_difference

theorem alyssa_cookie_count : alyssa_cookies = 129 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_cookie_count_l3331_333164


namespace NUMINAMATH_CALUDE_alice_probability_after_two_turns_l3331_333187

/-- Represents the probability of Alice having the ball after two turns in the basketball game. -/
def alice_has_ball_after_two_turns (
  alice_toss_prob : ℚ)  -- Probability of Alice tossing the ball to Bob
  (alice_keep_prob : ℚ)  -- Probability of Alice keeping the ball
  (bob_toss_prob : ℚ)    -- Probability of Bob tossing the ball to Alice
  (bob_keep_prob : ℚ) : ℚ :=  -- Probability of Bob keeping the ball
  alice_toss_prob * bob_toss_prob + alice_keep_prob * alice_keep_prob

/-- Theorem stating the probability of Alice having the ball after two turns -/
theorem alice_probability_after_two_turns :
  alice_has_ball_after_two_turns (2/3) (1/3) (1/4) (3/4) = 5/18 := by
  sorry

end NUMINAMATH_CALUDE_alice_probability_after_two_turns_l3331_333187


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3331_333188

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3331_333188


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3331_333102

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X : Polynomial ℝ)^4 + (X : Polynomial ℝ)^2 - 5 = 
  (X^2 - 3) * q + (4 * X^2 - 5) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3331_333102


namespace NUMINAMATH_CALUDE_exists_2x2_square_after_removal_l3331_333149

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a two-cell rectangle (domino) -/
structure Domino where
  cell1 : Cell
  cell2 : Cell

/-- The grid size -/
def gridSize : Nat := 100

/-- The number of dominoes removed -/
def removedDominoes : Nat := 1950

/-- Function to check if a cell is within the grid -/
def isValidCell (c : Cell) : Prop :=
  c.row < gridSize ∧ c.col < gridSize

/-- Function to check if two cells are adjacent -/
def areAdjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ (c1.col + 1 = c2.col ∨ c2.col + 1 = c1.col)) ∨
  (c1.col = c2.col ∧ (c1.row + 1 = c2.row ∨ c2.row + 1 = c1.row))

/-- Function to check if a domino is valid -/
def isValidDomino (d : Domino) : Prop :=
  isValidCell d.cell1 ∧ isValidCell d.cell2 ∧ areAdjacent d.cell1 d.cell2

/-- Theorem: After removing 1950 dominoes, there exists a 2x2 square in the remaining cells -/
theorem exists_2x2_square_after_removal 
  (removed : Finset Domino) 
  (h_removed : removed.card = removedDominoes) 
  (h_valid : ∀ d ∈ removed, isValidDomino d) :
  ∃ (c : Cell), isValidCell c ∧ 
    isValidCell { row := c.row, col := c.col + 1 } ∧ 
    isValidCell { row := c.row + 1, col := c.col } ∧ 
    isValidCell { row := c.row + 1, col := c.col + 1 } ∧
    (∀ d ∈ removed, d.cell1 ≠ c ∧ d.cell2 ≠ c) ∧
    (∀ d ∈ removed, d.cell1 ≠ { row := c.row, col := c.col + 1 } ∧ d.cell2 ≠ { row := c.row, col := c.col + 1 }) ∧
    (∀ d ∈ removed, d.cell1 ≠ { row := c.row + 1, col := c.col } ∧ d.cell2 ≠ { row := c.row + 1, col := c.col }) ∧
    (∀ d ∈ removed, d.cell1 ≠ { row := c.row + 1, col := c.col + 1 } ∧ d.cell2 ≠ { row := c.row + 1, col := c.col + 1 }) :=
by sorry

end NUMINAMATH_CALUDE_exists_2x2_square_after_removal_l3331_333149


namespace NUMINAMATH_CALUDE_inscribed_squares_area_l3331_333107

/-- Given three squares inscribed in right triangles with areas A, M, and N,
    where M = 5 and N = 12, prove that A = 17 + 4√15 -/
theorem inscribed_squares_area (A M N : ℝ) (hM : M = 5) (hN : N = 12) :
  A = (Real.sqrt M + Real.sqrt N) ^ 2 →
  A = 17 + 4 * Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_area_l3331_333107


namespace NUMINAMATH_CALUDE_sqrt_33_between_5_and_6_l3331_333101

theorem sqrt_33_between_5_and_6 : 5 < Real.sqrt 33 ∧ Real.sqrt 33 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_33_between_5_and_6_l3331_333101
