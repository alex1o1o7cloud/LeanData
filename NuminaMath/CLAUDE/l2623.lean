import Mathlib

namespace league_members_l2623_262341

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost of shorts in dollars -/
def shorts_cost : ℕ := tshirt_cost

/-- The total cost for one member's equipment in dollars -/
def member_cost : ℕ := 2 * (sock_cost + tshirt_cost + shorts_cost)

/-- The total cost for the league's equipment in dollars -/
def total_cost : ℕ := 4719

/-- The number of members in the league -/
def num_members : ℕ := 74

theorem league_members : 
  sock_cost = 6 ∧ 
  tshirt_cost = sock_cost + 7 ∧ 
  shorts_cost = tshirt_cost ∧
  member_cost = 2 * (sock_cost + tshirt_cost + shorts_cost) ∧
  total_cost = 4719 → 
  num_members * member_cost = total_cost :=
by sorry

end league_members_l2623_262341


namespace cut_cube_edge_count_l2623_262394

/-- Represents a cube with smaller cubes removed from its corners -/
structure CutCube where
  side_length : ℝ
  cut_length : ℝ

/-- Calculates the number of edges in a CutCube -/
def edge_count (c : CutCube) : ℕ :=
  12 + 8 * 9 / 3

/-- Theorem stating that a cube of side length 4 with corners of side length 1.5 removed has 36 edges -/
theorem cut_cube_edge_count :
  let c : CutCube := { side_length := 4, cut_length := 1.5 }
  edge_count c = 36 := by
  sorry

end cut_cube_edge_count_l2623_262394


namespace orange_cost_24_pounds_l2623_262364

/-- The cost of oranges given a rate and a quantity -/
def orange_cost (rate_price : ℚ) (rate_weight : ℚ) (weight : ℚ) : ℚ :=
  (rate_price / rate_weight) * weight

/-- Theorem: The cost of 24 pounds of oranges at a rate of $6 per 8 pounds is $18 -/
theorem orange_cost_24_pounds : orange_cost 6 8 24 = 18 := by
  sorry

end orange_cost_24_pounds_l2623_262364


namespace gain_percent_for_80_and_58_l2623_262328

/-- Calculates the gain percent given the number of articles at cost price and selling price that are equal in total value -/
def gainPercent (costArticles sellingArticles : ℕ) : ℚ :=
  let ratio : ℚ := costArticles / sellingArticles
  (ratio - 1) / ratio * 100

theorem gain_percent_for_80_and_58 :
  gainPercent 80 58 = 11 / 29 * 100 := by
  sorry

end gain_percent_for_80_and_58_l2623_262328


namespace two_digit_number_with_remainders_l2623_262391

theorem two_digit_number_with_remainders : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  n % 9 = 7 ∧ 
  n % 7 = 5 ∧ 
  n % 3 = 1 ∧ 
  n = 61 := by
sorry

end two_digit_number_with_remainders_l2623_262391


namespace compressor_stations_configuration_l2623_262350

/-- Represents a triangle with side lengths x, y, and z. -/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  triangle_inequality : x < y + z ∧ y < x + z ∧ z < x + y

/-- Theorem about the specific triangle configuration described in the problem. -/
theorem compressor_stations_configuration (a : ℝ) :
  ∃ (t : Triangle),
    t.x + t.y = 3 * t.z ∧
    t.z + t.y = t.x + a ∧
    t.x + t.z = 60 →
    0 < a ∧ a < 60 ∧
    (a = 30 → t.x = 35 ∧ t.y = 40 ∧ t.z = 25) :=
by sorry

#check compressor_stations_configuration

end compressor_stations_configuration_l2623_262350


namespace proposition_p_equivalence_l2623_262339

theorem proposition_p_equivalence :
  (∃ x, x < 1 ∧ x^2 < 1) ↔ ¬(∀ x, x < 1 → x^2 ≥ 1) :=
by sorry

end proposition_p_equivalence_l2623_262339


namespace max_sock_pairs_l2623_262320

theorem max_sock_pairs (initial_pairs : ℕ) (lost_socks : ℕ) (max_pairs : ℕ) : 
  initial_pairs = 10 →
  lost_socks = 5 →
  max_pairs = 5 →
  max_pairs = initial_pairs - (lost_socks / 2 + lost_socks % 2) :=
by sorry

end max_sock_pairs_l2623_262320


namespace arithmetic_sequence_remainder_l2623_262304

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_remainder (a₁ d aₙ : ℕ) (h1 : a₁ = 3) (h2 : d = 6) (h3 : aₙ = 309) :
  arithmetic_sequence_sum a₁ d aₙ % 7 = 2 := by
  sorry

end arithmetic_sequence_remainder_l2623_262304


namespace correct_calculation_l2623_262386

theorem correct_calculation (x : ℝ) : 2 * (3 * x + 14) = 946 → 2 * (x / 3 + 14) = 130 := by
  sorry

end correct_calculation_l2623_262386


namespace eccentricity_of_ellipse_through_roots_l2623_262323

-- Define the complex equation
def complex_equation (z : ℂ) : Prop :=
  (z - 2) * (z^2 + 3*z + 5) * (z^2 + 5*z + 8) = 0

-- Define the set of roots
def roots : Set ℂ :=
  {z : ℂ | complex_equation z}

-- Define the ellipse passing through the roots
def ellipse_through_roots (E : Set (ℝ × ℝ)) : Prop :=
  ∀ z ∈ roots, (z.re, z.im) ∈ E

-- Define the eccentricity of an ellipse
def eccentricity (E : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem eccentricity_of_ellipse_through_roots :
  ∃ E : Set (ℝ × ℝ), ellipse_through_roots E ∧ eccentricity E = Real.sqrt (1/7) :=
sorry

end eccentricity_of_ellipse_through_roots_l2623_262323


namespace expand_expression_l2623_262332

theorem expand_expression (x : ℝ) : 2 * (x + 3) * (x + 6) + x = 2 * x^2 + 19 * x + 36 := by
  sorry

end expand_expression_l2623_262332


namespace juniors_average_score_l2623_262322

theorem juniors_average_score 
  (total_students : ℝ) 
  (junior_ratio : ℝ) 
  (senior_ratio : ℝ) 
  (class_average : ℝ) 
  (senior_average : ℝ) 
  (h1 : junior_ratio = 0.2)
  (h2 : senior_ratio = 0.8)
  (h3 : junior_ratio + senior_ratio = 1)
  (h4 : class_average = 86)
  (h5 : senior_average = 85) :
  (class_average * total_students - senior_average * (senior_ratio * total_students)) / (junior_ratio * total_students) = 90 :=
by sorry

end juniors_average_score_l2623_262322


namespace present_age_of_b_l2623_262330

theorem present_age_of_b (a b : ℕ) : 
  (a + 30 = 2 * (b - 30)) →  -- In 30 years, A will be twice as old as B was 30 years ago
  (a = b + 5) →              -- A is now 5 years older than B
  b = 95 :=                  -- The present age of B is 95
by sorry

end present_age_of_b_l2623_262330


namespace greatest_integer_with_gcf_five_gcf_of_145_and_30_less_than_150_is_greatest_main_result_l2623_262324

theorem greatest_integer_with_gcf_five (n : ℕ) : n < 150 ∧ Nat.gcd n 30 = 5 → n ≤ 145 :=
by sorry

theorem gcf_of_145_and_30 : Nat.gcd 145 30 = 5 :=
by sorry

theorem less_than_150 : 145 < 150 :=
by sorry

theorem is_greatest : ∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ 145 :=
by sorry

theorem main_result : (∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ 
  (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n)) ∧ 
  (∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ 
  (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) ∧ n = 145) :=
by sorry

end greatest_integer_with_gcf_five_gcf_of_145_and_30_less_than_150_is_greatest_main_result_l2623_262324


namespace base4_calculation_l2623_262362

/-- Converts a base-4 number to base-10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base-10 number to base-4 --/
def toBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem base4_calculation :
  let a := toBase10 [1, 3, 2]  -- 231₄
  let b := toBase10 [1, 2]     -- 21₄
  let c := toBase10 [2, 3]     -- 32₄
  let d := toBase10 [2]        -- 2₄
  toBase4 (a * b + c / d) = [0, 3, 1, 6] := by
  sorry

end base4_calculation_l2623_262362


namespace min_students_with_both_devices_l2623_262336

theorem min_students_with_both_devices (n : ℕ) (laptop_users tablet_users : ℕ) : 
  laptop_users = (3 * n) / 7 →
  tablet_users = (5 * n) / 6 →
  ∃ (both : ℕ), both ≥ 11 ∧ n ≥ laptop_users + tablet_users - both :=
sorry

end min_students_with_both_devices_l2623_262336


namespace paintings_from_C_l2623_262316

-- Define the number of paintings from each school
variable (A B C : ℕ)

-- Define the total number of paintings
def T : ℕ := A + B + C

-- State the conditions
axiom not_from_A : B + C = 41
axiom not_from_B : A + C = 38
axiom from_A_and_B : A + B = 43

-- State the theorem to be proved
theorem paintings_from_C : C = 18 := by sorry

end paintings_from_C_l2623_262316


namespace orange_box_ratio_l2623_262300

theorem orange_box_ratio (total : ℕ) (given_to_mother : ℕ) (remaining : ℕ) :
  total = 9 →
  given_to_mother = 1 →
  remaining = 4 →
  (total - given_to_mother - remaining) * 2 = total - given_to_mother :=
by sorry

end orange_box_ratio_l2623_262300


namespace betty_cupcake_rate_l2623_262318

theorem betty_cupcake_rate : 
  ∀ (B : ℕ), -- Betty's cupcake rate per hour
  (5 * 8 - 3 * B = 10) → -- Difference in cupcakes after 5 hours
  B = 10 := by
sorry

end betty_cupcake_rate_l2623_262318


namespace N_swaps_rows_l2623_262344

/-- The matrix that swaps rows of a 2x2 matrix -/
def N : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]

/-- Theorem: N swaps the rows of any 2x2 matrix -/
theorem N_swaps_rows (a b c d : ℝ) :
  N • !![a, b; c, d] = !![c, d; a, b] := by
  sorry

end N_swaps_rows_l2623_262344


namespace tetrahedron_edges_lengths_l2623_262358

-- Define the tetrahedron and its circumscribed sphere
structure Tetrahedron :=
  (base_edge1 : ℝ)
  (base_edge2 : ℝ)
  (base_edge3 : ℝ)
  (inclined_edge : ℝ)
  (sphere_radius : ℝ)
  (volume : ℝ)

-- Define the conditions
def tetrahedron_conditions (t : Tetrahedron) : Prop :=
  t.base_edge1 = 2 * t.sphere_radius ∧
  t.base_edge2 / t.base_edge3 = 4 / 3 ∧
  t.volume = 40 ∧
  t.base_edge1^2 = t.base_edge2^2 + t.base_edge3^2 ∧
  t.inclined_edge^2 = t.sphere_radius^2 + (t.base_edge2 / 2)^2

-- Theorem statement
theorem tetrahedron_edges_lengths 
  (t : Tetrahedron) 
  (h : tetrahedron_conditions t) : 
  t.base_edge1 = 10 ∧ 
  t.base_edge2 = 8 ∧ 
  t.base_edge3 = 6 ∧ 
  t.inclined_edge = Real.sqrt 50 := 
sorry

end tetrahedron_edges_lengths_l2623_262358


namespace apple_cost_l2623_262303

/-- The cost of an item given the amount paid and change received -/
def itemCost (amountPaid changeReceived : ℚ) : ℚ :=
  amountPaid - changeReceived

/-- Proof that the apple costs $0.75 -/
theorem apple_cost (amountPaid changeReceived : ℚ) 
  (h1 : amountPaid = 5)
  (h2 : changeReceived = 4.25) : 
  itemCost amountPaid changeReceived = 0.75 := by
  sorry

#check apple_cost

end apple_cost_l2623_262303


namespace tangent_line_at_one_l2623_262352

/-- Given a function f: ℝ → ℝ satisfying the specified condition,
    proves that the tangent line to y = f(x) at (1, f(1)) has the equation x - y - 2 = 0 -/
theorem tangent_line_at_one (f : ℝ → ℝ) 
    (h : ∀ x, f (1 + x) = 2 * f (1 - x) - x^2 + 3*x + 1) : 
    ∃ m b, (∀ x, m * (x - 1) + f 1 = m * x + b) ∧ m = 1 ∧ b = -1 := by
  sorry

end tangent_line_at_one_l2623_262352


namespace ratio_transformation_l2623_262393

theorem ratio_transformation (x : ℚ) : 
  ((2 : ℚ) + 2) / (x + 2) = 4 / 5 → x = 3 := by
  sorry

end ratio_transformation_l2623_262393


namespace hiker_catches_cyclist_l2623_262306

/-- Proves that a hiker catches up to a cyclist in 15 minutes under specific conditions --/
theorem hiker_catches_cyclist (hiker_speed cyclist_speed : ℝ) (stop_time : ℝ) : 
  hiker_speed = 7 →
  cyclist_speed = 28 →
  stop_time = 5 / 60 →
  let distance_cyclist := cyclist_speed * stop_time
  let distance_hiker := hiker_speed * stop_time
  let distance_difference := distance_cyclist - distance_hiker
  let catch_up_time := distance_difference / hiker_speed
  catch_up_time * 60 = 15 := by
  sorry

#check hiker_catches_cyclist

end hiker_catches_cyclist_l2623_262306


namespace congruentAngles_equivalence_sameRemainder_equivalence_l2623_262308

-- Define a type for angles
structure Angle where
  measure : ℝ

-- Define congruence relation for angles
def congruentAngles (a b : Angle) : Prop := a.measure = b.measure

-- Define a type for integers with a specific modulus
structure ModInt (m : ℕ) where
  value : ℤ

-- Define same remainder relation for ModInt
def sameRemainder {m : ℕ} (a b : ModInt m) : Prop := a.value % m = b.value % m

-- Theorem: Congruence of angles is an equivalence relation
theorem congruentAngles_equivalence : Equivalence congruentAngles := by sorry

-- Theorem: Same remainder when divided by a certain number is an equivalence relation
theorem sameRemainder_equivalence (m : ℕ) : Equivalence (@sameRemainder m) := by sorry

end congruentAngles_equivalence_sameRemainder_equivalence_l2623_262308


namespace remainder_13_pow_21_mod_1000_l2623_262374

theorem remainder_13_pow_21_mod_1000 : 13^21 % 1000 = 413 := by
  sorry

end remainder_13_pow_21_mod_1000_l2623_262374


namespace triangle_angle_measure_l2623_262347

theorem triangle_angle_measure (A B C : Real) (a b c : Real) :
  -- Given conditions
  (4 * (Real.cos A)^2 + 4 * Real.cos B * Real.cos C + 1 = 4 * Real.sin B * Real.sin C) →
  (A < B) →
  (a = 2 * Real.sqrt 3) →
  (a / Real.sin A = 4) →  -- Circumradius condition
  -- Conclusion
  A = π / 3 := by
  sorry

end triangle_angle_measure_l2623_262347


namespace exist_positive_integers_satisfying_equation_l2623_262376

theorem exist_positive_integers_satisfying_equation : 
  ∃ (x y z : ℕ+), x^2006 + y^2006 = z^2007 := by
  sorry

end exist_positive_integers_satisfying_equation_l2623_262376


namespace symmetry_axis_l2623_262366

-- Define a function f with the given symmetry property
def f : ℝ → ℝ := sorry

-- State the symmetry property of f
axiom f_symmetry (x : ℝ) : f x = f (3 - x)

-- Define what it means for a vertical line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x y : ℝ, f x = y → f (2 * a - x) = y

-- Theorem stating that x = 1.5 is an axis of symmetry
theorem symmetry_axis :
  is_axis_of_symmetry 1.5 :=
sorry

end symmetry_axis_l2623_262366


namespace quadratic_equations_common_root_l2623_262331

theorem quadratic_equations_common_root (a b : ℝ) : 
  (∃! x, x^2 + a*x + b = 0 ∧ x^2 + b*x + a = 0) → 
  (a + b + 1 = 0 ∧ a ≠ b) := by
  sorry

end quadratic_equations_common_root_l2623_262331


namespace vacation_pictures_deleted_l2623_262335

theorem vacation_pictures_deleted (zoo_pics : ℕ) (museum_pics : ℕ) (remaining_pics : ℕ) : 
  zoo_pics = 41 → museum_pics = 29 → remaining_pics = 55 → 
  zoo_pics + museum_pics - remaining_pics = 15 := by
  sorry

end vacation_pictures_deleted_l2623_262335


namespace f_of_a_plus_one_l2623_262379

/-- The function f(x) = x^2 + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: For the function f(x) = x^2 + 1, f(a+1) = a^2 + 2a + 2 for any real number a -/
theorem f_of_a_plus_one (a : ℝ) : f (a + 1) = a^2 + 2*a + 2 := by
  sorry

end f_of_a_plus_one_l2623_262379


namespace factorial_equation_solutions_l2623_262343

theorem factorial_equation_solutions :
  ∀ x y z : ℕ+,
    (2 ^ x.val + 3 ^ y.val - 7 = Nat.factorial z.val) ↔ 
    ((x, y, z) = (2, 2, 3) ∨ (x, y, z) = (2, 3, 4)) := by
  sorry

end factorial_equation_solutions_l2623_262343


namespace students_not_liking_sports_l2623_262348

theorem students_not_liking_sports (total : ℕ) (basketball : ℕ) (tableTennis : ℕ) (both : ℕ) :
  total = 30 →
  basketball = 15 →
  tableTennis = 10 →
  both = 3 →
  total - (basketball + tableTennis - both) = 8 :=
by sorry

end students_not_liking_sports_l2623_262348


namespace algebraic_expression_symmetry_l2623_262361

theorem algebraic_expression_symmetry (a b c : ℝ) :
  (a * (-5)^4 + b * (-5)^2 + c = 3) →
  (a * 5^4 + b * 5^2 + c = 3) := by
sorry

end algebraic_expression_symmetry_l2623_262361


namespace max_N_is_six_l2623_262392

/-- Definition of I_k -/
def I (k : ℕ) : ℕ := 10^(k+1) + 32

/-- Definition of N(k) -/
def N (k : ℕ) : ℕ := (I k).factors.count 2

/-- Theorem: The maximum value of N(k) is 6 -/
theorem max_N_is_six :
  (∀ k : ℕ, N k ≤ 6) ∧ (∃ k : ℕ, N k = 6) := by sorry

end max_N_is_six_l2623_262392


namespace min_value_of_f_over_x_range_of_a_for_inequality_l2623_262373

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 1 + a

-- Theorem for part I
theorem min_value_of_f_over_x (x : ℝ) (h : x > 0) :
  ∃ (min_val : ℝ), min_val = -2 ∧ ∀ (y : ℝ), y > 0 → f 2 y / y ≥ min_val :=
sorry

-- Theorem for part II
theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f a x ≤ a) ↔ a ≥ -2 + Real.sqrt 7 :=
sorry

end min_value_of_f_over_x_range_of_a_for_inequality_l2623_262373


namespace student_arrangements_l2623_262388

theorem student_arrangements (n : ℕ) (h : n = 5) : 
  (n - 1) * Nat.factorial (n - 1) = 96 := by
  sorry

end student_arrangements_l2623_262388


namespace vegetable_ghee_weight_l2623_262311

/-- The weight of one liter of vegetable ghee for brand 'b' in grams -/
def weight_b : ℝ := 850

/-- The ratio of brand 'a' to brand 'b' in the mixture by volume -/
def mixture_ratio : ℚ := 3/2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3640

/-- The weight of one liter of vegetable ghee for brand 'a' in grams -/
def weight_a : ℝ := 950

theorem vegetable_ghee_weight : 
  (weight_a * (mixture_ratio / (mixture_ratio + 1)) * total_volume) + 
  (weight_b * (1 / (mixture_ratio + 1)) * total_volume) = total_weight :=
sorry

end vegetable_ghee_weight_l2623_262311


namespace classmate_height_most_suitable_for_census_l2623_262346

/-- Represents a survey option -/
inductive SurveyOption
  | LightBulbLifespan
  | ClassmateHeight
  | NationwideStudentViewing
  | MissileAccuracy

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  population_size : ℕ
  is_destructive : Bool
  data_collection_difficulty : ℕ

/-- Defines the characteristics of each survey option -/
def survey_characteristics : SurveyOption → SurveyCharacteristics
  | SurveyOption.LightBulbLifespan => ⟨100, true, 5⟩
  | SurveyOption.ClassmateHeight => ⟨30, false, 1⟩
  | SurveyOption.NationwideStudentViewing => ⟨1000000, false, 8⟩
  | SurveyOption.MissileAccuracy => ⟨50, true, 7⟩

/-- Determines if a survey is suitable for a census based on its characteristics -/
def is_census_suitable (s : SurveyCharacteristics) : Bool :=
  s.population_size ≤ 100 ∧ ¬s.is_destructive ∧ s.data_collection_difficulty ≤ 3

/-- Theorem stating that classmate height survey is most suitable for census -/
theorem classmate_height_most_suitable_for_census :
  ∀ (option : SurveyOption),
    option ≠ SurveyOption.ClassmateHeight →
    is_census_suitable (survey_characteristics SurveyOption.ClassmateHeight) ∧
    ¬is_census_suitable (survey_characteristics option) :=
  sorry


end classmate_height_most_suitable_for_census_l2623_262346


namespace a_2000_mod_9_l2623_262375

-- Define the sequence
def a : ℕ → ℕ
  | 0 => 1995
  | n + 1 => (n + 1) * a n + 1

-- State the theorem
theorem a_2000_mod_9 : a 2000 % 9 = 5 := by
  sorry

end a_2000_mod_9_l2623_262375


namespace isosceles_triangle_34_perimeter_l2623_262333

/-- An isosceles triangle with sides 3 and 4 -/
structure IsoscelesTriangle34 where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : (side1 = side2 ∧ side3 = 3) ∨ (side1 = side3 ∧ side2 = 3)
  has_side_4 : side1 = 4 ∨ side2 = 4 ∨ side3 = 4

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle34) : ℝ := t.side1 + t.side2 + t.side3

/-- Theorem: The perimeter of an isosceles triangle with sides 3 and 4 is either 10 or 11 -/
theorem isosceles_triangle_34_perimeter (t : IsoscelesTriangle34) : 
  perimeter t = 10 ∨ perimeter t = 11 := by
  sorry

end isosceles_triangle_34_perimeter_l2623_262333


namespace semicircle_area_with_inscribed_rectangle_l2623_262345

theorem semicircle_area_with_inscribed_rectangle (π : Real) :
  let rectangle_width : Real := 1
  let rectangle_length : Real := 3
  let diameter : Real := (rectangle_width ^ 2 + rectangle_length ^ 2).sqrt
  let radius : Real := diameter / 2
  let semicircle_area : Real := π * radius ^ 2 / 2
  semicircle_area = 13 * π / 8 := by
  sorry

end semicircle_area_with_inscribed_rectangle_l2623_262345


namespace unique_solution_of_equation_l2623_262397

theorem unique_solution_of_equation :
  ∃! x : ℝ, (x^16 + 1) * (x^12 + x^8 + x^4 + 1) = 18 * x^8 :=
by sorry

end unique_solution_of_equation_l2623_262397


namespace point_C_coordinates_l2623_262301

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)

-- Define the line that point C is on
def line_C (x y : ℝ) : Prop := 3 * x - y + 3 = 0

-- Define the area of triangle ABC
def triangle_area (C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem point_C_coordinates :
  ∀ C : ℝ × ℝ,
  line_C C.1 C.2 →
  triangle_area C = 10 →
  C = (-1, 0) ∨ C = (5/3, 8) :=
sorry

end point_C_coordinates_l2623_262301


namespace estimate_white_balls_l2623_262380

/-- Represents the contents of the box -/
structure Box where
  black : ℕ
  white : ℕ

/-- Represents the result of the drawing experiment -/
structure DrawResult where
  total : ℕ
  black : ℕ

/-- Calculates the expected number of white balls given the box contents and draw results -/
def expectedWhiteBalls (box : Box) (result : DrawResult) : ℚ :=
  (box.black : ℚ) * (result.total - result.black : ℚ) / result.black

/-- The main theorem statement -/
theorem estimate_white_balls (box : Box) (result : DrawResult) :
  box.black = 4 ∧ result.total = 40 ∧ result.black = 10 →
  expectedWhiteBalls box result = 12 := by
  sorry

end estimate_white_balls_l2623_262380


namespace cost_price_calculation_l2623_262368

def selling_price : ℝ := 24000

def discount_rate : ℝ := 0.1

def profit_rate : ℝ := 0.08

theorem cost_price_calculation (cp : ℝ) : 
  cp = 20000 ↔ 
  (selling_price * (1 - discount_rate) = cp * (1 + profit_rate)) ∧
  (selling_price > 0) ∧ 
  (cp > 0) :=
sorry

end cost_price_calculation_l2623_262368


namespace max_viewership_l2623_262313

structure Series where
  runtime : ℕ
  commercials : ℕ
  viewers : ℕ

def seriesA : Series := { runtime := 80, commercials := 1, viewers := 600000 }
def seriesB : Series := { runtime := 40, commercials := 1, viewers := 200000 }

def totalProgramTime : ℕ := 320
def minCommercials : ℕ := 6

def Schedule := ℕ × ℕ  -- (number of A episodes, number of B episodes)

def isValidSchedule (s : Schedule) : Prop :=
  s.1 * seriesA.runtime + s.2 * seriesB.runtime ≤ totalProgramTime ∧
  s.1 * seriesA.commercials + s.2 * seriesB.commercials ≥ minCommercials

def viewership (s : Schedule) : ℕ :=
  s.1 * seriesA.viewers + s.2 * seriesB.viewers

theorem max_viewership :
  ∃ (s : Schedule), isValidSchedule s ∧
    ∀ (s' : Schedule), isValidSchedule s' → viewership s' ≤ viewership s ∧
    viewership s = 2000000 :=
  sorry

end max_viewership_l2623_262313


namespace smallest_prime_after_seven_nonprimes_l2623_262387

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a number is the first prime after 7 consecutive non-primes
def isFirstPrimeAfter7NonPrimes (p : ℕ) : Prop :=
  isPrime p ∧
  ∀ k : ℕ, k ∈ Finset.range 7 → ¬isPrime (p - k - 1) ∧
  ∀ q : ℕ, q < p → isFirstPrimeAfter7NonPrimes q → False

-- State the theorem
theorem smallest_prime_after_seven_nonprimes :
  isFirstPrimeAfter7NonPrimes 97 := by sorry

end smallest_prime_after_seven_nonprimes_l2623_262387


namespace shaded_area_is_110_l2623_262371

/-- Represents a triangle inscribed in a hexagon --/
inductive InscribedTriangle
  | Small
  | Medium
  | Large

/-- The area of an inscribed triangle in terms of the number of unit triangles it contains --/
def triangle_area (t : InscribedTriangle) : ℕ :=
  match t with
  | InscribedTriangle.Small => 1
  | InscribedTriangle.Medium => 3
  | InscribedTriangle.Large => 7

/-- The area of a unit equilateral triangle in the hexagon --/
def unit_triangle_area : ℕ := 10

/-- The total area of the shaded part --/
def shaded_area : ℕ :=
  (triangle_area InscribedTriangle.Small +
   triangle_area InscribedTriangle.Medium +
   triangle_area InscribedTriangle.Large) * unit_triangle_area

theorem shaded_area_is_110 : shaded_area = 110 := by
  sorry


end shaded_area_is_110_l2623_262371


namespace remainder_problem_l2623_262396

theorem remainder_problem (k : ℕ) (h1 : k > 0) (h2 : k < 84) 
  (h3 : k % 5 = 2) (h4 : k % 6 = 5) (h5 : k % 8 = 7) : k % 9 = 2 := by
  sorry

end remainder_problem_l2623_262396


namespace equilateral_triangle_pq_l2623_262384

/-- Given an equilateral triangle with vertices at (0,0), (p, 13), and (q, 41),
    prove that the product pq equals -2123/3 -/
theorem equilateral_triangle_pq (p q : ℝ) : 
  (∃ (z : ℂ), z^3 = 1 ∧ z ≠ 1 ∧ z * (p + 13*I) = q + 41*I) →
  p * q = -2123/3 := by
  sorry

end equilateral_triangle_pq_l2623_262384


namespace inequality_proof_l2623_262399

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (x^4)/(y*(1-y^2)) + (y^4)/(z*(1-z^2)) + (z^4)/(x*(1-x^2)) ≥ 1/8 := by
  sorry

end inequality_proof_l2623_262399


namespace garbage_collection_difference_l2623_262319

theorem garbage_collection_difference (daliah_amount dewei_amount zane_amount : ℝ) : 
  daliah_amount = 17.5 →
  zane_amount = 62 →
  zane_amount = 4 * dewei_amount →
  dewei_amount < daliah_amount →
  daliah_amount - dewei_amount = 2 := by
sorry

end garbage_collection_difference_l2623_262319


namespace construction_materials_l2623_262334

theorem construction_materials (concrete stone total : Real) 
  (h1 : concrete = 0.17)
  (h2 : stone = 0.5)
  (h3 : total = 0.83) :
  total - (concrete + stone) = 0.16 := by
  sorry

end construction_materials_l2623_262334


namespace quadratic_integer_roots_l2623_262357

-- Define the quadratic equation
def quadratic_equation (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 12*a

-- Define a function to check if a number is an integer
def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

-- Define a function to count the number of real a values that satisfy the condition
def count_a_values : ℕ := sorry

-- Theorem statement
theorem quadratic_integer_roots :
  count_a_values = 15 := by sorry

end quadratic_integer_roots_l2623_262357


namespace house_selling_price_l2623_262353

def commission_rate : ℝ := 0.06
def commission_amount : ℝ := 8880

theorem house_selling_price :
  ∃ (selling_price : ℝ),
    selling_price * commission_rate = commission_amount ∧
    selling_price = 148000 := by
  sorry

end house_selling_price_l2623_262353


namespace max_consecutive_integers_sum_l2623_262342

theorem max_consecutive_integers_sum (n : ℕ) : n = 31 ↔ 
  (n ≥ 3 ∧ 
   (∀ k : ℕ, k ≥ 3 → k ≤ n → (k * (k + 1)) / 2 - 3 ≤ 500) ∧
   (∀ m : ℕ, m > n → (m * (m + 1)) / 2 - 3 > 500)) :=
by sorry

end max_consecutive_integers_sum_l2623_262342


namespace age_group_problem_l2623_262305

theorem age_group_problem (n : ℕ) (A : ℝ) : 
  (n + 1) * (A + 7) = n * A + 39 →
  (n + 1) * (A - 1) = n * A + 15 →
  n = 3 := by sorry

end age_group_problem_l2623_262305


namespace fourth_episode_duration_l2623_262315

theorem fourth_episode_duration (episode1 episode2 episode3 : ℕ) 
  (total_duration : ℕ) (h1 : episode1 = 58) (h2 : episode2 = 62) 
  (h3 : episode3 = 65) (h4 : total_duration = 4 * 60) : 
  total_duration - (episode1 + episode2 + episode3) = 55 := by
  sorry

end fourth_episode_duration_l2623_262315


namespace dean_taller_than_ron_l2623_262321

theorem dean_taller_than_ron (water_depth : ℝ) (ron_height : ℝ) (dean_height : ℝ) :
  water_depth = 255 ∧ ron_height = 13 ∧ water_depth = 15 * dean_height →
  dean_height - ron_height = 4 := by
  sorry

end dean_taller_than_ron_l2623_262321


namespace square_count_figure_100_l2623_262314

/-- Represents the number of squares in the nth figure -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

theorem square_count_figure_100 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 100 = 30301 := by
  sorry

end square_count_figure_100_l2623_262314


namespace area_of_shaded_region_l2623_262355

-- Define the side lengths of the squares
def side1 : ℝ := 6
def side2 : ℝ := 8

-- Define π as 3.14
def π : ℝ := 3.14

-- Define the area of the shaded region
def shaded_area : ℝ := 50.24

-- Theorem statement
theorem area_of_shaded_region :
  ∃ (f : ℝ → ℝ → ℝ → ℝ), f side1 side2 π = shaded_area :=
sorry

end area_of_shaded_region_l2623_262355


namespace product_nine_consecutive_divisible_by_ten_l2623_262326

theorem product_nine_consecutive_divisible_by_ten (n : ℕ) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7) * (n + 8)) = 10 * k :=
by sorry

end product_nine_consecutive_divisible_by_ten_l2623_262326


namespace area_of_twelve_sided_figure_l2623_262398

/-- A vertex is represented by its x and y coordinates -/
structure Vertex :=
  (x : ℝ)
  (y : ℝ)

/-- A polygon is represented by a list of vertices -/
def Polygon := List Vertex

/-- The vertices of our 12-sided figure -/
def twelveSidedFigure : Polygon := [
  ⟨1, 3⟩, ⟨2, 4⟩, ⟨2, 5⟩, ⟨3, 6⟩, ⟨4, 6⟩, ⟨5, 5⟩,
  ⟨6, 4⟩, ⟨6, 3⟩, ⟨5, 2⟩, ⟨4, 1⟩, ⟨3, 1⟩, ⟨2, 2⟩
]

/-- Function to calculate the area of a polygon -/
def areaOfPolygon (p : Polygon) : ℝ := sorry

/-- Theorem stating that the area of our 12-sided figure is 16 cm² -/
theorem area_of_twelve_sided_figure :
  areaOfPolygon twelveSidedFigure = 16 := by sorry

end area_of_twelve_sided_figure_l2623_262398


namespace log_like_function_72_l2623_262389

/-- A function satisfying f(ab) = f(a) + f(b) for all a and b -/
def LogLikeFunction (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a * b) = f a + f b

/-- Theorem: If f is a LogLikeFunction with f(2) = m and f(3) = n, then f(72) = 3m + 2n -/
theorem log_like_function_72 (f : ℝ → ℝ) (m n : ℝ) 
  (h_log_like : LogLikeFunction f) (h_2 : f 2 = m) (h_3 : f 3 = n) : 
  f 72 = 3 * m + 2 * n := by
sorry

end log_like_function_72_l2623_262389


namespace area_of_triangle_PQR_l2623_262354

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def circleP : Circle := { center := (0, 1), radius := 1 }
def circleQ : Circle := { center := (3, 2), radius := 2 }
def circleR : Circle := { center := (4, 3), radius := 3 }

-- Define the line l (implicitly defined by the tangent points)

-- Define the theorem
theorem area_of_triangle_PQR :
  let P := circleP.center
  let Q := circleQ.center
  let R := circleR.center
  let area := abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)
  area = Real.sqrt 6 - Real.sqrt 2 :=
sorry


end area_of_triangle_PQR_l2623_262354


namespace subcommittee_count_l2623_262340

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid subcommittees -/
def validSubcommittees (totalMembers teachers subcommitteeSize : ℕ) : ℕ :=
  choose totalMembers subcommitteeSize - choose (totalMembers - teachers) subcommitteeSize

theorem subcommittee_count :
  validSubcommittees 12 5 5 = 771 := by sorry

end subcommittee_count_l2623_262340


namespace chessboard_numbering_exists_l2623_262325

theorem chessboard_numbering_exists : 
  ∃ f : ℕ → ℕ → ℕ, 
    (∀ i j, i ∈ Finset.range 8 ∧ j ∈ Finset.range 8 → f i j ∈ Finset.range 64) ∧ 
    (∀ i j, i ∈ Finset.range 7 ∧ j ∈ Finset.range 7 → 
      (f i j + f (i+1) j + f i (j+1) + f (i+1) (j+1)) % 4 = 0) ∧
    (∀ n, n ∈ Finset.range 64 → ∃ i j, i ∈ Finset.range 8 ∧ j ∈ Finset.range 8 ∧ f i j = n + 1) :=
by
  sorry

end chessboard_numbering_exists_l2623_262325


namespace gcd_12m_18n_lower_bound_l2623_262395

theorem gcd_12m_18n_lower_bound (m n : ℕ+) (h : Nat.gcd m n = 18) :
  Nat.gcd (12 * m) (18 * n) ≥ 108 := by
  sorry

end gcd_12m_18n_lower_bound_l2623_262395


namespace prob_tails_heads_heads_l2623_262338

-- Define a coin flip as a type with two possible outcomes
inductive CoinFlip : Type
| Heads : CoinFlip
| Tails : CoinFlip

-- Define a sequence of three coin flips
def ThreeFlips := (CoinFlip × CoinFlip × CoinFlip)

-- Define the probability of getting tails on a single flip
def prob_tails : ℚ := 1 / 2

-- Define the desired outcome: Tails, Heads, Heads
def desired_outcome : ThreeFlips := (CoinFlip.Tails, CoinFlip.Heads, CoinFlip.Heads)

-- Theorem: The probability of getting the desired outcome is 1/8
theorem prob_tails_heads_heads : 
  (prob_tails * (1 - prob_tails) * (1 - prob_tails) : ℚ) = 1 / 8 := by
  sorry

end prob_tails_heads_heads_l2623_262338


namespace sum_in_base4_l2623_262390

/-- Represents a number in base 4 -/
def Base4 : Type := List (Fin 4)

/-- Addition of two Base4 numbers -/
def add_base4 : Base4 → Base4 → Base4 := sorry

/-- Conversion from a natural number to Base4 -/
def nat_to_base4 : ℕ → Base4 := sorry

/-- Conversion from Base4 to a natural number -/
def base4_to_nat : Base4 → ℕ := sorry

theorem sum_in_base4 :
  let a : Base4 := nat_to_base4 211
  let b : Base4 := nat_to_base4 332
  let c : Base4 := nat_to_base4 123
  let result : Base4 := nat_to_base4 1120
  add_base4 (add_base4 a b) c = result := by sorry

end sum_in_base4_l2623_262390


namespace min_tangent_length_l2623_262307

/-- The minimum length of a tangent from a point on y = x + 2 to (x-3)² + (y+1)² = 2 is 4 -/
theorem min_tangent_length : 
  let line := {p : ℝ × ℝ | p.2 = p.1 + 2}
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 + 1)^2 = 2}
  ∃ (min_length : ℝ), 
    (∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line → q ∈ circle → 
      ‖p - q‖ ≥ min_length) ∧
    (∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧ ‖p - q‖ = min_length) ∧
    min_length = 4 :=
by
  sorry

end min_tangent_length_l2623_262307


namespace exists_four_unacquainted_l2623_262377

/-- A type representing a person in the group -/
def Person : Type := Fin 10

/-- The acquaintance relation between people -/
def acquainted : Person → Person → Prop := sorry

theorem exists_four_unacquainted 
  (h1 : ∀ p : Person, ∃! (q r : Person), q ≠ r ∧ acquainted p q ∧ acquainted p r)
  (h2 : ∀ p q : Person, acquainted p q → acquainted q p) :
  ∃ (a b c d : Person), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ¬acquainted a b ∧ ¬acquainted a c ∧ ¬acquainted a d ∧
    ¬acquainted b c ∧ ¬acquainted b d ∧ ¬acquainted c d :=
sorry

end exists_four_unacquainted_l2623_262377


namespace squares_ending_in_76_l2623_262385

theorem squares_ending_in_76 : 
  {x : ℕ | x^2 % 100 = 76} = {24, 26, 74, 76} := by sorry

end squares_ending_in_76_l2623_262385


namespace inequality_system_solution_l2623_262356

theorem inequality_system_solution (p : ℝ) :
  (19 * p < 10 ∧ p > 1/2) ↔ (1/2 < p ∧ p < 10/19) := by
  sorry

end inequality_system_solution_l2623_262356


namespace strawberry_jam_money_l2623_262378

-- Define the given conditions
def betty_strawberries : ℕ := 16
def matthew_strawberries : ℕ := betty_strawberries + 20
def natalie_strawberries : ℕ := matthew_strawberries / 2
def strawberries_per_jar : ℕ := 7
def price_per_jar : ℕ := 4

-- Define the theorem
theorem strawberry_jam_money : 
  (betty_strawberries + matthew_strawberries + natalie_strawberries) / strawberries_per_jar * price_per_jar = 40 := by
  sorry

end strawberry_jam_money_l2623_262378


namespace black_car_speed_l2623_262365

/-- Proves that given the conditions of the car problem, the black car's speed is 50 mph -/
theorem black_car_speed (red_speed : ℝ) (initial_gap : ℝ) (overtake_time : ℝ) : ℝ :=
  let black_speed := (initial_gap + red_speed * overtake_time) / overtake_time
  by
    sorry

#check black_car_speed 40 30 3 = 50

end black_car_speed_l2623_262365


namespace line_parallel_value_l2623_262317

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + (a^2 - 1) = 0

-- Define the parallel condition for two lines
def parallel (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) : Prop :=
  (A₁ * B₂ - A₂ * B₁ = 0) ∧ ((A₁ * C₂ - A₂ * C₁ ≠ 0) ∨ (B₁ * C₂ - B₂ * C₁ ≠ 0))

-- Define the coincident condition for two lines
def coincident (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) : Prop :=
  (A₁ * B₂ - A₂ * B₁ = 0) ∧ (A₁ * C₂ - A₂ * C₁ = 0) ∧ (B₁ * C₂ - B₂ * C₁ = 0)

-- Theorem statement
theorem line_parallel_value (a : ℝ) : 
  (parallel a 2 6 1 (a-1) (a^2-1)) ∧ 
  ¬(coincident a 2 6 1 (a-1) (a^2-1)) → 
  a = -1 := by sorry

end line_parallel_value_l2623_262317


namespace student_arrangement_l2623_262360

theorem student_arrangement (n : ℕ) (k : ℕ) (m : ℕ) : 
  n = 7 → k = 6 → m = 3 → 
  (n.choose k) * (k.choose m) * ((k - m).choose m) = 140 :=
sorry

end student_arrangement_l2623_262360


namespace triangle_equilateral_l2623_262359

theorem triangle_equilateral (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^4 = b^4 + c^4 - b^2*c^2)
  (h5 : b^4 = c^4 + a^4 - a^2*c^2) :
  a = b ∧ b = c := by
  sorry

end triangle_equilateral_l2623_262359


namespace circle_passes_through_points_circle_equation_equivalence_l2623_262337

-- Define the circle passing through points O(0,0), A(1,1), and B(4,2)
def circle_through_points (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

-- Define the standard form of the circle
def circle_standard_form (x y : ℝ) : Prop :=
  (x - 4)^2 + (y + 3)^2 = 25

-- Theorem stating that the circle passes through the given points
theorem circle_passes_through_points :
  circle_through_points 0 0 ∧
  circle_through_points 1 1 ∧
  circle_through_points 4 2 := by sorry

-- Theorem stating the equivalence of the general and standard forms
theorem circle_equation_equivalence :
  ∀ x y : ℝ, circle_through_points x y ↔ circle_standard_form x y := by sorry

end circle_passes_through_points_circle_equation_equivalence_l2623_262337


namespace susan_works_four_days_per_week_l2623_262383

/-- Represents Susan's work schedule and vacation details -/
structure WorkSchedule where
  hourlyRate : ℚ
  hoursPerDay : ℕ
  vacationDays : ℕ
  paidVacationDays : ℕ
  missedPay : ℚ

/-- Calculates the number of days Susan works per week -/
def daysWorkedPerWeek (schedule : WorkSchedule) : ℚ :=
  let totalVacationDays := 2 * 7
  let unpaidVacationDays := totalVacationDays - schedule.paidVacationDays
  unpaidVacationDays / 2

/-- Theorem stating that Susan works 4 days a week -/
theorem susan_works_four_days_per_week (schedule : WorkSchedule)
  (h1 : schedule.hourlyRate = 15)
  (h2 : schedule.hoursPerDay = 8)
  (h3 : schedule.vacationDays = 14)
  (h4 : schedule.paidVacationDays = 6)
  (h5 : schedule.missedPay = 480) :
  daysWorkedPerWeek schedule = 4 := by
  sorry


end susan_works_four_days_per_week_l2623_262383


namespace cellphone_selection_theorem_l2623_262351

/-- The number of service providers -/
def total_providers : ℕ := 25

/-- The number of siblings (including Laura) -/
def num_siblings : ℕ := 4

/-- The number of ways to select providers for all siblings -/
def ways_to_select_providers : ℕ := 
  (total_providers - 1) * (total_providers - 2) * (total_providers - 3)

theorem cellphone_selection_theorem :
  ways_to_select_providers = 12144 := by
  sorry

end cellphone_selection_theorem_l2623_262351


namespace total_spent_is_450_l2623_262310

/-- The total amount spent by Leonard and Michael on gifts for their father -/
def total_spent (leonard_wallet : ℕ) (leonard_sneakers : ℕ) (leonard_sneakers_count : ℕ)
                (michael_backpack : ℕ) (michael_jeans : ℕ) (michael_jeans_count : ℕ) : ℕ :=
  leonard_wallet + leonard_sneakers * leonard_sneakers_count +
  michael_backpack + michael_jeans * michael_jeans_count

/-- Theorem stating that the total amount spent by Leonard and Michael is $450 -/
theorem total_spent_is_450 :
  total_spent 50 100 2 100 50 2 = 450 := by
  sorry


end total_spent_is_450_l2623_262310


namespace book_arrangements_eq_34560_l2623_262312

/-- The number of ways to arrange 11 books (3 Arabic, 2 English, 4 Spanish, and 2 French) on a shelf,
    keeping Arabic, Spanish, and English books together respectively. -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 11
  let arabic_books : ℕ := 3
  let english_books : ℕ := 2
  let spanish_books : ℕ := 4
  let french_books : ℕ := 2
  let group_arrangements : ℕ := Nat.factorial 5
  let arabic_internal_arrangements : ℕ := Nat.factorial arabic_books
  let english_internal_arrangements : ℕ := Nat.factorial english_books
  let spanish_internal_arrangements : ℕ := Nat.factorial spanish_books
  group_arrangements * arabic_internal_arrangements * english_internal_arrangements * spanish_internal_arrangements

theorem book_arrangements_eq_34560 : book_arrangements = 34560 := by
  sorry

end book_arrangements_eq_34560_l2623_262312


namespace cafeteria_pies_correct_l2623_262381

def cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

theorem cafeteria_pies_correct : cafeteria_pies 47 27 4 = 5 := by
  sorry

end cafeteria_pies_correct_l2623_262381


namespace caleb_dandelion_puffs_l2623_262309

/-- Represents the problem of Caleb's dandelion puffs distribution --/
def dandelion_puffs_problem (total : ℕ) (sister grandmother dog : ℕ) (friends num_per_friend : ℕ) : Prop :=
  ∃ (mom : ℕ),
    total = mom + sister + grandmother + dog + (friends * num_per_friend) ∧
    total = 40 ∧
    sister = 3 ∧
    grandmother = 5 ∧
    dog = 2 ∧
    friends = 3 ∧
    num_per_friend = 9

/-- The solution to Caleb's dandelion puffs problem --/
theorem caleb_dandelion_puffs :
  dandelion_puffs_problem 40 3 5 2 3 9 → ∃ (mom : ℕ), mom = 3 := by
  sorry

end caleb_dandelion_puffs_l2623_262309


namespace factory_production_l2623_262349

/-- Calculates the total television production in the second year given the daily production rate in the first year and the reduction percentage. -/
def secondYearProduction (dailyRate : ℕ) (reductionPercent : ℕ) : ℕ :=
  let firstYearTotal := dailyRate * 365
  let reduction := firstYearTotal * reductionPercent / 100
  firstYearTotal - reduction

/-- Theorem stating that for a factory producing 10 televisions per day in the first year
    and reducing production by 10% in the second year, the total production in the second year is 3285. -/
theorem factory_production :
  secondYearProduction 10 10 = 3285 := by
  sorry

end factory_production_l2623_262349


namespace bake_sale_group_composition_l2623_262370

theorem bake_sale_group_composition (total : ℕ) (girls : ℕ) : 
  girls = (60 : ℕ) * total / 100 →
  (girls - 3 : ℕ) * 2 = total →
  girls = 18 :=
by
  sorry

end bake_sale_group_composition_l2623_262370


namespace quadratic_properties_l2623_262329

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 12 * x + 10

-- State the theorem
theorem quadratic_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 0 = 10 := by
  sorry

end quadratic_properties_l2623_262329


namespace ngo_employees_l2623_262327

/-- The number of illiterate employees -/
def num_illiterate : ℕ := 20

/-- The decrease in daily average wages of illiterate employees in Rs -/
def wage_decrease_illiterate : ℕ := 15

/-- The decrease in average salary of all employees in Rs per day -/
def avg_salary_decrease : ℕ := 10

/-- The number of literate employees -/
def num_literate : ℕ := 10

theorem ngo_employees :
  num_literate = 10 :=
by sorry

end ngo_employees_l2623_262327


namespace smallest_constant_inequality_l2623_262302

theorem smallest_constant_inequality (D : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2 ≥ D * (x - y)) ↔ D ≥ -2 :=
by sorry

end smallest_constant_inequality_l2623_262302


namespace teena_speed_is_55_l2623_262372

-- Define the given conditions
def yoe_speed : ℝ := 40
def initial_distance : ℝ := 7.5
def final_relative_distance : ℝ := 15
def time : ℝ := 1.5  -- 90 minutes in hours

-- Define Teena's speed as a variable
def teena_speed : ℝ := 55

-- Theorem statement
theorem teena_speed_is_55 :
  yoe_speed * time + initial_distance + final_relative_distance = teena_speed * time :=
by sorry

end teena_speed_is_55_l2623_262372


namespace least_n_with_k_ge_10_M_mod_500_l2623_262382

/-- Sum of digits in base 6 representation -/
def h (n : ℕ) : ℕ := sorry

/-- Sum of digits in base 10 representation of h(n) -/
def j (n : ℕ) : ℕ := sorry

/-- Sum of squares of digits in base 12 representation of j(n) -/
def k (n : ℕ) : ℕ := sorry

/-- The least value of n such that k(n) ≥ 10 -/
def M : ℕ := sorry

theorem least_n_with_k_ge_10 : M = 31 := by sorry

theorem M_mod_500 : M % 500 = 31 := by sorry

end least_n_with_k_ge_10_M_mod_500_l2623_262382


namespace smallest_addition_for_divisibility_l2623_262369

theorem smallest_addition_for_divisibility : ∃! x : ℕ, 
  (x ≤ y → ∀ y : ℕ, (758492136547 + y) % 17 = 0 ∧ (758492136547 + y) % 3 = 0) ∧
  (758492136547 + x) % 17 = 0 ∧ 
  (758492136547 + x) % 3 = 0 := by
  sorry

end smallest_addition_for_divisibility_l2623_262369


namespace prime_divides_power_difference_l2623_262367

theorem prime_divides_power_difference (p : ℕ) (n : ℕ) (hp : Nat.Prime p) :
  p ∣ (3^(n+p) - 3^(n+1)) := by
  sorry

end prime_divides_power_difference_l2623_262367


namespace accepted_to_rejected_ratio_egg_processing_change_l2623_262363

/-- Represents the daily egg processing at a plant -/
structure EggProcessing where
  total : ℕ
  accepted : ℕ
  rejected : ℕ
  h_total : total = accepted + rejected

/-- The original egg processing scenario -/
def original : EggProcessing :=
  { total := 400,
    accepted := 384,
    rejected := 16,
    h_total := rfl }

/-- The modified egg processing scenario -/
def modified : EggProcessing :=
  { total := 400,
    accepted := 396,
    rejected := 4,
    h_total := rfl }

/-- Theorem stating the ratio of accepted to rejected eggs in the modified scenario -/
theorem accepted_to_rejected_ratio :
  modified.accepted / modified.rejected = 99 := by
  sorry

/-- Proof that the ratio of accepted to rejected eggs changes as described -/
theorem egg_processing_change (orig : EggProcessing) (mod : EggProcessing)
  (h_orig : orig = original)
  (h_mod : mod = modified)
  (h_total_unchanged : orig.total = mod.total)
  (h_accepted_increase : mod.accepted = orig.accepted + 12) :
  mod.accepted / mod.rejected = 99 := by
  sorry

end accepted_to_rejected_ratio_egg_processing_change_l2623_262363
