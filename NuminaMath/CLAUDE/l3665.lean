import Mathlib

namespace constant_term_expansion_l3665_366571

theorem constant_term_expansion (n : ℕ+) : 
  (∃ r : ℕ, r = 6 ∧ 3*n - 4*r = 0) → n = 8 := by
  sorry

end constant_term_expansion_l3665_366571


namespace distinct_projections_exist_l3665_366519

/-- Represents a student's marks as a point in 12-dimensional space -/
def Student := Fin 12 → ℝ

/-- The set of 7 students -/
def Students := Fin 7 → Student

theorem distinct_projections_exist (students : Students) 
  (h : ∀ i j, i ≠ j → students i ≠ students j) :
  ∃ (subjects : Fin 6 → Fin 12), 
    ∀ i j, i ≠ j → 
      ∃ k, (students i (subjects k)) ≠ (students j (subjects k)) := by
  sorry

end distinct_projections_exist_l3665_366519


namespace birds_in_tree_l3665_366528

theorem birds_in_tree (initial_birds : ℕ) (new_birds : ℕ) (total_birds : ℕ) : 
  initial_birds = 14 → new_birds = 21 → total_birds = initial_birds + new_birds → total_birds = 35 := by
  sorry

end birds_in_tree_l3665_366528


namespace hypotenuse_length_l3665_366579

/-- Given a right-angled triangle with sides a, b, and c (hypotenuse),
    where the sum of squares of all sides is 2500,
    prove that the length of the hypotenuse is 25√2. -/
theorem hypotenuse_length (a b c : ℝ) 
  (right_angle : a^2 + b^2 = c^2)  -- right-angled triangle condition
  (sum_squares : a^2 + b^2 + c^2 = 2500)  -- sum of squares condition
  : c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l3665_366579


namespace cross_section_area_fraction_l3665_366554

theorem cross_section_area_fraction (r : ℝ) (r_pos : r > 0) : 
  let sphere_surface_area := 4 * Real.pi * r^2
  let cross_section_radius := r / 2
  let cross_section_area := Real.pi * cross_section_radius^2
  cross_section_area / sphere_surface_area = 1 / 4 := by
  sorry

end cross_section_area_fraction_l3665_366554


namespace ratio_and_mean_problem_l3665_366541

theorem ratio_and_mean_problem (a b c : ℕ+) (h_ratio : (a : ℚ) / b = 2 / 3 ∧ (b : ℚ) / c = 3 / 4)
  (h_mean : (a + b + c : ℚ) / 3 = 42) : a = 28 := by
  sorry

end ratio_and_mean_problem_l3665_366541


namespace trajectory_equation_l3665_366539

-- Define the point M
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition for point M
def satisfiesCondition (M : Point) : Prop :=
  Real.sqrt ((M.y + 5)^2 + M.x^2) - Real.sqrt ((M.y - 5)^2 + M.x^2) = 8

-- Define the trajectory equation
def isOnTrajectory (M : Point) : Prop :=
  M.y^2 / 16 - M.x^2 / 9 = 1 ∧ M.y > 0

-- Theorem statement
theorem trajectory_equation (M : Point) :
  satisfiesCondition M → isOnTrajectory M :=
by
  sorry

end trajectory_equation_l3665_366539


namespace parcel_boxes_count_l3665_366555

/-- Represents the position of a parcel in a rectangular arrangement of boxes -/
structure ParcelPosition where
  left : Nat
  right : Nat
  front : Nat
  back : Nat

/-- Calculates the total number of parcel boxes given the position of a specific parcel -/
def totalParcelBoxes (pos : ParcelPosition) : Nat :=
  (pos.left + pos.right - 1) * (pos.front + pos.back - 1)

/-- Theorem stating that given the specific parcel position, the total number of boxes is 399 -/
theorem parcel_boxes_count (pos : ParcelPosition) 
  (h_left : pos.left = 7)
  (h_right : pos.right = 13)
  (h_front : pos.front = 8)
  (h_back : pos.back = 14) : 
  totalParcelBoxes pos = 399 := by
  sorry

#eval totalParcelBoxes ⟨7, 13, 8, 14⟩

end parcel_boxes_count_l3665_366555


namespace cubic_sum_minus_product_l3665_366531

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 13) 
  (sum_prod_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1027 := by
  sorry

end cubic_sum_minus_product_l3665_366531


namespace nursing_home_medicine_boxes_l3665_366585

/-- The total number of boxes of medicine received by the nursing home -/
def total_boxes (vitamin_boxes supplement_boxes : ℕ) : ℕ :=
  vitamin_boxes + supplement_boxes

/-- Theorem stating that the nursing home received 760 boxes of medicine -/
theorem nursing_home_medicine_boxes : 
  total_boxes 472 288 = 760 := by
  sorry

end nursing_home_medicine_boxes_l3665_366585


namespace restaurant_bill_theorem_l3665_366586

/-- Calculates the total amount to pay after applying discounts to two bills -/
def total_amount_after_discount (bill1 bill2 discount1 discount2 : ℚ) : ℚ :=
  (bill1 * (1 - discount1 / 100)) + (bill2 * (1 - discount2 / 100))

/-- Theorem stating that the total amount Bob and Kate pay after discounts is $53 -/
theorem restaurant_bill_theorem :
  total_amount_after_discount 30 25 5 2 = 53 := by
  sorry

end restaurant_bill_theorem_l3665_366586


namespace number_puzzle_l3665_366563

theorem number_puzzle (x y : ℝ) : x = 33 → (x / 4) + y = 15 → y = 6.75 := by
  sorry

end number_puzzle_l3665_366563


namespace length_AB_l3665_366533

/-- Parabola C: y^2 = 8x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 8*x

/-- Line l: y = (√3/3)(x-2) -/
def line_l (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - 2)

/-- A and B are intersection points of C and l -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola_C A.1 A.2 ∧ line_l A.1 A.2 ∧
  parabola_C B.1 B.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

/-- The length of AB is 32 -/
theorem length_AB (A B : ℝ × ℝ) (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 32 := by sorry

end length_AB_l3665_366533


namespace plane_perpendicularity_condition_l3665_366551

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the properties and relations
variable (subset : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity_condition 
  (α β : Plane) (a : Line) 
  (h_subset : subset a α) :
  (∀ (a : Line), subset a α → perpendicular_line_plane a β → perpendicular_plane_plane α β) ∧ 
  (∃ (a : Line), subset a α ∧ perpendicular_plane_plane α β ∧ ¬perpendicular_line_plane a β) :=
sorry

end plane_perpendicularity_condition_l3665_366551


namespace prom_couples_count_l3665_366589

theorem prom_couples_count (total_students : ℕ) (solo_students : ℕ) (couples : ℕ) : 
  total_students = 123 → 
  solo_students = 3 → 
  couples = (total_students - solo_students) / 2 → 
  couples = 60 := by
  sorry

end prom_couples_count_l3665_366589


namespace total_mascots_is_16x_l3665_366596

/-- Represents the number of mascots Jina has -/
structure Mascots where
  x : ℕ  -- number of teddies
  y : ℕ  -- number of bunnies
  z : ℕ  -- number of koalas

/-- Calculates the total number of mascots after Jina's mom gives her more teddies -/
def totalMascots (m : Mascots) : ℕ :=
  let x_new := m.x + 2 * m.y
  x_new + m.y + m.z

/-- Theorem stating the total number of mascots is 16 times the original number of teddies -/
theorem total_mascots_is_16x (m : Mascots)
    (h1 : m.y = 3 * m.x)  -- Jina has 3 times more bunnies than teddies
    (h2 : m.z = 2 * m.y)  -- Jina has twice the number of koalas as she has bunnies
    : totalMascots m = 16 * m.x := by
  sorry

#check total_mascots_is_16x

end total_mascots_is_16x_l3665_366596


namespace cubic_root_problem_l3665_366545

/-- A monic cubic polynomial -/
def MonicCubic (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

theorem cubic_root_problem (r : ℝ) (f g : ℝ → ℝ) 
    (hf : ∃ a b c, f = MonicCubic a b c)
    (hg : ∃ a b c, g = MonicCubic a b c)
    (hf_roots : f (r + 2) = 0 ∧ f (r + 4) = 0)
    (hg_roots : g (r + 3) = 0 ∧ g (r + 5) = 0)
    (h_diff : ∀ x, f x - g x = 2*r + 1) :
  r = 1/4 := by
  sorry

end cubic_root_problem_l3665_366545


namespace carnation_bouquet_combinations_l3665_366564

def distribute_carnations (total : ℕ) (types : ℕ) (extras : ℕ) : ℕ :=
  Nat.choose (extras + types - 1) (types - 1)

theorem carnation_bouquet_combinations :
  distribute_carnations 5 3 2 = 6 := by
  sorry

end carnation_bouquet_combinations_l3665_366564


namespace largest_number_l3665_366559

theorem largest_number : ∀ (a b c : ℝ), 
  a = -12.4 → b = -1.23 → c = -0.13 → 
  (0 ≥ a) ∧ (0 ≥ b) ∧ (0 ≥ c) ∧ (0 ≥ 0) :=
by
  sorry

#check largest_number

end largest_number_l3665_366559


namespace new_average_after_dropping_lowest_l3665_366549

def calculate_new_average (num_tests : ℕ) (original_average : ℚ) (lowest_score : ℚ) : ℚ :=
  ((num_tests : ℚ) * original_average - lowest_score) / ((num_tests : ℚ) - 1)

theorem new_average_after_dropping_lowest
  (num_tests : ℕ)
  (original_average : ℚ)
  (lowest_score : ℚ)
  (h1 : num_tests = 4)
  (h2 : original_average = 35)
  (h3 : lowest_score = 20) :
  calculate_new_average num_tests original_average lowest_score = 40 :=
by
  sorry

end new_average_after_dropping_lowest_l3665_366549


namespace paint_mixture_ratio_l3665_366536

/-- Given a paint mixture ratio of 3:2:4 for blue:green:white paint, 
    if 12 quarts of white paint are used, then 6 quarts of green paint should be used. -/
theorem paint_mixture_ratio (blue green white : ℚ) : 
  blue / green = 3 / 2 ∧ 
  green / white = 2 / 4 ∧ 
  white = 12 → 
  green = 6 := by
sorry

end paint_mixture_ratio_l3665_366536


namespace power_multiplication_l3665_366512

theorem power_multiplication (m : ℝ) : (m^2)^3 * m^4 = m^10 := by
  sorry

end power_multiplication_l3665_366512


namespace mary_nickels_problem_l3665_366509

/-- The number of nickels Mary's dad gave her -/
def nickels_from_dad (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

theorem mary_nickels_problem :
  let initial_nickels : ℕ := 7
  let final_nickels : ℕ := 12
  nickels_from_dad initial_nickels final_nickels = 5 := by
  sorry

end mary_nickels_problem_l3665_366509


namespace simplify_expression_l3665_366566

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^2 + y^2)⁻¹ * (x⁻¹ + y⁻¹) = (x^3*y + x*y^3)⁻¹ * (x + y) :=
by sorry

end simplify_expression_l3665_366566


namespace second_caterer_cheaper_at_41_l3665_366573

/-- Represents the pricing model of a caterer -/
structure CatererPricing where
  basicFee : ℕ
  pricePerPerson : ℕ → ℕ

/-- The pricing model for the first caterer -/
def firstCaterer : CatererPricing :=
  { basicFee := 150,
    pricePerPerson := fun _ => 17 }

/-- The pricing model for the second caterer -/
def secondCaterer : CatererPricing :=
  { basicFee := 250,
    pricePerPerson := fun x => if x ≤ 40 then 15 else 13 }

/-- Calculate the total price for a caterer given the number of people -/
def totalPrice (c : CatererPricing) (people : ℕ) : ℕ :=
  c.basicFee + c.pricePerPerson people * people

/-- The theorem stating that 41 is the least number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper_at_41 :
  (∀ n < 41, totalPrice firstCaterer n ≤ totalPrice secondCaterer n) ∧
  (totalPrice secondCaterer 41 < totalPrice firstCaterer 41) :=
sorry

end second_caterer_cheaper_at_41_l3665_366573


namespace binomial_18_6_l3665_366516

theorem binomial_18_6 : Nat.choose 18 6 = 13260 := by
  sorry

end binomial_18_6_l3665_366516


namespace number_manipulation_l3665_366574

theorem number_manipulation (x : ℝ) : (x - 5) / 7 = 7 → (x - 6) / 8 = 6 := by
  sorry

end number_manipulation_l3665_366574


namespace suraj_innings_l3665_366591

/-- 
Proves that the number of innings Suraj played before the last one is 16,
given the conditions of the problem.
-/
theorem suraj_innings : 
  ∀ (n : ℕ) (A : ℚ),
  (A + 4 = 28) →                             -- New average after increase
  (n * A + 92 = (n + 1) * 28) →              -- Total runs equation
  (n = 16) := by
sorry

end suraj_innings_l3665_366591


namespace boyds_boy_friends_percentage_l3665_366556

theorem boyds_boy_friends_percentage 
  (julian_total_friends : ℕ)
  (julian_boys_percentage : ℚ)
  (julian_girls_percentage : ℚ)
  (boyd_total_friends : ℕ)
  (h1 : julian_total_friends = 80)
  (h2 : julian_boys_percentage = 60 / 100)
  (h3 : julian_girls_percentage = 40 / 100)
  (h4 : julian_boys_percentage + julian_girls_percentage = 1)
  (h5 : boyd_total_friends = 100)
  (h6 : (julian_girls_percentage * julian_total_friends : ℚ) * 2 = boyd_total_friends - (boyd_total_friends - (julian_girls_percentage * julian_total_friends : ℚ) * 2)) :
  (boyd_total_friends - (julian_girls_percentage * julian_total_friends : ℚ) * 2) / boyd_total_friends = 36 / 100 := by
  sorry

end boyds_boy_friends_percentage_l3665_366556


namespace fifteen_ways_to_assign_teachers_l3665_366582

/-- The number of ways to assign teachers to classes -/
def assign_teachers (n_teachers : ℕ) (n_classes : ℕ) (classes_per_teacher : ℕ) : ℕ :=
  (Nat.choose n_classes classes_per_teacher * 
   Nat.choose (n_classes - classes_per_teacher) classes_per_teacher * 
   Nat.choose (n_classes - 2 * classes_per_teacher) classes_per_teacher) / 
  Nat.factorial n_teachers

/-- Theorem stating that there are 15 ways to assign 3 teachers to 6 classes -/
theorem fifteen_ways_to_assign_teachers : 
  assign_teachers 3 6 2 = 15 := by
  sorry

end fifteen_ways_to_assign_teachers_l3665_366582


namespace unique_k_solution_l3665_366588

theorem unique_k_solution (k : ℤ) : 
  (∀ (a b c : ℝ), (a + b + c) * (a * b + b * c + c * a) + k * a * b * c = (a + b) * (b + c) * (c + a)) ↔ 
  k = -1 := by
sorry

end unique_k_solution_l3665_366588


namespace complex_equation_solution_l3665_366538

theorem complex_equation_solution :
  ∀ z : ℂ, (Complex.I - 1) * z = 2 → z = -1 - Complex.I := by
  sorry

end complex_equation_solution_l3665_366538


namespace pascal_triangle_interior_sum_l3665_366599

/-- Represents the sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The sum of interior numbers in the sixth row of Pascal's Triangle -/
def sixth_row_sum : ℕ := 30

/-- Theorem: If the sum of interior numbers in the sixth row of Pascal's Triangle is 30,
    then the sum of interior numbers in the eighth row is 126 -/
theorem pascal_triangle_interior_sum :
  interior_sum 6 = sixth_row_sum → interior_sum 8 = 126 := by
  sorry

end pascal_triangle_interior_sum_l3665_366599


namespace michael_fish_count_l3665_366520

theorem michael_fish_count (initial_fish : ℕ) (given_fish : ℕ) : 
  initial_fish = 31 → given_fish = 18 → initial_fish + given_fish = 49 := by
sorry

end michael_fish_count_l3665_366520


namespace quadratic_equation_root_difference_l3665_366526

theorem quadratic_equation_root_difference (k : ℝ) : 
  (∃ a b : ℝ, 3 * a^2 + 2 * a + k = 0 ∧ 
              3 * b^2 + 2 * b + k = 0 ∧ 
              |a - b| = (a^2 + b^2).sqrt) ↔ 
  (k = 0 ∨ k = -4/15) :=
sorry

end quadratic_equation_root_difference_l3665_366526


namespace second_class_size_l3665_366584

theorem second_class_size (first_class_size : ℕ) (first_class_avg : ℚ) 
  (second_class_avg : ℚ) (total_avg : ℚ) :
  first_class_size = 30 →
  first_class_avg = 40 →
  second_class_avg = 80 →
  total_avg = 65 →
  ∃ (second_class_size : ℕ),
    (first_class_size * first_class_avg + second_class_size * second_class_avg) / 
    (first_class_size + second_class_size) = total_avg ∧
    second_class_size = 50 := by
  sorry

end second_class_size_l3665_366584


namespace sams_balloons_l3665_366565

theorem sams_balloons (fred_balloons : ℕ) (mary_balloons : ℕ) (total_balloons : ℕ) 
  (h1 : fred_balloons = 5)
  (h2 : mary_balloons = 7)
  (h3 : total_balloons = 18) :
  total_balloons - fred_balloons - mary_balloons = 6 :=
by sorry

end sams_balloons_l3665_366565


namespace systems_solutions_l3665_366593

-- Define the systems of equations
def system1 (x y : ℝ) : Prop :=
  y = 2 * x - 5 ∧ 3 * x + 4 * y = 2

def system2 (x y : ℝ) : Prop :=
  3 * x - y = 8 ∧ (y - 1) / 3 = (x + 5) / 5

-- State the theorem
theorem systems_solutions :
  (∃ (x y : ℝ), system1 x y ∧ x = 2 ∧ y = -1) ∧
  (∃ (x y : ℝ), system2 x y ∧ x = 5 ∧ y = 7) := by
  sorry

end systems_solutions_l3665_366593


namespace train_speed_calculation_l3665_366560

-- Define the given constants
def train_length : ℝ := 160
def bridge_length : ℝ := 215
def crossing_time : ℝ := 30

-- Define the speed conversion factor
def m_per_s_to_km_per_hr : ℝ := 3.6

-- Theorem statement
theorem train_speed_calculation :
  let total_distance := train_length + bridge_length
  let speed_m_per_s := total_distance / crossing_time
  let speed_km_per_hr := speed_m_per_s * m_per_s_to_km_per_hr
  speed_km_per_hr = 45 := by sorry

end train_speed_calculation_l3665_366560


namespace high_school_total_students_l3665_366587

/-- Proves that the total number of students in a high school is 1800 given specific sampling conditions --/
theorem high_school_total_students
  (first_grade_students : ℕ)
  (total_sample_size : ℕ)
  (second_grade_sample : ℕ)
  (third_grade_sample : ℕ)
  (h1 : first_grade_students = 600)
  (h2 : total_sample_size = 45)
  (h3 : second_grade_sample = 20)
  (h4 : third_grade_sample = 10)
  (h5 : ∃ (total_students : ℕ), 
    (total_sample_size : ℚ) / total_students = 
    ((total_sample_size - second_grade_sample - third_grade_sample) : ℚ) / first_grade_students) :
  ∃ (total_students : ℕ), total_students = 1800 :=
by
  sorry

end high_school_total_students_l3665_366587


namespace cos_is_periodic_l3665_366576

-- Define the concept of a periodic function
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

-- Define the concept of a trigonometric function
def IsTrigonometric (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, f = fun x ↦ a * Real.cos (b * x) + c * Real.sin (b * x)

-- State the theorem
theorem cos_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric (fun x ↦ Real.cos x) →
  IsPeriodic (fun x ↦ Real.cos x) :=
by
  sorry

end cos_is_periodic_l3665_366576


namespace unknown_number_proof_l3665_366570

theorem unknown_number_proof (n : ℝ) (h : (12 : ℝ) * n^4 / 432 = 36) : n = 6 := by
  sorry

end unknown_number_proof_l3665_366570


namespace range_of_a_l3665_366595

theorem range_of_a (x a : ℝ) : 
  (∀ x, x - a ≥ 1 → x ≥ 1) ∧ 
  (1 - a ≥ 1) ∧ 
  ¬(-1 - a ≥ 1) → 
  -2 < a ∧ a ≤ 0 := by
sorry

end range_of_a_l3665_366595


namespace negative_sqrt_six_squared_equals_six_l3665_366513

theorem negative_sqrt_six_squared_equals_six : (-Real.sqrt 6)^2 = 6 := by
  sorry

end negative_sqrt_six_squared_equals_six_l3665_366513


namespace negation_of_universal_proposition_l3665_366500

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → Real.log x - x + 1 ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ Real.log x - x + 1 > 0) :=
by sorry

end negation_of_universal_proposition_l3665_366500


namespace greatest_integer_a_l3665_366568

theorem greatest_integer_a : ∀ a : ℤ,
  (∃ x : ℤ, (x - a) * (x - 7) + 3 = 0) →
  a ≤ 11 :=
by sorry

end greatest_integer_a_l3665_366568


namespace triangle_identity_l3665_366508

/-- For any triangle with sides a, b, c, circumradius R, and altitude CH from vertex C to side AB,
    the identity (a² + b² - c²) / (ab) = CH / R holds. -/
theorem triangle_identity (a b c R CH : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hR : R > 0) (hCH : CH > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + b^2 - c^2) / (a * b) = CH / R := by
  sorry

end triangle_identity_l3665_366508


namespace euler_number_proof_l3665_366558

def gauss_number : ℂ := Complex.mk 6 4

theorem euler_number_proof (product : ℂ) (h1 : product = Complex.mk 48 (-18)) :
  ∃ (euler_number : ℂ), euler_number * gauss_number = product ∧ euler_number = Complex.mk 4 (-6) := by
  sorry

end euler_number_proof_l3665_366558


namespace seating_arrangement_solution_l3665_366583

/-- A seating arrangement with rows of 7 or 9 people -/
structure SeatingArrangement where
  rows_of_9 : ℕ
  rows_of_7 : ℕ

/-- The total number of people seated -/
def total_seated (s : SeatingArrangement) : ℕ :=
  9 * s.rows_of_9 + 7 * s.rows_of_7

/-- The seating arrangement is valid if it seats exactly 112 people -/
def is_valid (s : SeatingArrangement) : Prop :=
  total_seated s = 112

theorem seating_arrangement_solution :
  ∃ (s : SeatingArrangement), is_valid s ∧ s.rows_of_9 = 7 :=
sorry

end seating_arrangement_solution_l3665_366583


namespace count_wings_l3665_366514

/-- The number of planes in the air exhibition -/
def num_planes : ℕ := 54

/-- The number of wings per plane -/
def wings_per_plane : ℕ := 2

/-- The total number of wings counted -/
def total_wings : ℕ := num_planes * wings_per_plane

theorem count_wings : total_wings = 108 := by
  sorry

end count_wings_l3665_366514


namespace tan_theta_minus_pi_over_four_l3665_366581

theorem tan_theta_minus_pi_over_four (θ : ℝ) (h : Real.cos θ - 3 * Real.sin θ = 0) :
  Real.tan (θ - π/4) = -1/2 := by sorry

end tan_theta_minus_pi_over_four_l3665_366581


namespace imaginary_part_of_complex_fraction_l3665_366518

theorem imaginary_part_of_complex_fraction : Complex.im (4 * I / (1 - I)) = 2 := by
  sorry

end imaginary_part_of_complex_fraction_l3665_366518


namespace second_company_can_hire_three_geniuses_l3665_366552

/-- Represents a programmer --/
structure Programmer where
  id : Nat

/-- Represents a genius programmer --/
structure Genius extends Programmer

/-- Represents the hiring game between two companies --/
structure HiringGame where
  programmers : List Programmer
  geniuses : List Genius
  acquaintances : List (Programmer × Programmer)

/-- Represents a company's hiring strategy --/
structure HiringStrategy where
  nextHire : List Programmer → List Programmer → Option Programmer

/-- The result of the hiring game --/
inductive GameResult
  | FirstCompanyWins
  | SecondCompanyWins

/-- Simulates the hiring game given two strategies --/
def playGame (game : HiringGame) (strategy1 strategy2 : HiringStrategy) : GameResult :=
  sorry

/-- Theorem stating that there exists a winning strategy for the second company --/
theorem second_company_can_hire_three_geniuses :
  ∃ (game : HiringGame) (strategy : HiringStrategy),
    (game.geniuses.length = 4) →
    ∀ (opponent_strategy : HiringStrategy),
      playGame game opponent_strategy strategy = GameResult.SecondCompanyWins :=
sorry

end second_company_can_hire_three_geniuses_l3665_366552


namespace awards_sum_is_80_l3665_366517

/-- The number of awards won by Scott -/
def scott_awards : ℕ := 4

/-- The number of awards won by Jessie -/
def jessie_awards : ℕ := 3 * scott_awards

/-- The number of awards won by the rival athlete -/
def rival_awards : ℕ := 2 * jessie_awards

/-- The number of awards won by Brad -/
def brad_awards : ℕ := (5 * rival_awards) / 3

/-- The total number of awards won by all four athletes -/
def total_awards : ℕ := scott_awards + jessie_awards + rival_awards + brad_awards

theorem awards_sum_is_80 : total_awards = 80 := by
  sorry

end awards_sum_is_80_l3665_366517


namespace tan_negative_1125_degrees_l3665_366572

theorem tan_negative_1125_degrees : Real.tan ((-1125 : ℝ) * π / 180) = 1 := by
  sorry

end tan_negative_1125_degrees_l3665_366572


namespace homework_problem_l3665_366547

theorem homework_problem (p t : ℕ) : 
  p > 0 → 
  t > 0 → 
  p ≥ 15 → 
  3 * p - 5 ≥ 20 → 
  p * t = (3 * p - 5) * (t - 3) → 
  p * t = 100 := by
  sorry

end homework_problem_l3665_366547


namespace bread_slice_cost_l3665_366592

/-- Calculates the cost per slice of bread in cents -/
def cost_per_slice (num_loaves : ℕ) (slices_per_loaf : ℕ) (amount_paid : ℕ) (change : ℕ) : ℕ :=
  let total_cost := amount_paid - change
  let total_slices := num_loaves * slices_per_loaf
  (total_cost * 100) / total_slices

/-- Proves that the cost per slice is 40 cents given the problem conditions -/
theorem bread_slice_cost :
  cost_per_slice 3 20 40 16 = 40 := by
  sorry

#eval cost_per_slice 3 20 40 16

end bread_slice_cost_l3665_366592


namespace parabola_perpendicular_chords_theorem_l3665_366575

/-- A parabola with vertex at the origin and focus on the positive x-axis -/
structure Parabola where
  p : ℝ
  equation : ℝ × ℝ → Prop := fun (x, y) ↦ y^2 = 2 * p * x

/-- A line passing through two points -/
def Line (A B : ℝ × ℝ) : ℝ × ℝ → Prop :=
  fun P ↦ (P.2 - A.2) * (B.1 - A.1) = (P.1 - A.1) * (B.2 - A.2)

/-- Two lines are perpendicular -/
def Perpendicular (L₁ L₂ : (ℝ × ℝ → Prop)) : Prop :=
  ∃ A B C D, L₁ A ∧ L₁ B ∧ L₂ C ∧ L₂ D ∧
    (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

/-- The projection of a point onto a line -/
def Projection (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ → Prop :=
  fun H ↦ L H ∧ Perpendicular (Line P H) L

theorem parabola_perpendicular_chords_theorem (Γ : Parabola) :
  ∀ A B, Γ.equation A ∧ Γ.equation B ∧ 
         Perpendicular (Line (0, 0) A) (Line (0, 0) B) →
  (∃ M₀, M₀ = (2 * Γ.p, 0) ∧ Line A B M₀) ∧
  (∀ H, Projection (0, 0) (Line A B) H → 
        H.1^2 + H.2^2 - 2 * Γ.p * H.1 = 0) :=
sorry

end parabola_perpendicular_chords_theorem_l3665_366575


namespace estimate_fish_population_l3665_366529

/-- Estimates the number of fish in a pond using the catch-mark-recapture method. -/
theorem estimate_fish_population (initial_catch : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  initial_catch > 0 →
  second_catch > 0 →
  marked_in_second > 0 →
  marked_in_second ≤ second_catch →
  marked_in_second ≤ initial_catch →
  (initial_catch * second_catch) / marked_in_second = 1200 →
  ∃ (estimated_population : ℕ), estimated_population = 1200 :=
by sorry

end estimate_fish_population_l3665_366529


namespace roselyn_remaining_books_l3665_366594

/-- The number of books Roselyn has after giving books to Mara and Rebecca -/
def books_remaining (initial_books rebecca_books : ℕ) : ℕ :=
  initial_books - (rebecca_books + 3 * rebecca_books)

/-- Theorem stating that Roselyn has 60 books after giving books to Mara and Rebecca -/
theorem roselyn_remaining_books :
  books_remaining 220 40 = 60 := by
  sorry

end roselyn_remaining_books_l3665_366594


namespace smallest_angle_in_triangle_l3665_366535

theorem smallest_angle_in_triangle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  max a (max b c) = 120 →  -- The largest angle is 120°
  b / c = 3 / 2 →  -- Ratio of the other two angles is 3:2
  b > c →  -- Ensure b is the middle angle and c is the smallest
  c = 24 :=  -- The smallest angle is 24°
by sorry

end smallest_angle_in_triangle_l3665_366535


namespace sqrt_equation_solution_l3665_366511

theorem sqrt_equation_solution (a : ℝ) (h : a ≥ -1/4) :
  ∃ x : ℝ, x ≥ 0 ∧ Real.sqrt (a + Real.sqrt (a + x)) = x ∧ x = (1 + Real.sqrt (1 + 4*a)) / 2 := by
  sorry

end sqrt_equation_solution_l3665_366511


namespace expression_equals_eighteen_l3665_366510

theorem expression_equals_eighteen (x : ℝ) (h : x + 1 = 4) :
  (-3)^3 + (-3)^2 + (-3*x)^1 + 3*x^1 + 3^2 + 3^3 = 18 := by
  sorry

end expression_equals_eighteen_l3665_366510


namespace inverse_abs_is_geometric_sequence_preserving_l3665_366504

/-- A function is geometric sequence-preserving if it transforms any non-constant
    geometric sequence into another geometric sequence. -/
def IsGeometricSequencePreserving (f : ℝ → ℝ) : Prop :=
  ∀ (a : ℕ → ℝ) (q : ℝ),
    (∀ n, a n ≠ 0) →
    (∀ n, a (n + 1) = q * a n) →
    q ≠ 1 →
    ∃ r : ℝ, r ≠ 1 ∧ ∀ n, f (a (n + 1)) = r * f (a n)

/-- The function f(x) = 1/|x| is geometric sequence-preserving. -/
theorem inverse_abs_is_geometric_sequence_preserving :
    IsGeometricSequencePreserving (fun x ↦ 1 / |x|) := by
  sorry


end inverse_abs_is_geometric_sequence_preserving_l3665_366504


namespace length_of_AB_l3665_366530

-- Define the circle Γ
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 3}

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - m * p.2 - 1 = 0}
def l₂ (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | m * p.1 + p.2 - m = 0}

-- Define the intersection points
def A (m : ℝ) : ℝ × ℝ := sorry
def B (m : ℝ) : ℝ × ℝ := sorry
def C (m : ℝ) : ℝ × ℝ := sorry
def D (m : ℝ) : ℝ × ℝ := sorry

-- State the theorem
theorem length_of_AB (m : ℝ) : 
  A m ∈ Γ ∧ A m ∈ l₁ m ∧
  B m ∈ Γ ∧ B m ∈ l₂ m ∧
  C m ∈ Γ ∧ C m ∈ l₁ m ∧
  D m ∈ Γ ∧ D m ∈ l₂ m ∧
  (A m).2 > 0 ∧ (B m).2 > 0 ∧
  (C m).2 < 0 ∧ (D m).2 < 0 ∧
  (D m).2 - (C m).2 = (D m).1 - (C m).1 →
  Real.sqrt ((A m).1 - (B m).1)^2 + ((A m).2 - (B m).2)^2 = 2 * Real.sqrt 2 := by
  sorry

end length_of_AB_l3665_366530


namespace boat_reachable_area_l3665_366521

/-- Represents the speed of the boat in miles per hour -/
structure BoatSpeed where
  river : ℝ
  land : ℝ

/-- Calculates the area reachable by the boat given its speed and time limit -/
def reachable_area (speed : BoatSpeed) (time_limit : ℝ) : ℝ :=
  sorry

theorem boat_reachable_area :
  let speed : BoatSpeed := { river := 40, land := 10 }
  let time_limit : ℝ := 12 / 60 -- 12 minutes in hours
  reachable_area speed time_limit = 232 * π / 6 := by
  sorry

#eval (232 + 6 : Nat)

end boat_reachable_area_l3665_366521


namespace paving_cost_calculation_l3665_366580

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving the given rectangular floor is Rs. 28,875 -/
theorem paving_cost_calculation :
  paving_cost 5.5 3.75 1400 = 28875 := by
  sorry

end paving_cost_calculation_l3665_366580


namespace triangle_medians_theorem_l3665_366569

/-- Given a triangle with side lengths a, b, and c, and orthogonal medians m_a and m_b -/
def Triangle (a b c : ℝ) (m_a m_b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  m_a = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2) ∧
  m_b = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2) ∧
  m_a * m_b = 0  -- orthogonality condition

theorem triangle_medians_theorem {a b c m_a m_b : ℝ} (h : Triangle a b c m_a m_b) :
  let m_c := (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)
  -- 1. The medians form a right-angled triangle
  m_a^2 + m_b^2 = m_c^2 ∧
  -- 2. The inequality holds
  5*(a^2 + b^2 - c^2) ≥ 8*a*b :=
by sorry

end triangle_medians_theorem_l3665_366569


namespace union_of_M_and_N_l3665_366546

open Set

def M : Set ℝ := {x | x^2 - 4*x < 0}
def N : Set ℝ := {x | |x| ≤ 2}

theorem union_of_M_and_N : M ∪ N = Icc (-2) 4 := by sorry

end union_of_M_and_N_l3665_366546


namespace player1_receives_57_coins_l3665_366534

/-- Represents the number of players and sectors on the table -/
def n : ℕ := 9

/-- Represents the total number of rotations -/
def total_rotations : ℕ := 11

/-- Represents the coins received by player 4 -/
def player4_coins : ℕ := 90

/-- Represents the coins received by player 8 -/
def player8_coins : ℕ := 35

/-- Represents the coins received by player 1 -/
def player1_coins : ℕ := 57

/-- Theorem stating that given the conditions, player 1 receives 57 coins -/
theorem player1_receives_57_coins :
  n = 9 →
  total_rotations = 11 →
  player4_coins = 90 →
  player8_coins = 35 →
  player1_coins = 57 :=
by sorry

end player1_receives_57_coins_l3665_366534


namespace magic_box_solution_l3665_366501

-- Define the magic box function
def magicBox (a b : ℝ) : ℝ := a^2 + b - 1

-- State the theorem
theorem magic_box_solution :
  ∀ m : ℝ, magicBox m (-2*m) = 2 → m = 3 ∨ m = -1 :=
by
  sorry

end magic_box_solution_l3665_366501


namespace trajectory_is_parabola_l3665_366543

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line
def line_l : Set (ℝ × ℝ) := {p | p.1 = -3}

-- Define point A
def point_A : ℝ × ℝ := (3, 0)

-- Define the properties of the moving circle
def is_valid_circle (c : Circle) : Prop :=
  (c.center.1 - 3)^2 + c.center.2^2 = c.radius^2 ∧  -- passes through A(3,0)
  c.center.1 + c.radius = -3                        -- tangent to x = -3

-- Theorem statement
theorem trajectory_is_parabola :
  ∀ c : Circle, is_valid_circle c →
  ∃ x y : ℝ, c.center = (x, y) ∧ y^2 = 12 * x :=
sorry

end trajectory_is_parabola_l3665_366543


namespace number_puzzle_l3665_366523

theorem number_puzzle : ∃ x : ℝ, (x / 5 + 6 = 65) ∧ (x = 295) := by
  sorry

end number_puzzle_l3665_366523


namespace third_bouquet_carnations_l3665_366542

/-- Theorem: Given three bouquets of carnations with specific conditions, 
    the third bouquet contains 13 carnations. -/
theorem third_bouquet_carnations 
  (total_bouquets : ℕ)
  (first_bouquet : ℕ)
  (second_bouquet : ℕ)
  (average_carnations : ℕ)
  (h1 : total_bouquets = 3)
  (h2 : first_bouquet = 9)
  (h3 : second_bouquet = 14)
  (h4 : average_carnations = 12)
  (h5 : average_carnations * total_bouquets = first_bouquet + second_bouquet + (total_bouquets - 2)) :
  total_bouquets - 2 = 13 := by
  sorry


end third_bouquet_carnations_l3665_366542


namespace two_digit_sums_of_six_powers_of_two_l3665_366567

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_sum_of_six_powers_of_two (n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧
    n = 2^0 + 2^a + 2^b + 2^c + 2^d + 2^e + 2^f

theorem two_digit_sums_of_six_powers_of_two :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, is_two_digit n ∧ is_sum_of_six_powers_of_two n) ∧
    s.card = 2 :=
sorry

end two_digit_sums_of_six_powers_of_two_l3665_366567


namespace imaginary_part_of_z_l3665_366561

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := i / (1 + i)
  Complex.im z = (1 : ℝ) / 2 :=
by sorry

end imaginary_part_of_z_l3665_366561


namespace fifth_term_of_geometric_sequence_l3665_366525

/-- Given a geometric sequence of positive integers where the first term is 5 and the fourth term is 405,
    prove that the fifth term is 405. -/
theorem fifth_term_of_geometric_sequence (a : ℕ+) (r : ℕ+) : 
  a = 5 → a * r^3 = 405 → a * r^4 = 405 := by
  sorry

end fifth_term_of_geometric_sequence_l3665_366525


namespace convex_quadrilateral_triangle_angles_l3665_366578

theorem convex_quadrilateral_triangle_angles 
  (α β γ θ₁ θ₂ θ₃ θ₄ : Real) : 
  (α + β + γ = Real.pi) →  -- Sum of angles in a triangle is π radians (180°)
  (θ₁ + θ₂ + θ₃ + θ₄ = 2 * Real.pi) →  -- Sum of angles in a quadrilateral is 2π radians (360°)
  (θ₁ = α) → (θ₂ = β) → (θ₃ = γ) →  -- Three angles of quadrilateral equal to triangle angles
  ¬(θ₁ < Real.pi ∧ θ₂ < Real.pi ∧ θ₃ < Real.pi ∧ θ₄ < Real.pi)  -- Negation of convexity condition
  := by sorry

end convex_quadrilateral_triangle_angles_l3665_366578


namespace train_length_l3665_366505

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) : 
  speed = 36 * (1000 / 3600) → 
  time = 27.997760179185665 → 
  bridge_length = 180 → 
  speed * time - bridge_length = 99.97760179185665 := by
sorry

#eval (36 * (1000 / 3600) * 27.997760179185665 - 180)

end train_length_l3665_366505


namespace inequality_condition_l3665_366577

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 4 → (1 + x) * Real.log x + x ≤ x * a) ↔ 
  a ≥ (5 * Real.log 2) / 2 :=
sorry

end inequality_condition_l3665_366577


namespace pages_left_after_three_weeks_l3665_366506

-- Define the structure for a book
structure Book where
  totalPages : ℕ
  pagesRead : ℕ
  pagesPerDay : ℕ

-- Define Elliot's books
def book1 : Book := ⟨512, 194, 30⟩
def book2 : Book := ⟨298, 0, 20⟩
def book3 : Book := ⟨365, 50, 25⟩
def book4 : Book := ⟨421, 0, 15⟩

-- Define the number of days
def days : ℕ := 21

-- Function to calculate pages left after reading
def pagesLeftAfterReading (b : Book) (days : ℕ) : ℕ :=
  max 0 (b.totalPages - b.pagesRead - b.pagesPerDay * days)

-- Theorem statement
theorem pages_left_after_three_weeks :
  pagesLeftAfterReading book1 days + 
  pagesLeftAfterReading book2 days + 
  pagesLeftAfterReading book3 days + 
  pagesLeftAfterReading book4 days = 106 := by
  sorry

end pages_left_after_three_weeks_l3665_366506


namespace worst_player_is_niece_l3665_366540

-- Define the players
inductive Player
| Grandmother
| Niece
| Grandson
| SonInLaw

-- Define the sex of a player
inductive Sex
| Male
| Female

-- Define the generation of a player
inductive Generation
| Old
| Middle
| Young

-- Function to determine the sex of a player
def sex : Player → Sex
| Player.Grandmother => Sex.Female
| Player.Niece => Sex.Female
| Player.Grandson => Sex.Male
| Player.SonInLaw => Sex.Male

-- Function to determine the generation of a player
def generation : Player → Generation
| Player.Grandmother => Generation.Old
| Player.Niece => Generation.Young
| Player.Grandson => Generation.Young
| Player.SonInLaw => Generation.Middle

-- Function to determine if two players are cousins
def areCousins : Player → Player → Bool
| Player.Niece, Player.Grandson => true
| Player.Grandson, Player.Niece => true
| _, _ => false

-- Theorem statement
theorem worst_player_is_niece :
  ∀ (worst best : Player),
  (∃ cousin : Player, areCousins worst cousin ∧ sex cousin ≠ sex best) →
  generation worst ≠ generation best →
  worst = Player.Niece :=
by sorry

end worst_player_is_niece_l3665_366540


namespace cos_negative_480_deg_l3665_366590

theorem cos_negative_480_deg : Real.cos (-(480 * π / 180)) = -1/2 := by
  sorry

end cos_negative_480_deg_l3665_366590


namespace no_solution_equation_l3665_366598

theorem no_solution_equation :
  ¬ ∃ x : ℝ, x - 9 / (x - 5) = 5 - 9 / (x - 5) :=
sorry

end no_solution_equation_l3665_366598


namespace vector_equality_implies_norm_equality_l3665_366553

theorem vector_equality_implies_norm_equality 
  {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] 
  (a b : n) (ha : a ≠ 0) (hb : b ≠ 0) :
  a + 2 • b = 0 → ‖a - b‖ = ‖a‖ + ‖b‖ := by
sorry

end vector_equality_implies_norm_equality_l3665_366553


namespace random_walk_properties_l3665_366503

/-- Represents a random walk on a line -/
structure RandomWalk where
  a : ℕ  -- number of steps to the right
  b : ℕ  -- number of steps to the left
  h : a > b

/-- The maximum possible range of the random walk -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of the random walk -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences achieving the maximum range -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

theorem random_walk_properties (w : RandomWalk) :
  (max_range w = w.a) ∧
  (min_range w = w.a - w.b) ∧
  (max_range_sequences w = w.b + 1) := by
  sorry

end random_walk_properties_l3665_366503


namespace largest_t_value_for_temperature_l3665_366522

theorem largest_t_value_for_temperature (t : ℝ) :
  let f : ℝ → ℝ := λ x => -x^2 + 10*x + 60
  let solutions := {x : ℝ | f x = 80}
  ∃ max_t ∈ solutions, ∀ t ∈ solutions, t ≤ max_t ∧ max_t = 5 + 3 * Real.sqrt 5 :=
by sorry

end largest_t_value_for_temperature_l3665_366522


namespace xy_equation_l3665_366524

theorem xy_equation (x y : ℝ) 
  (h1 : (x + y) / 3 = 1.888888888888889)
  (h2 : x + 2 * y = 10) :
  x + y = 5.666666666666667 := by
  sorry

end xy_equation_l3665_366524


namespace val_coins_value_l3665_366544

/-- Calculates the total value of Val's coins given the initial number of nickels and the number of additional nickels found. -/
def total_value (initial_nickels : ℕ) (found_nickels : ℕ) : ℚ :=
  let total_nickels := initial_nickels + found_nickels
  let dimes := 3 * initial_nickels
  let quarters := 2 * dimes
  let nickel_value := (5 : ℚ) / 100
  let dime_value := (10 : ℚ) / 100
  let quarter_value := (25 : ℚ) / 100
  (total_nickels : ℚ) * nickel_value + (dimes : ℚ) * dime_value + (quarters : ℚ) * quarter_value

theorem val_coins_value :
  total_value 20 40 = 39 := by
  sorry

end val_coins_value_l3665_366544


namespace inequality_proof_l3665_366597

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : 1/a + 1/b + 1/c = 1) :
  Real.sqrt (a + b*c) + Real.sqrt (b + c*a) + Real.sqrt (c + a*b) ≥ 
  Real.sqrt (a*b*c) + Real.sqrt a + Real.sqrt b + Real.sqrt c :=
sorry

end inequality_proof_l3665_366597


namespace one_fourth_of_8_2_l3665_366557

theorem one_fourth_of_8_2 : (8.2 : ℚ) / 4 = 41 / 20 := by sorry

end one_fourth_of_8_2_l3665_366557


namespace cubic_monomial_exists_l3665_366548

/-- A cubic monomial with variables x and y and a negative coefficient exists. -/
theorem cubic_monomial_exists : ∃ (a : ℝ) (i j : ℕ), 
  a < 0 ∧ i + j = 3 ∧ (λ (x y : ℝ) => a * x^i * y^j) ≠ 0 := by
  sorry

end cubic_monomial_exists_l3665_366548


namespace parallelogram_side_length_l3665_366515

theorem parallelogram_side_length
  (s : ℝ)
  (side1 : ℝ)
  (side2 : ℝ)
  (angle : ℝ)
  (area : ℝ)
  (h1 : side1 = s)
  (h2 : side2 = 3 * s)
  (h3 : angle = π / 3)  -- 60 degrees in radians
  (h4 : area = 27 * Real.sqrt 3)
  (h5 : area = side1 * side2 * Real.sin angle) :
  s = 3 := by
sorry

end parallelogram_side_length_l3665_366515


namespace cone_lateral_surface_area_l3665_366537

/-- The lateral surface area of a cone with base radius 2 cm and slant height 5 cm is 10π cm². -/
theorem cone_lateral_surface_area : 
  let r : ℝ := 2  -- radius in cm
  let l : ℝ := 5  -- slant height in cm
  let lateral_area := (1/2) * l * (2 * Real.pi * r)
  lateral_area = 10 * Real.pi
  := by sorry

end cone_lateral_surface_area_l3665_366537


namespace savings_percentage_l3665_366507

def monthly_salary : ℝ := 1000
def savings_after_increase : ℝ := 175
def expense_increase_rate : ℝ := 0.10

theorem savings_percentage :
  ∃ (savings_rate : ℝ),
    savings_rate * monthly_salary = monthly_salary - (monthly_salary - savings_rate * monthly_salary) * (1 + expense_increase_rate) ∧
    savings_rate * monthly_salary = savings_after_increase ∧
    savings_rate = 0.25 :=
by sorry

end savings_percentage_l3665_366507


namespace sara_movie_purchase_cost_l3665_366562

/-- The amount Sara spent on buying a movie, given her other movie-related expenses --/
theorem sara_movie_purchase_cost (ticket_price : ℝ) (ticket_count : ℕ) 
  (rental_cost : ℝ) (total_spent : ℝ) (h1 : ticket_price = 10.62) 
  (h2 : ticket_count = 2) (h3 : rental_cost = 1.59) (h4 : total_spent = 36.78) : 
  total_spent - (ticket_price * ↑ticket_count + rental_cost) = 13.95 := by
  sorry

end sara_movie_purchase_cost_l3665_366562


namespace max_b_value_l3665_366527

/-- Given two functions f and g with a common point and equal tangents, 
    prove the maximum value of b -/
theorem max_b_value (a : ℝ) (h_a : a > 0) :
  let f := fun x : ℝ => (1/2) * x^2 + 2*a*x
  let g := fun x b : ℝ => 3*a^2 * Real.log x + b
  ∃ (x₀ b : ℝ), 
    (f x₀ = g x₀ b) ∧ 
    (deriv f x₀ = deriv (fun x => g x b) x₀) →
    (∀ b' : ℝ, ∃ x : ℝ, f x = g x b' → b' ≤ (3/2) * Real.exp ((2:ℝ)/3)) :=
by sorry

end max_b_value_l3665_366527


namespace largest_divisor_of_expression_l3665_366550

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (10*x + 2) * (10*x + 6) * (5*x + 5) = 960 * k) ∧
  (∀ (m : ℤ), m > 960 → ¬(∀ (y : ℤ), Odd y → ∃ (l : ℤ), (10*y + 2) * (10*y + 6) * (5*y + 5) = m * l)) :=
by sorry

end largest_divisor_of_expression_l3665_366550


namespace inscribed_hexagon_radius_theorem_l3665_366532

/-- A hexagon inscribed in a circle with radius R, where three consecutive sides are equal to a
    and the other three consecutive sides are equal to b. -/
structure InscribedHexagon (R a b : ℝ) : Prop where
  radius_positive : R > 0
  side_a_positive : a > 0
  side_b_positive : b > 0
  three_sides_a : ∃ (AB BC CD : ℝ), AB = a ∧ BC = a ∧ CD = a
  three_sides_b : ∃ (DE EF FA : ℝ), DE = b ∧ EF = b ∧ FA = b

/-- The theorem stating the relationship between the radius R and sides a and b of the inscribed hexagon. -/
theorem inscribed_hexagon_radius_theorem (R a b : ℝ) (h : InscribedHexagon R a b) :
  R^2 = (a^2 + b^2 + a*b) / 3 := by
  sorry

end inscribed_hexagon_radius_theorem_l3665_366532


namespace rounding_proof_l3665_366502

def base : ℚ := 1003 / 1000

def power : ℕ := 4

def exact_result : ℚ := base ^ power

def rounded_result : ℚ := 1012 / 1000

def decimal_places : ℕ := 3

theorem rounding_proof : 
  (round (exact_result * 10^decimal_places) / 10^decimal_places) = rounded_result := by
  sorry

end rounding_proof_l3665_366502
