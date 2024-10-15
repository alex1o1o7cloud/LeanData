import Mathlib

namespace NUMINAMATH_CALUDE_factorization_equality_l1643_164338

theorem factorization_equality (a b : ℝ) :
  2*a*b^2 - 6*a^2*b^2 + 4*a^3*b^2 = 2*a*b^2*(2*a - 1)*(a - 1) := by sorry

end NUMINAMATH_CALUDE_factorization_equality_l1643_164338


namespace NUMINAMATH_CALUDE_f_always_positive_l1643_164392

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + 4*x^3 + a*x^2 - 4*x + 1

/-- Theorem stating that f(x) is always positive if and only if a > 2 -/
theorem f_always_positive (a : ℝ) : (∀ x : ℝ, f a x > 0) ↔ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_f_always_positive_l1643_164392


namespace NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l1643_164326

theorem lattice_points_on_hyperbola :
  let equation := fun (x y : ℤ) => x^2 - y^2 = 1500^2
  (∑' p : ℤ × ℤ, if equation p.1 p.2 then 1 else 0) = 90 :=
sorry

end NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l1643_164326


namespace NUMINAMATH_CALUDE_complex_division_equality_l1643_164367

theorem complex_division_equality : (1 - Complex.I) / Complex.I = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_equality_l1643_164367


namespace NUMINAMATH_CALUDE_jiaotong_primary_school_students_l1643_164316

theorem jiaotong_primary_school_students (b g : ℕ) : 
  b = 7 * g ∧ b = g + 900 → b + g = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jiaotong_primary_school_students_l1643_164316


namespace NUMINAMATH_CALUDE_intersection_k_value_l1643_164329

/-- Given two lines that intersect at x = 5, prove that k = 10 -/
theorem intersection_k_value (k : ℝ) : 
  (∃ y : ℝ, 3 * 5 - y = k ∧ -5 - y = -10) → k = 10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_k_value_l1643_164329


namespace NUMINAMATH_CALUDE_product_sum_inequality_l1643_164357

theorem product_sum_inequality (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = p ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a * b = p → x + y ≤ a + b :=
sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l1643_164357


namespace NUMINAMATH_CALUDE_sum_of_perimeters_theorem_l1643_164370

/-- The sum of perimeters of all polygons in the sequence formed by repeatedly
    joining mid-points of an n-sided regular polygon with initial side length 60 cm. -/
def sum_of_perimeters (n : ℕ) : ℝ :=
  n * 120

/-- Theorem: The sum of perimeters of all polygons in the sequence formed by repeatedly
    joining mid-points of an n-sided regular polygon with initial side length 60 cm
    is equal to n * 120 cm. -/
theorem sum_of_perimeters_theorem (n : ℕ) (h : n > 0) :
  let initial_side_length : ℝ := 60
  let perimeter_sequence : ℕ → ℝ := λ k => n * (initial_side_length / 2^(k - 1))
  (∑' k, perimeter_sequence k) = sum_of_perimeters n :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_perimeters_theorem_l1643_164370


namespace NUMINAMATH_CALUDE_vasilyev_car_loan_payment_l1643_164397

/-- Calculates the maximum monthly car loan payment for the Vasilyev family --/
def max_car_loan_payment (total_income : ℝ) (total_expenses : ℝ) (emergency_fund_rate : ℝ) : ℝ :=
  let remaining_income := total_income - total_expenses
  let emergency_fund := emergency_fund_rate * remaining_income
  total_income - total_expenses - emergency_fund

/-- Theorem stating the maximum monthly car loan payment for the Vasilyev family --/
theorem vasilyev_car_loan_payment :
  max_car_loan_payment 84600 49800 0.1 = 31320 := by
  sorry

end NUMINAMATH_CALUDE_vasilyev_car_loan_payment_l1643_164397


namespace NUMINAMATH_CALUDE_matrix_product_50_l1643_164320

def matrix_product (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range n).foldl
    (fun acc k => acc * !![1, 2*(k+1); 0, 1])
    !![1, 0; 0, 1]

theorem matrix_product_50 :
  matrix_product 50 = !![1, 2550; 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_50_l1643_164320


namespace NUMINAMATH_CALUDE_bowlfuls_in_box_l1643_164334

/-- Represents the number of clusters of oats in each spoonful -/
def clusters_per_spoonful : ℕ := 4

/-- Represents the number of spoonfuls in each bowl of cereal -/
def spoonfuls_per_bowl : ℕ := 25

/-- Represents the total number of clusters of oats in each box -/
def clusters_per_box : ℕ := 500

/-- Calculates the number of bowlfuls of cereal in each box -/
def bowlfuls_per_box : ℕ :=
  clusters_per_box / (clusters_per_spoonful * spoonfuls_per_bowl)

/-- Theorem stating that the number of bowlfuls of cereal in each box is 5 -/
theorem bowlfuls_in_box : bowlfuls_per_box = 5 := by
  sorry

end NUMINAMATH_CALUDE_bowlfuls_in_box_l1643_164334


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l1643_164324

theorem largest_integer_inequality : 
  ∀ y : ℤ, y ≤ 0 ↔ (y : ℚ) / 4 + 3 / 7 < 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l1643_164324


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1643_164333

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B : A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1643_164333


namespace NUMINAMATH_CALUDE_jackies_free_time_l1643_164337

def hours_in_day : ℕ := 24

def working_hours : ℕ := 8
def exercise_hours : ℕ := 3
def sleep_hours : ℕ := 8

def scheduled_hours : ℕ := working_hours + exercise_hours + sleep_hours

theorem jackies_free_time : hours_in_day - scheduled_hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_jackies_free_time_l1643_164337


namespace NUMINAMATH_CALUDE_cylinder_surface_area_and_volume_l1643_164374

/-- Given a cylinder with cross-sectional area M and axial section area N,
    prove its surface area and volume. -/
theorem cylinder_surface_area_and_volume (M N : ℝ) (M_pos : M > 0) (N_pos : N > 0) :
  ∃ (surface_area volume : ℝ),
    surface_area = N * Real.pi + 2 * M ∧
    volume = (N / 2) * Real.sqrt (M * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_and_volume_l1643_164374


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l1643_164317

/-- Given the initial length of Isabella's hair, the amount it grew, and the final length,
    prove that the initial length plus the growth equals the final length. -/
theorem isabellas_hair_growth (initial_length growth final_length : ℝ) 
    (h1 : growth = 6)
    (h2 : final_length = 24)
    (h3 : initial_length + growth = final_length) : 
  initial_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l1643_164317


namespace NUMINAMATH_CALUDE_ndfl_calculation_l1643_164390

/-- Calculates the total NDFL (personal income tax) on securities income -/
def calculate_ndfl (dividend_income : ℕ) (ofz_income : ℕ) (corporate_bond_income : ℕ) 
                   (shares_sold : ℕ) (sale_price_per_share : ℕ) (purchase_price_per_share : ℕ) : ℕ :=
  let dividend_tax := dividend_income * 13 / 100
  let corporate_bond_tax := corporate_bond_income * 13 / 100
  let capital_gain := shares_sold * (sale_price_per_share - purchase_price_per_share)
  let capital_gain_tax := capital_gain * 13 / 100
  dividend_tax + corporate_bond_tax + capital_gain_tax

/-- The total NDFL on securities income is 11,050 rubles -/
theorem ndfl_calculation : 
  calculate_ndfl 50000 40000 30000 100 200 150 = 11050 := by
  sorry

end NUMINAMATH_CALUDE_ndfl_calculation_l1643_164390


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l1643_164387

theorem scientific_notation_equality : 935000000 = 9.35 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l1643_164387


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1643_164325

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₄ + a₂ + a₀ = -121 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1643_164325


namespace NUMINAMATH_CALUDE_roadster_paving_cement_usage_l1643_164385

/-- The amount of cement used for Lexi's street in tons -/
def lexi_cement : ℝ := 10

/-- The amount of cement used for Tess's street in tons -/
def tess_cement : ℝ := 5.1

/-- The total amount of cement used by Roadster's Paving Company in tons -/
def total_cement : ℝ := lexi_cement + tess_cement

theorem roadster_paving_cement_usage :
  total_cement = 15.1 := by sorry

end NUMINAMATH_CALUDE_roadster_paving_cement_usage_l1643_164385


namespace NUMINAMATH_CALUDE_buddys_gym_class_size_l1643_164364

theorem buddys_gym_class_size (group1 : ℕ) (group2 : ℕ) 
  (h1 : group1 = 34) (h2 : group2 = 37) : group1 + group2 = 71 := by
  sorry

end NUMINAMATH_CALUDE_buddys_gym_class_size_l1643_164364


namespace NUMINAMATH_CALUDE_vector_equation_vectors_parallel_l1643_164304

/-- Given vectors in R² --/
def a : Fin 2 → ℚ := ![3, 2]
def b : Fin 2 → ℚ := ![-1, 2]
def c : Fin 2 → ℚ := ![4, 1]

/-- Theorem for part 1 --/
theorem vector_equation :
  a = (5/9 : ℚ) • b + (8/9 : ℚ) • c := by sorry

/-- Helper function to check if two vectors are parallel --/
def are_parallel (v w : Fin 2 → ℚ) : Prop :=
  v 0 * w 1 = v 1 * w 0

/-- Theorem for part 2 --/
theorem vectors_parallel :
  are_parallel (a + (-16/3 : ℚ) • c) (2 • b - a) := by sorry

end NUMINAMATH_CALUDE_vector_equation_vectors_parallel_l1643_164304


namespace NUMINAMATH_CALUDE_suzie_reading_rate_l1643_164389

/-- The number of pages Liza reads in an hour -/
def liza_pages_per_hour : ℕ := 20

/-- The number of additional pages Liza reads compared to Suzie in 3 hours -/
def liza_additional_pages : ℕ := 15

/-- The number of hours considered -/
def hours : ℕ := 3

/-- The number of pages Suzie reads in an hour -/
def suzie_pages_per_hour : ℕ := 15

theorem suzie_reading_rate :
  suzie_pages_per_hour = (liza_pages_per_hour * hours - liza_additional_pages) / hours :=
by sorry

end NUMINAMATH_CALUDE_suzie_reading_rate_l1643_164389


namespace NUMINAMATH_CALUDE_wireless_internet_percentage_l1643_164378

/-- The percentage of major airline companies that offer free on-board snacks -/
def snacks_percentage : ℝ := 70

/-- The greatest possible percentage of major airline companies that offer both wireless internet and free on-board snacks -/
def both_services_percentage : ℝ := 50

/-- The percentage of major airline companies that equip their planes with wireless internet access -/
def wireless_percentage : ℝ := 50

theorem wireless_internet_percentage :
  wireless_percentage = 50 :=
sorry

end NUMINAMATH_CALUDE_wireless_internet_percentage_l1643_164378


namespace NUMINAMATH_CALUDE_rental_company_properties_l1643_164355

/-- Represents the rental company's car rental scenario. -/
structure RentalCompany where
  totalCars : ℕ := 100
  initialRent : ℕ := 3000
  rentIncrement : ℕ := 50
  rentedCarMaintenance : ℕ := 150
  nonRentedCarMaintenance : ℕ := 50

/-- Calculates the number of cars rented given a specific rent. -/
def carsRented (company : RentalCompany) (rent : ℕ) : ℕ :=
  company.totalCars - (rent - company.initialRent) / company.rentIncrement

/-- Calculates the monthly revenue given a specific rent. -/
def monthlyRevenue (company : RentalCompany) (rent : ℕ) : ℕ :=
  let rented := carsRented company rent
  rent * rented - company.rentedCarMaintenance * rented - 
    company.nonRentedCarMaintenance * (company.totalCars - rented)

/-- Theorem stating the properties of the rental company scenario. -/
theorem rental_company_properties (company : RentalCompany) : 
  carsRented company 3600 = 88 ∧ 
  (∃ (maxRent : ℕ), maxRent = 4050 ∧ 
    (∀ (rent : ℕ), monthlyRevenue company rent ≤ monthlyRevenue company maxRent) ∧
    monthlyRevenue company maxRent = 307050) := by
  sorry


end NUMINAMATH_CALUDE_rental_company_properties_l1643_164355


namespace NUMINAMATH_CALUDE_g_of_f_3_l1643_164319

def f (x : ℝ) : ℝ := x^3 + 3

def g (x : ℝ) : ℝ := 2*x^2 + 2*x + x^3 + 1

theorem g_of_f_3 : g (f 3) = 28861 := by
  sorry

end NUMINAMATH_CALUDE_g_of_f_3_l1643_164319


namespace NUMINAMATH_CALUDE_stratified_sample_o_blood_type_l1643_164393

/-- Calculates the number of students with blood type O in a stratified sample -/
def stratifiedSampleO (totalStudents : ℕ) (oTypeStudents : ℕ) (sampleSize : ℕ) : ℕ :=
  (oTypeStudents * sampleSize) / totalStudents

/-- Theorem: In a stratified sample of 40 students from a population of 500 students, 
    where 200 students have blood type O, the number of students with blood type O 
    in the sample should be 16. -/
theorem stratified_sample_o_blood_type 
  (totalStudents : ℕ) 
  (oTypeStudents : ℕ) 
  (sampleSize : ℕ) 
  (h1 : totalStudents = 500) 
  (h2 : oTypeStudents = 200) 
  (h3 : sampleSize = 40) :
  stratifiedSampleO totalStudents oTypeStudents sampleSize = 16 := by
  sorry

#eval stratifiedSampleO 500 200 40

end NUMINAMATH_CALUDE_stratified_sample_o_blood_type_l1643_164393


namespace NUMINAMATH_CALUDE_birthday_square_l1643_164300

theorem birthday_square (x y : ℕ+) (h1 : 40000 + 1000 * x + 100 * y + 29 < 100000) : 
  ∃ (T : ℕ), T = 2379 ∧ T^2 = 40000 + 1000 * x + 100 * y + 29 := by
  sorry

end NUMINAMATH_CALUDE_birthday_square_l1643_164300


namespace NUMINAMATH_CALUDE_sandy_book_purchase_l1643_164321

theorem sandy_book_purchase (books_shop1 : ℕ) (cost_shop1 : ℕ) (cost_shop2 : ℕ) (avg_price : ℚ) :
  books_shop1 = 65 →
  cost_shop1 = 1280 →
  cost_shop2 = 880 →
  avg_price = 18 →
  ∃ (books_shop2 : ℕ), 
    (books_shop1 + books_shop2) * avg_price = cost_shop1 + cost_shop2 ∧
    books_shop2 = 55 :=
by sorry

end NUMINAMATH_CALUDE_sandy_book_purchase_l1643_164321


namespace NUMINAMATH_CALUDE_absolute_value_not_positive_l1643_164348

theorem absolute_value_not_positive (x : ℚ) : |4*x + 6| ≤ 0 ↔ x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_not_positive_l1643_164348


namespace NUMINAMATH_CALUDE_vasya_always_wins_l1643_164313

/-- Represents the state of the game with the number of piles -/
structure GameState :=
  (piles : ℕ)

/-- Represents a player in the game -/
inductive Player
| Petya
| Vasya

/-- Defines a single move in the game -/
def move (state : GameState) : GameState :=
  { piles := state.piles + 2 }

/-- Determines if a given state is a winning state for the current player -/
def is_winning_state (state : GameState) : Prop :=
  ∃ (n : ℕ), state.piles = 2 * n + 1

/-- The main theorem stating that Vasya (second player) always wins -/
theorem vasya_always_wins :
  ∀ (initial_state : GameState),
  initial_state.piles = 3 →
  is_winning_state (move initial_state) :=
sorry

end NUMINAMATH_CALUDE_vasya_always_wins_l1643_164313


namespace NUMINAMATH_CALUDE_peach_difference_is_eight_l1643_164311

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 14

/-- The number of yellow peaches in the basket -/
def yellow_peaches : ℕ := 6

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := 2

/-- The difference between the number of green peaches and yellow peaches -/
def peach_difference : ℕ := green_peaches - yellow_peaches

theorem peach_difference_is_eight : peach_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_is_eight_l1643_164311


namespace NUMINAMATH_CALUDE_sum_of_integers_l1643_164398

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1643_164398


namespace NUMINAMATH_CALUDE_focus_of_specific_parabola_l1643_164350

/-- The focus of a parabola defined by y = ax^2 + bx + c -/
def parabola_focus (a b c : ℝ) : ℝ × ℝ :=
  sorry

theorem focus_of_specific_parabola :
  parabola_focus 9 6 (-4) = (-1/3, -59/12) := by
  sorry

end NUMINAMATH_CALUDE_focus_of_specific_parabola_l1643_164350


namespace NUMINAMATH_CALUDE_circle_center_sum_l1643_164380

/-- For a circle with equation x^2 + y^2 = 6x + 8y - 15, if (h, k) is its center, then h + k = 7 -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 8*y - 15 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - (6*h + 8*k - 15))) →
  h + k = 7 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1643_164380


namespace NUMINAMATH_CALUDE_rhombus_side_length_l1643_164307

/-- A rhombus with an inscribed circle of radius 2, where the diagonal divides the rhombus into two equilateral triangles, has a side length of 8√3/3. -/
theorem rhombus_side_length (r : ℝ) (s : ℝ) :
  r = 2 →  -- The radius of the inscribed circle is 2
  s > 0 →  -- The side length is positive
  s^2 = (s/2)^2 + 16 →  -- From the diagonal relationship
  s * 4 = (s * s) / 2 →  -- Area equality
  s = 8 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l1643_164307


namespace NUMINAMATH_CALUDE_sin_75_plus_sin_15_l1643_164394

theorem sin_75_plus_sin_15 : Real.sin (75 * π / 180) + Real.sin (15 * π / 180) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_plus_sin_15_l1643_164394


namespace NUMINAMATH_CALUDE_trees_in_gray_areas_trees_in_gray_areas_proof_l1643_164356

/-- Given three pictures with an equal number of trees, where the white areas
contain 82, 82, and 100 trees respectively, the total number of trees in the
gray areas is 26. -/
theorem trees_in_gray_areas : ℕ → ℕ → ℕ → Prop :=
  fun (total : ℕ) (x : ℕ) (y : ℕ) =>
    (total = 82 + x) ∧
    (total = 82 + y) ∧
    (total = 100) →
    x + y = 26

/-- Proof of the theorem -/
theorem trees_in_gray_areas_proof : ∃ (total : ℕ), trees_in_gray_areas total 18 8 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_gray_areas_trees_in_gray_areas_proof_l1643_164356


namespace NUMINAMATH_CALUDE_exam_pass_count_l1643_164372

theorem exam_pass_count (total_candidates : ℕ) (avg_all : ℚ) (avg_pass : ℚ) (avg_fail : ℚ) : 
  total_candidates = 120 →
  avg_all = 35 →
  avg_pass = 39 →
  avg_fail = 15 →
  ∃ (pass_count : ℕ), pass_count = 100 ∧ pass_count ≤ total_candidates :=
by sorry

end NUMINAMATH_CALUDE_exam_pass_count_l1643_164372


namespace NUMINAMATH_CALUDE_boat_length_boat_length_is_four_l1643_164343

/-- The length of a boat given specific conditions --/
theorem boat_length (breadth : ℝ) (sink_depth : ℝ) (man_mass : ℝ) 
                    (water_density : ℝ) (gravity : ℝ) : ℝ :=
  let boat_length := 4
  let volume_displaced := man_mass * gravity / (water_density * gravity)
  let calculated_length := volume_displaced / (breadth * sink_depth)
  
  -- Assumptions
  have h1 : breadth = 3 := by sorry
  have h2 : sink_depth = 0.01 := by sorry
  have h3 : man_mass = 120 := by sorry
  have h4 : water_density = 1000 := by sorry
  have h5 : gravity = 9.8 := by sorry
  
  -- Proof that the calculated length equals the given boat length
  have h6 : calculated_length = boat_length := by sorry
  
  boat_length

/-- Main theorem stating the boat length is 4 meters --/
theorem boat_length_is_four : 
  boat_length 3 0.01 120 1000 9.8 = 4 := by sorry

end NUMINAMATH_CALUDE_boat_length_boat_length_is_four_l1643_164343


namespace NUMINAMATH_CALUDE_intersection_when_m_is_two_subset_condition_l1643_164345

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}
def B : Set ℝ := {x | -4 ≤ x ∧ x ≤ 2}

-- Theorem 1: When m = 2, A ∩ B = [1, 2]
theorem intersection_when_m_is_two : 
  A 2 ∩ B = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2: A ⊆ (A ∩ B) if and only if -2 ≤ m ≤ 1/2
theorem subset_condition (m : ℝ) : 
  A m ⊆ (A m ∩ B) ↔ -2 ≤ m ∧ m ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_two_subset_condition_l1643_164345


namespace NUMINAMATH_CALUDE_derivative_of_f_l1643_164312

-- Define the function f(x) = (3x+4)(2x+6)
def f (x : ℝ) : ℝ := (3*x + 4) * (2*x + 6)

-- State the theorem
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 12*x + 26 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l1643_164312


namespace NUMINAMATH_CALUDE_most_colored_pencils_l1643_164340

theorem most_colored_pencils (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 24 →
  red = total / 4 →
  blue = red + 6 →
  yellow = total - red - blue →
  blue > red ∧ blue > yellow :=
by
  sorry

end NUMINAMATH_CALUDE_most_colored_pencils_l1643_164340


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1643_164369

/-- Given a geometric sequence {a_n} with the specified conditions, prove that a₆ + a₇ + a₈ = 32 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (h_sum1 : a 1 + a 2 + a 3 = 1)
  (h_sum2 : a 2 + a 3 + a 4 = 2) :
  a 6 + a 7 + a 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1643_164369


namespace NUMINAMATH_CALUDE_profit_calculation_correct_l1643_164361

/-- Represents the demand scenario --/
inductive DemandScenario
  | High
  | Moderate
  | Low

/-- Calculates the profit for a given demand scenario --/
def calculate_profit (scenario : DemandScenario) : ℚ :=
  let total_cloth : ℕ := 40
  let profit_per_meter : ℚ := 35
  let high_discount : ℚ := 0.1
  let moderate_discount : ℚ := 0.05
  let sales_tax : ℚ := 0.05
  let low_demand_cloth : ℕ := 30

  match scenario with
  | DemandScenario.High =>
    let original_profit := total_cloth * profit_per_meter
    let discounted_profit := original_profit * (1 - high_discount)
    discounted_profit * (1 - sales_tax)
  | DemandScenario.Moderate =>
    let original_profit := total_cloth * profit_per_meter
    let discounted_profit := original_profit * (1 - moderate_discount)
    discounted_profit * (1 - sales_tax)
  | DemandScenario.Low =>
    let original_profit := low_demand_cloth * profit_per_meter
    original_profit * (1 - sales_tax)

theorem profit_calculation_correct :
  (calculate_profit DemandScenario.High = 1197) ∧
  (calculate_profit DemandScenario.Moderate = 1263.5) ∧
  (calculate_profit DemandScenario.Low = 997.5) :=
by sorry

end NUMINAMATH_CALUDE_profit_calculation_correct_l1643_164361


namespace NUMINAMATH_CALUDE_expression_evaluation_l1643_164327

/-- Given a = 3, b = 2, and c = 1, prove that (a^3 + b^2 + c)^2 - (a^3 + b^2 - c)^2 = 124 -/
theorem expression_evaluation (a b c : ℕ) (ha : a = 3) (hb : b = 2) (hc : c = 1) :
  (a^3 + b^2 + c)^2 - (a^3 + b^2 - c)^2 = 124 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1643_164327


namespace NUMINAMATH_CALUDE_fixed_point_line_l1643_164384

/-- Given a line that always passes through a fixed point, prove that the line
    passing through this fixed point and the origin has the equation y = 2x -/
theorem fixed_point_line (a : ℝ) : 
  (∃ (x₀ y₀ : ℝ), ∀ (x y : ℝ), a * x + y + a + 2 = 0 → x = x₀ ∧ y = y₀) → 
  ∃ (m : ℝ), ∀ (x y : ℝ), 
    (a * x₀ + y₀ + a + 2 = 0 ∧ 
     y - y₀ = m * (x - x₀) ∧ 
     0 - y₀ = m * (0 - x₀)) → 
    y = 2 * x :=
sorry

end NUMINAMATH_CALUDE_fixed_point_line_l1643_164384


namespace NUMINAMATH_CALUDE_investment_ratio_l1643_164347

/-- Given two investors P and Q who divide their profit in the ratio 2:3,
    where P invested 30000, prove that Q invested 45000. -/
theorem investment_ratio (p q : ℕ) (profit_ratio : ℚ) :
  profit_ratio = 2 / 3 →
  p = 30000 →
  q * profit_ratio = p * (1 - profit_ratio) →
  q = 45000 := by
sorry

end NUMINAMATH_CALUDE_investment_ratio_l1643_164347


namespace NUMINAMATH_CALUDE_perpendicular_to_plane_false_l1643_164318

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)  -- line is subset of plane
variable (perp : Line → Line → Prop)     -- line is perpendicular to line
variable (perpPlane : Line → Plane → Prop)  -- line is perpendicular to plane

-- State the theorem
theorem perpendicular_to_plane_false
  (l m n : Line) (α : Plane)
  (h1 : subset m α)
  (h2 : subset n α)
  (h3 : perp l m)
  (h4 : perp l n) :
  ¬ (perpPlane l α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_plane_false_l1643_164318


namespace NUMINAMATH_CALUDE_jungkook_age_relation_l1643_164332

theorem jungkook_age_relation :
  ∃ (x : ℕ), 
    (46 - x : ℤ) = 4 * (16 - x : ℤ) ∧ 
    x ≤ 16 ∧ 
    x ≤ 46 ∧ 
    x = 6 :=
by sorry

end NUMINAMATH_CALUDE_jungkook_age_relation_l1643_164332


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implication_l1643_164339

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_implication 
  (a b c : Line) (α β γ : Plane) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h3 : perpendicular a α) 
  (h4 : perpendicular b β) 
  (h5 : parallel_lines a b) : 
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implication_l1643_164339


namespace NUMINAMATH_CALUDE_range_of_a_l1643_164303

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → a ∈ Set.Iic 1 ∪ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1643_164303


namespace NUMINAMATH_CALUDE_gregs_gold_is_20_l1643_164358

/-- Represents the amount of gold Greg has -/
def gregs_gold : ℝ := sorry

/-- Represents the amount of gold Katie has -/
def katies_gold : ℝ := sorry

/-- The total amount of gold is 100 -/
axiom total_gold : gregs_gold + katies_gold = 100

/-- Greg has four times less gold than Katie -/
axiom gold_ratio : gregs_gold = katies_gold / 4

/-- Theorem stating that Greg's gold amount is 20 -/
theorem gregs_gold_is_20 : gregs_gold = 20 := by sorry

end NUMINAMATH_CALUDE_gregs_gold_is_20_l1643_164358


namespace NUMINAMATH_CALUDE_garden_problem_l1643_164375

theorem garden_problem (a : ℝ) : 
  a > 0 → 
  (a + 3)^2 = 2 * a^2 + 9 → 
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_garden_problem_l1643_164375


namespace NUMINAMATH_CALUDE_factor_count_l1643_164386

/-- The number of positive factors of 180 that are also multiples of 15 -/
def count_factors : ℕ :=
  (Finset.filter (λ x => x ∣ 180 ∧ 15 ∣ x) (Finset.range 181)).card

theorem factor_count : count_factors = 6 := by
  sorry

end NUMINAMATH_CALUDE_factor_count_l1643_164386


namespace NUMINAMATH_CALUDE_complement_A_in_U_l1643_164391

def U : Set ℝ := {x | x < 2}
def A : Set ℝ := {x | x^2 < x}

theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l1643_164391


namespace NUMINAMATH_CALUDE_extended_equilateral_area_ratio_l1643_164351

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Extends a triangle by a factor along each side -/
def extendTriangle (t : Triangle) (factor : ℝ) : Triangle := sorry

/-- Main theorem: The area of an extended equilateral triangle is 9 times the original -/
theorem extended_equilateral_area_ratio (t : Triangle) :
  isEquilateral t →
  area (extendTriangle t 3) = 9 * area t := by sorry

end NUMINAMATH_CALUDE_extended_equilateral_area_ratio_l1643_164351


namespace NUMINAMATH_CALUDE_money_distribution_l1643_164302

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (AC : A + C = 200)
  (BC : B + C = 360) :
  C = 60 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1643_164302


namespace NUMINAMATH_CALUDE_potato_bag_weight_l1643_164373

theorem potato_bag_weight : ∀ w : ℝ, w = 12 / (w / 2) → w = 12 := by
  sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l1643_164373


namespace NUMINAMATH_CALUDE_fish_ratio_l1643_164399

def fish_problem (O B R : ℕ) : Prop :=
  O = B + 25 ∧
  B = 75 ∧
  (O + B + R) / 3 = 75

theorem fish_ratio : ∀ O B R : ℕ, fish_problem O B R → R * 2 = O :=
sorry

end NUMINAMATH_CALUDE_fish_ratio_l1643_164399


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l1643_164346

theorem cubic_root_sum_product (p q r : ℂ) : 
  (5 * p^3 - 10 * p^2 + 17 * p - 7 = 0) →
  (5 * q^3 - 10 * q^2 + 17 * q - 7 = 0) →
  (5 * r^3 - 10 * r^2 + 17 * r - 7 = 0) →
  p * q + q * r + r * p = 17 / 5 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l1643_164346


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l1643_164381

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_iff_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c for which the lines y = 5x - 3 and y = (3c)x + 1 are parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = 5 * x - 3 ↔ y = (3 * c) * x + 1) ↔ c = 5 / 3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l1643_164381


namespace NUMINAMATH_CALUDE_circle_c_equation_l1643_164342

/-- A circle C with the following properties:
    1. Its center is on the line x - 3y = 0
    2. It is tangent to the negative half-axis of the y-axis
    3. The chord cut by C on the x-axis is 4√2 in length -/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : center.1 - 3 * center.2 = 0
  tangent_to_negative_y : center.2 < 0 ∧ radius = -center.2
  chord_length : 4 * Real.sqrt 2 = 2 * Real.sqrt (2 * radius * center.1)

/-- The equation of circle C is (x + 3)² + (y + 1)² = 9 -/
theorem circle_c_equation (c : CircleC) : 
  ∀ x y : ℝ, (x + 3)^2 + (y + 1)^2 = 9 ↔ 
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_c_equation_l1643_164342


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l1643_164365

theorem equilateral_triangle_area (p : ℝ) (h : p > 0) :
  let perimeter := 3 * p
  let side_length := perimeter / 3
  let area := (Real.sqrt 3 / 4) * side_length ^ 2
  area = (Real.sqrt 3 / 4) * p ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l1643_164365


namespace NUMINAMATH_CALUDE_percentage_of_students_owning_only_cats_l1643_164376

theorem percentage_of_students_owning_only_cats
  (total_students : ℕ)
  (cat_owners : ℕ)
  (dog_owners : ℕ)
  (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 75)
  (h3 : dog_owners = 150)
  (h4 : both_owners = 25) :
  (cat_owners - both_owners) * 100 / total_students = 10 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_students_owning_only_cats_l1643_164376


namespace NUMINAMATH_CALUDE_total_lifting_capacity_is_250_l1643_164382

/-- Calculates the new combined total lifting capacity given initial weights and increases -/
def new_total_lifting_capacity (initial_clean_and_jerk : ℝ) (initial_snatch : ℝ) : ℝ :=
  (2 * initial_clean_and_jerk) + (initial_snatch * 1.8)

/-- Proves that the new combined total lifting capacity is 250 kg -/
theorem total_lifting_capacity_is_250 :
  new_total_lifting_capacity 80 50 = 250 := by
  sorry

end NUMINAMATH_CALUDE_total_lifting_capacity_is_250_l1643_164382


namespace NUMINAMATH_CALUDE_angle_is_90_degrees_l1643_164328

def vector1 : ℝ × ℝ := (4, -3)
def vector2 : ℝ × ℝ := (6, 8)

def angle_between_vectors (v1 v2 : ℝ × ℝ) : ℝ := sorry

theorem angle_is_90_degrees :
  angle_between_vectors vector1 vector2 = 90 := by sorry

end NUMINAMATH_CALUDE_angle_is_90_degrees_l1643_164328


namespace NUMINAMATH_CALUDE_egyptian_fraction_representation_l1643_164353

theorem egyptian_fraction_representation : ∃! (b₂ b₃ b₄ b₅ b₆ b₇ : ℕ),
  (17 : ℚ) / 23 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 11 := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_representation_l1643_164353


namespace NUMINAMATH_CALUDE_pentagon_side_length_l1643_164359

/-- The side length of a regular pentagon with perimeter equal to that of an equilateral triangle with side length 20/9 cm is 4/3 cm. -/
theorem pentagon_side_length (s : ℝ) : 
  (5 * s = 3 * (20 / 9)) → s = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_side_length_l1643_164359


namespace NUMINAMATH_CALUDE_min_points_for_obtuse_triangle_l1643_164308

/-- A color representing red, yellow, or blue -/
inductive Color
  | Red
  | Yellow
  | Blue

/-- A point on the circumference of a circle -/
structure CirclePoint where
  angle : Real
  color : Color

/-- A function that colors every point on the circle's circumference -/
def colorCircle : Real → Color := sorry

/-- Predicate to check if all three colors are present on the circle -/
def allColorsPresent (colorCircle : Real → Color) : Prop := sorry

/-- Predicate to check if three points form an obtuse triangle -/
def isObtuseTriangle (p1 p2 p3 : CirclePoint) : Prop := sorry

/-- The minimum number of points that guarantees an obtuse triangle of the same color -/
def minPointsForObtuseTriangle : Nat := sorry

/-- Theorem stating the minimum number of points required -/
theorem min_points_for_obtuse_triangle :
  ∀ (colorCircle : Real → Color),
    allColorsPresent colorCircle →
    (∀ (points : Finset CirclePoint),
      points.card ≥ minPointsForObtuseTriangle →
      ∃ (p1 p2 p3 : CirclePoint),
        p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
        p1.color = p2.color ∧ p2.color = p3.color ∧
        isObtuseTriangle p1 p2 p3) ∧
    minPointsForObtuseTriangle = 13 :=
by sorry

end NUMINAMATH_CALUDE_min_points_for_obtuse_triangle_l1643_164308


namespace NUMINAMATH_CALUDE_nineteenth_triangular_number_l1643_164396

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

/-- The 19th triangular number is 210 -/
theorem nineteenth_triangular_number : triangular_number 19 = 210 := by
  sorry

end NUMINAMATH_CALUDE_nineteenth_triangular_number_l1643_164396


namespace NUMINAMATH_CALUDE_vector_v_satisfies_conditions_l1643_164362

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Parametric line in 2D space -/
structure Line2D where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

def line_l : Line2D where
  x := λ t => 2 + 2*t
  y := λ t => 3 + t

def line_m : Line2D where
  x := λ s => -3 + 2*s
  y := λ s => 5 + s

def point_A : Point2D :=
  { x := line_l.x 1, y := line_l.y 1 }

def point_B : Point2D :=
  { x := line_m.x 2, y := line_m.y 2 }

def vector_BA : Vector2D :=
  { x := point_B.x - point_A.x, y := point_B.y - point_A.y }

def direction_m : Vector2D :=
  { x := 2, y := 1 }

def perpendicular_m : Vector2D :=
  { x := 1, y := -2 }

def projection (v w : Vector2D) : Vector2D :=
  { x := 0, y := 0 } -- Placeholder definition

theorem vector_v_satisfies_conditions :
  ∃ (v : Vector2D),
    v.x + v.y = 3 ∧
    ∃ (P : Point2D),
      projection vector_BA v = { x := P.x - point_A.x, y := P.y - point_A.y } ∧
      (P.x - point_A.x) * direction_m.x + (P.y - point_A.y) * direction_m.y = 0 ∧
      v = { x := 3, y := -6 } :=
sorry

end NUMINAMATH_CALUDE_vector_v_satisfies_conditions_l1643_164362


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1643_164314

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 2) : y = 2 - 3 * x := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1643_164314


namespace NUMINAMATH_CALUDE_fixed_charge_is_45_l1643_164388

/-- Represents Chris's internet bill structure and usage -/
structure InternetBill where
  fixed_charge : ℝ  -- Fixed monthly charge for 100 GB
  over_charge_per_gb : ℝ  -- Charge per GB over 100 GB limit
  total_bill : ℝ  -- Total bill amount
  gb_over_limit : ℝ  -- Number of GB over the 100 GB limit

/-- Theorem stating that given the conditions, the fixed monthly charge is $45 -/
theorem fixed_charge_is_45 (bill : InternetBill) 
  (h1 : bill.over_charge_per_gb = 0.25)
  (h2 : bill.total_bill = 65)
  (h3 : bill.gb_over_limit = 80) : 
  bill.fixed_charge = 45 := by
  sorry

#check fixed_charge_is_45

end NUMINAMATH_CALUDE_fixed_charge_is_45_l1643_164388


namespace NUMINAMATH_CALUDE_train_length_problem_l1643_164335

/-- The length of two trains passing each other -/
theorem train_length_problem (speed_kmh : ℝ) (crossing_time : ℝ) : 
  speed_kmh = 18 ∧ crossing_time = 24 →
  ∃ (train_length : ℝ), train_length = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l1643_164335


namespace NUMINAMATH_CALUDE_fisherman_pelican_difference_l1643_164336

/-- The number of fish caught by the pelican -/
def pelican_fish : ℕ := 13

/-- The number of fish caught by the kingfisher -/
def kingfisher_fish : ℕ := pelican_fish + 7

/-- The total number of fish caught by the pelican and kingfisher -/
def total_fish : ℕ := pelican_fish + kingfisher_fish

/-- The number of fish caught by the fisherman -/
def fisherman_fish : ℕ := 3 * total_fish

theorem fisherman_pelican_difference :
  fisherman_fish - pelican_fish = 86 := by sorry

end NUMINAMATH_CALUDE_fisherman_pelican_difference_l1643_164336


namespace NUMINAMATH_CALUDE_two_digit_number_division_l1643_164344

theorem two_digit_number_division (x y : ℕ) : 
  (1 ≤ x) ∧ (x ≤ 9) ∧ (0 ≤ y) ∧ (y ≤ 9) →
  (10 * x + y) / (x + y) = 7 ∧ (10 * x + y) % (x + y) = 6 →
  (10 * x + y = 62) ∨ (10 * x + y = 83) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_division_l1643_164344


namespace NUMINAMATH_CALUDE_four_periods_required_l1643_164322

/-- The number of periods required for all students to present their projects -/
def required_periods (num_students : ℕ) (presentation_time : ℕ) (period_length : ℕ) : ℕ :=
  (num_students * presentation_time + period_length - 1) / period_length

/-- Proof that 4 periods are required for the given conditions -/
theorem four_periods_required :
  required_periods 32 5 40 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_periods_required_l1643_164322


namespace NUMINAMATH_CALUDE_constant_water_level_l1643_164354

/-- Represents a water pipe that can fill or empty a tank -/
structure Pipe where
  rate : ℚ  -- Rate of fill/empty (positive for fill, negative for empty)

/-- Represents a water tank system with multiple pipes -/
structure TankSystem where
  pipes : List Pipe

def TankSystem.netRate (system : TankSystem) : ℚ :=
  system.pipes.map (λ p => p.rate) |>.sum

theorem constant_water_level (pipeA pipeB pipeC : Pipe) 
  (hA : pipeA.rate = 1 / 15)
  (hB : pipeB.rate = -1 / 6)  -- Negative because it empties the tank
  (hC : pipeC.rate = 1 / 10) :
  TankSystem.netRate { pipes := [pipeA, pipeB, pipeC] } = 0 := by
  sorry

#check constant_water_level

end NUMINAMATH_CALUDE_constant_water_level_l1643_164354


namespace NUMINAMATH_CALUDE_dot_product_ab_bc_l1643_164309

/-- Given two vectors AB and AC in 2D space, prove that their dot product with BC is -8. -/
theorem dot_product_ab_bc (AB AC : ℝ × ℝ) (h1 : AB = (4, 2)) (h2 : AC = (1, 4)) :
  AB • (AC - AB) = -8 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_ab_bc_l1643_164309


namespace NUMINAMATH_CALUDE_sum_of_22_and_62_l1643_164363

theorem sum_of_22_and_62 : 22 + 62 = 84 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_22_and_62_l1643_164363


namespace NUMINAMATH_CALUDE_root_equation_value_l1643_164377

theorem root_equation_value (m : ℝ) : m^2 - m - 110 = 0 → (m - 1)^2 + m = 111 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l1643_164377


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1643_164315

theorem fixed_point_on_line (k : ℝ) : 1 = k * (-2) + 2 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1643_164315


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1643_164379

theorem largest_digit_divisible_by_six :
  ∀ N : ℕ, N ≤ 9 → (4517 * 10 + N) % 6 = 0 → N ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1643_164379


namespace NUMINAMATH_CALUDE_sum_of_constants_l1643_164305

def polynomial (x : ℝ) : ℝ := x^3 - 7*x^2 + 14*x - 8

def t (k : ℕ) : ℝ := sorry

theorem sum_of_constants (x y z : ℝ) : 
  (∀ k ≥ 2, t (k+1) = x * t k + y * t (k-1) + z * t (k-2)) →
  t 0 = 3 →
  t 1 = 7 →
  t 2 = 15 →
  x + y + z = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_constants_l1643_164305


namespace NUMINAMATH_CALUDE_bird_population_theorem_l1643_164323

theorem bird_population_theorem (total : ℝ) (total_pos : total > 0) : 
  let hawks := 0.3 * total
  let non_hawks := total - hawks
  let paddyfield_warblers := 0.4 * non_hawks
  let kingfishers := 0.25 * paddyfield_warblers
  let other_birds := total - (hawks + paddyfield_warblers + kingfishers)
  (other_birds / total) * 100 = 35 := by
sorry

end NUMINAMATH_CALUDE_bird_population_theorem_l1643_164323


namespace NUMINAMATH_CALUDE_sandy_change_l1643_164306

def football_price : ℚ := 9.14
def baseball_price : ℚ := 6.81
def payment : ℚ := 20

theorem sandy_change : payment - (football_price + baseball_price) = 4.05 := by
  sorry

end NUMINAMATH_CALUDE_sandy_change_l1643_164306


namespace NUMINAMATH_CALUDE_horner_operations_degree_5_l1643_164360

/-- The number of operations required to evaluate a polynomial using Horner's method -/
def horner_operations (degree : ℕ) : ℕ :=
  2 * degree

/-- Theorem: The number of operations to evaluate a polynomial of degree 5 using Horner's method is 10 -/
theorem horner_operations_degree_5 :
  horner_operations 5 = 10 := by
  sorry

#eval horner_operations 5

end NUMINAMATH_CALUDE_horner_operations_degree_5_l1643_164360


namespace NUMINAMATH_CALUDE_sin_130_equals_sin_50_l1643_164371

theorem sin_130_equals_sin_50 : Real.sin (130 * π / 180) = Real.sin (50 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_130_equals_sin_50_l1643_164371


namespace NUMINAMATH_CALUDE_blossom_room_area_l1643_164301

/-- Converts feet and inches to centimeters -/
def to_cm (feet : ℕ) (inches : ℕ) : ℝ :=
  (feet : ℝ) * 30.48 + (inches : ℝ) * 2.54

/-- Calculates the area of a room in square centimeters -/
def room_area (length_feet : ℕ) (length_inches : ℕ) (width_feet : ℕ) (width_inches : ℕ) : ℝ :=
  (to_cm length_feet length_inches) * (to_cm width_feet width_inches)

theorem blossom_room_area :
  room_area 14 8 10 5 = 141935.4 := by
  sorry

end NUMINAMATH_CALUDE_blossom_room_area_l1643_164301


namespace NUMINAMATH_CALUDE_perp_line_plane_from_conditions_l1643_164349

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line_line : Line → Line → Prop)

-- Axiom: If a line is perpendicular to two planes, those planes are parallel
axiom perp_two_planes_parallel (n : Line) (α β : Plane) :
  perp_line_plane n α → perp_line_plane n β → α = β

-- Axiom: If a line is perpendicular to one of two parallel planes, it's perpendicular to the other
axiom perp_parallel_planes (m : Line) (α β : Plane) :
  α = β → perp_line_plane m α → perp_line_plane m β

-- Theorem to prove
theorem perp_line_plane_from_conditions (n m : Line) (α β : Plane) :
  perp_line_plane n α →
  perp_line_plane n β →
  perp_line_plane m α →
  perp_line_plane m β :=
by sorry

end NUMINAMATH_CALUDE_perp_line_plane_from_conditions_l1643_164349


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l1643_164395

/-- The sampling interval for systematic sampling. -/
def sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The sampling interval for a systematic sampling of 30 students
    from a population of 1200 students is 40. -/
theorem systematic_sampling_interval :
  sampling_interval 1200 30 = 40 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l1643_164395


namespace NUMINAMATH_CALUDE_height_difference_l1643_164352

-- Define variables for heights
variable (h_A h_B h_D h_E h_F h_G : ℝ)

-- Define the conditions
def condition1 : Prop := h_A - h_D = 4.5
def condition2 : Prop := h_E - h_D = -1.7
def condition3 : Prop := h_F - h_E = -0.8
def condition4 : Prop := h_G - h_F = 1.9
def condition5 : Prop := h_B - h_G = 3.6

-- Theorem statement
theorem height_difference 
  (c1 : condition1 h_A h_D)
  (c2 : condition2 h_E h_D)
  (c3 : condition3 h_F h_E)
  (c4 : condition4 h_G h_F)
  (c5 : condition5 h_B h_G) :
  h_A > h_B :=
by sorry

end NUMINAMATH_CALUDE_height_difference_l1643_164352


namespace NUMINAMATH_CALUDE_salary_sum_proof_l1643_164330

/-- Given 5 people with an average salary of 8600 and one person's salary of 5000,
    prove that the sum of the other 4 people's salaries is 38000 -/
theorem salary_sum_proof (average_salary : ℕ) (num_people : ℕ) (one_salary : ℕ) 
  (h1 : average_salary = 8600)
  (h2 : num_people = 5)
  (h3 : one_salary = 5000) :
  average_salary * num_people - one_salary = 38000 := by
  sorry

#check salary_sum_proof

end NUMINAMATH_CALUDE_salary_sum_proof_l1643_164330


namespace NUMINAMATH_CALUDE_team_win_percentage_l1643_164368

theorem team_win_percentage (first_games : ℕ) (first_wins : ℕ) (remaining_games : ℕ) (remaining_wins : ℕ) :
  first_games = 50 →
  first_wins = 40 →
  remaining_games = 40 →
  remaining_wins = 23 →
  (first_wins + remaining_wins : ℚ) / (first_games + remaining_games : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_team_win_percentage_l1643_164368


namespace NUMINAMATH_CALUDE_horner_rule_V₁_l1643_164366

-- Define the polynomial coefficients
def a₄ : ℝ := 3
def a₃ : ℝ := 0
def a₂ : ℝ := 2
def a₁ : ℝ := 1
def a₀ : ℝ := 4

-- Define the x value
def x : ℝ := 10

-- Define Horner's Rule first step
def V₀ : ℝ := a₄

-- Define Horner's Rule second step (V₁)
def V₁ : ℝ := V₀ * x + a₃

-- Theorem statement
theorem horner_rule_V₁ : V₁ = 32 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_V₁_l1643_164366


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l1643_164310

def sum_fractions : ℚ :=
  3 + 1/3 + 4 + 1/4 + 5 + 1/6 + 6 + 1/8 + 7 + 1/9

theorem smallest_whole_number_above_sum : 
  ∃ n : ℕ, n = 26 ∧ (∀ m : ℕ, m < n → (m : ℚ) ≤ sum_fractions) ∧ sum_fractions < (n : ℚ) :=
sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l1643_164310


namespace NUMINAMATH_CALUDE_percentage_problem_l1643_164341

theorem percentage_problem (x y : ℝ) (P : ℝ) 
  (h1 : 0.7 * (x - y) = (P / 100) * (x + y))
  (h2 : y = 0.4 * x) : 
  P = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1643_164341


namespace NUMINAMATH_CALUDE_discount_store_purchase_solution_l1643_164331

/-- Represents the purchase scenario at the discount store -/
structure DiscountStorePurchase where
  totalItems : ℕ
  itemsAt9Yuan : ℕ
  totalCost : ℕ

/-- Theorem stating the number of items priced at 9 yuan -/
theorem discount_store_purchase_solution :
  ∀ (purchase : DiscountStorePurchase),
    purchase.totalItems % 2 = 0 ∧
    purchase.totalCost = 172 ∧
    purchase.totalCost = 8 * (purchase.totalItems - purchase.itemsAt9Yuan) + 9 * purchase.itemsAt9Yuan →
    purchase.itemsAt9Yuan = 12 := by
  sorry

end NUMINAMATH_CALUDE_discount_store_purchase_solution_l1643_164331


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1643_164383

/-- A quadratic function with specific properties -/
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ a * (x + 2)^2 + b

/-- The chord length intercepted by the x-axis is 2√3 -/
def chord_length (a b : ℝ) : Prop := ∃ x₁ x₂, x₁ ≠ x₂ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0 ∧ |x₁ - x₂| = 2 * Real.sqrt 3

/-- The function passes through (0, 1) -/
def passes_through_origin (a b : ℝ) : Prop := f a b 0 = 1

/-- The function passes through (-2+√3, 0) -/
def passes_through_intercept (a b : ℝ) : Prop := f a b (-2 + Real.sqrt 3) = 0

theorem quadratic_function_properties :
  ∀ a b : ℝ, chord_length a b → passes_through_origin a b → passes_through_intercept a b →
  (∀ x, f a b x = (x + 2)^2 - 3) ∧
  (∀ k, k < 13/4 → ∃ x ∈ Set.Icc (-1 : ℝ) 1, f a b ((1/2 : ℝ)^x) > k) :=
sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l1643_164383
