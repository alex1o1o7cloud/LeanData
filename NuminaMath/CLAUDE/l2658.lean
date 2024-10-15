import Mathlib

namespace NUMINAMATH_CALUDE_august_tips_multiple_l2658_265887

theorem august_tips_multiple (total_months : ℕ) (other_months : ℕ) (august_ratio : ℝ) :
  total_months = 7 →
  other_months = 6 →
  august_ratio = 0.5714285714285714 →
  ∃ (avg_other_months : ℝ),
    avg_other_months > 0 →
    august_ratio * (8 * avg_other_months + other_months * avg_other_months) = 8 * avg_other_months :=
by sorry

end NUMINAMATH_CALUDE_august_tips_multiple_l2658_265887


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2658_265800

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -4 * p.1 + 6}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 5 * p.1 - 3}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(1, 2)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2658_265800


namespace NUMINAMATH_CALUDE_sin_285_degrees_l2658_265834

theorem sin_285_degrees : 
  Real.sin (285 * π / 180) = -((Real.sqrt 6 + Real.sqrt 2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_285_degrees_l2658_265834


namespace NUMINAMATH_CALUDE_worker_b_completion_time_l2658_265803

/-- The time it takes for three workers to complete a job together and individually -/
def JobCompletion (t_together t_a t_b t_c : ℝ) : Prop :=
  (1 / t_together) = (1 / t_a) + (1 / t_b) + (1 / t_c)

/-- Theorem stating that given the conditions, worker B completes the job in 6 days -/
theorem worker_b_completion_time :
  ∀ (t_together : ℝ),
  t_together = 3.428571428571429 →
  JobCompletion t_together 24 6 12 :=
by sorry

end NUMINAMATH_CALUDE_worker_b_completion_time_l2658_265803


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l2658_265897

theorem tripled_base_and_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (3 * a) ^ (3 * b) = a ^ b * x ^ b → x = 27 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l2658_265897


namespace NUMINAMATH_CALUDE_third_stop_off_count_l2658_265851

/-- Represents the number of people on a bus at different stops -/
structure BusOccupancy where
  initial : Nat
  after_first_stop : Nat
  after_second_stop : Nat
  after_third_stop : Nat

/-- Calculates the number of people who got off at the third stop -/
def people_off_third_stop (bus : BusOccupancy) (people_on_third : Nat) : Nat :=
  bus.after_second_stop - bus.after_third_stop + people_on_third

/-- Theorem stating the number of people who got off at the third stop -/
theorem third_stop_off_count (bus : BusOccupancy) 
  (h1 : bus.initial = 50)
  (h2 : bus.after_first_stop = bus.initial - 15)
  (h3 : bus.after_second_stop = bus.after_first_stop - 8 + 2)
  (h4 : bus.after_third_stop = 28)
  (h5 : people_on_third = 3) : 
  people_off_third_stop bus people_on_third = 4 := by
  sorry


end NUMINAMATH_CALUDE_third_stop_off_count_l2658_265851


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l2658_265818

theorem inequality_system_solutions (m : ℝ) : 
  (∃ (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x + 5 > 0 ∧ x - m ≤ 1))) ↔ 
  (-3 ≤ m ∧ m < -2) :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l2658_265818


namespace NUMINAMATH_CALUDE_units_digit_of_1583_pow_1246_l2658_265885

theorem units_digit_of_1583_pow_1246 : ∃ n : ℕ, 1583^1246 ≡ 9 [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_1583_pow_1246_l2658_265885


namespace NUMINAMATH_CALUDE_box_dimensions_l2658_265808

theorem box_dimensions (a b c : ℕ) 
  (h1 : a + c = 17) 
  (h2 : a + b = 13) 
  (h3 : b + c = 20) : 
  (a = 5 ∧ b = 8 ∧ c = 12) ∨ 
  (a = 5 ∧ b = 12 ∧ c = 8) ∨ 
  (a = 8 ∧ b = 5 ∧ c = 12) ∨ 
  (a = 8 ∧ b = 12 ∧ c = 5) ∨ 
  (a = 12 ∧ b = 5 ∧ c = 8) ∨ 
  (a = 12 ∧ b = 8 ∧ c = 5) :=
sorry

end NUMINAMATH_CALUDE_box_dimensions_l2658_265808


namespace NUMINAMATH_CALUDE_smallest_N_value_l2658_265806

/-- Represents a point in the rectangular array -/
structure Point where
  row : Nat
  col : Nat

/-- The original numbering function -/
def original_number (N : Nat) (p : Point) : Nat :=
  (p.row - 1) * N + p.col

/-- The new numbering function -/
def new_number (p : Point) : Nat :=
  5 * (p.col - 1) + p.row

/-- The theorem stating the smallest possible value of N -/
theorem smallest_N_value : ∃ (N : Nat) (p₁ p₂ p₃ p₄ p₅ : Point),
  N > 0 ∧
  p₁.row = 1 ∧ p₂.row = 2 ∧ p₃.row = 3 ∧ p₄.row = 4 ∧ p₅.row = 5 ∧
  p₁.col ≤ N ∧ p₂.col ≤ N ∧ p₃.col ≤ N ∧ p₄.col ≤ N ∧ p₅.col ≤ N ∧
  original_number N p₁ = new_number p₂ ∧
  original_number N p₂ = new_number p₁ ∧
  original_number N p₃ = new_number p₄ ∧
  original_number N p₄ = new_number p₅ ∧
  original_number N p₅ = new_number p₃ ∧
  (∀ (M : Nat) (q₁ q₂ q₃ q₄ q₅ : Point),
    M > 0 ∧
    q₁.row = 1 ∧ q₂.row = 2 ∧ q₃.row = 3 ∧ q₄.row = 4 ∧ q₅.row = 5 ∧
    q₁.col ≤ M ∧ q₂.col ≤ M ∧ q₃.col ≤ M ∧ q₄.col ≤ M ∧ q₅.col ≤ M ∧
    original_number M q₁ = new_number q₂ ∧
    original_number M q₂ = new_number q₁ ∧
    original_number M q₃ = new_number q₄ ∧
    original_number M q₄ = new_number q₅ ∧
    original_number M q₅ = new_number q₃ →
    M ≥ N) ∧
  N = 149 := by
  sorry

end NUMINAMATH_CALUDE_smallest_N_value_l2658_265806


namespace NUMINAMATH_CALUDE_restocking_theorem_l2658_265863

/-- Calculates the amount of ingredients needed to restock --/
def ingredients_to_buy (initial_flour initial_sugar initial_chips : ℕ)
                       (mon_flour mon_sugar mon_chips : ℕ)
                       (tue_flour tue_sugar tue_chips : ℕ)
                       (wed_flour wed_chips : ℕ)
                       (full_flour full_sugar full_chips : ℕ) :
                       (ℕ × ℕ × ℕ) :=
  let remaining_flour := initial_flour - mon_flour - tue_flour
  let spilled_flour := remaining_flour / 2
  let final_flour := if spilled_flour > wed_flour then spilled_flour - wed_flour else 0
  let flour_to_buy := full_flour + (if spilled_flour > wed_flour then 0 else wed_flour - spilled_flour)
  let sugar_to_buy := full_sugar - (initial_sugar - mon_sugar - tue_sugar)
  let chips_to_buy := full_chips - (initial_chips - mon_chips - tue_chips - wed_chips)
  (flour_to_buy, sugar_to_buy, chips_to_buy)

theorem restocking_theorem :
  ingredients_to_buy 500 300 400 150 120 200 240 90 150 100 90 500 300 400 = (545, 210, 440) := by
  sorry

end NUMINAMATH_CALUDE_restocking_theorem_l2658_265863


namespace NUMINAMATH_CALUDE_bells_sync_theorem_l2658_265884

/-- The time in minutes when all bells ring together -/
def bell_sync_time : ℕ := 360

/-- Periods of bell ringing for each institution in minutes -/
def museum_period : ℕ := 18
def library_period : ℕ := 24
def town_hall_period : ℕ := 30
def hospital_period : ℕ := 36

theorem bells_sync_theorem :
  bell_sync_time = Nat.lcm museum_period (Nat.lcm library_period (Nat.lcm town_hall_period hospital_period)) ∧
  bell_sync_time % museum_period = 0 ∧
  bell_sync_time % library_period = 0 ∧
  bell_sync_time % town_hall_period = 0 ∧
  bell_sync_time % hospital_period = 0 :=
by sorry

end NUMINAMATH_CALUDE_bells_sync_theorem_l2658_265884


namespace NUMINAMATH_CALUDE_topsoil_cost_l2658_265881

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards : ℝ := 3

/-- The total cost of topsoil in dollars -/
def total_cost : ℝ := cubic_yards * cubic_feet_per_cubic_yard * cost_per_cubic_foot

theorem topsoil_cost : total_cost = 648 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l2658_265881


namespace NUMINAMATH_CALUDE_rice_A_more_stable_than_B_l2658_265894

/-- Represents a rice variety with its yield variance -/
structure RiceVariety where
  name : String
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines when a rice variety is considered more stable than another -/
def more_stable (a b : RiceVariety) : Prop :=
  a.variance < b.variance

/-- The main theorem stating that rice variety A is more stable than B -/
theorem rice_A_more_stable_than_B (A B : RiceVariety) 
  (hA : A.name = "A" ∧ A.variance = 794)
  (hB : B.name = "B" ∧ B.variance = 958) : 
  more_stable A B := by
  sorry

end NUMINAMATH_CALUDE_rice_A_more_stable_than_B_l2658_265894


namespace NUMINAMATH_CALUDE_sector_central_angle_l2658_265836

/-- Given a sector with circumference 8 and area 4, prove that its central angle is 2 radians -/
theorem sector_central_angle (r : ℝ) (θ : ℝ) 
  (h_circumference : r * θ + 2 * r = 8) 
  (h_area : (1/2) * r^2 * θ = 4) : 
  θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2658_265836


namespace NUMINAMATH_CALUDE_M_properly_contains_N_l2658_265848

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}
def N : Set ℝ := {x : ℝ | ∃ y, y = Real.log (x - 2)}

-- Theorem stating that M properly contains N
theorem M_properly_contains_N : M ⊃ N := by
  sorry

end NUMINAMATH_CALUDE_M_properly_contains_N_l2658_265848


namespace NUMINAMATH_CALUDE_sqrt_22_properties_l2658_265821

theorem sqrt_22_properties (h : 4 < Real.sqrt 22 ∧ Real.sqrt 22 < 5) :
  (∃ (i : ℤ) (d : ℝ), i = 4 ∧ d = Real.sqrt 22 - 4 ∧ Real.sqrt 22 = i + d) ∧
  (∃ (m n : ℝ), 
    m = 7 - Real.sqrt 22 - Int.floor (7 - Real.sqrt 22) ∧
    n = 7 + Real.sqrt 22 - Int.floor (7 + Real.sqrt 22) ∧
    m + n = 1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_22_properties_l2658_265821


namespace NUMINAMATH_CALUDE_shuffleboard_games_l2658_265868

/-- The number of games won by Jerry -/
def jerry_wins : ℕ := 7

/-- The number of games won by Dave -/
def dave_wins : ℕ := jerry_wins + 3

/-- The number of games won by Ken -/
def ken_wins : ℕ := dave_wins + 5

/-- The number of games won by Larry -/
def larry_wins : ℕ := 2 * jerry_wins

/-- The total number of ties -/
def total_ties : ℕ := jerry_wins

/-- The total number of games played -/
def total_games : ℕ := ken_wins + dave_wins + jerry_wins + larry_wins + total_ties

theorem shuffleboard_games :
  (∀ player : ℕ, player ∈ [ken_wins, dave_wins, jerry_wins, larry_wins] → player ≥ 5) →
  total_games = 53 := by
  sorry

end NUMINAMATH_CALUDE_shuffleboard_games_l2658_265868


namespace NUMINAMATH_CALUDE_product_is_2008th_power_l2658_265830

theorem product_is_2008th_power : ∃ (a b c n : ℕ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  b = (a + c) / 2 ∧
  a * b * c = n^2008 := by
sorry

end NUMINAMATH_CALUDE_product_is_2008th_power_l2658_265830


namespace NUMINAMATH_CALUDE_base_conversion_problem_l2658_265825

theorem base_conversion_problem (b : ℕ+) : 
  (b : ℝ)^5 ≤ 125 ∧ 125 < (b : ℝ)^6 ↔ b = 2 :=
sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l2658_265825


namespace NUMINAMATH_CALUDE_item_list_price_l2658_265886

/-- The list price of an item -/
def list_price : ℝ := 40

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Alice's commission rate -/
def alice_rate : ℝ := 0.15

/-- Bob's commission rate -/
def bob_rate : ℝ := 0.25

/-- Alice's commission -/
def alice_commission (x : ℝ) : ℝ := alice_rate * alice_price x

/-- Bob's commission -/
def bob_commission (x : ℝ) : ℝ := bob_rate * bob_price x

theorem item_list_price :
  alice_commission list_price = bob_commission list_price :=
by sorry

end NUMINAMATH_CALUDE_item_list_price_l2658_265886


namespace NUMINAMATH_CALUDE_equation_solution_l2658_265872

theorem equation_solution : ∃ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2658_265872


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2658_265859

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_with_complement : A ∩ (Set.univ \ B) = {1, 5, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2658_265859


namespace NUMINAMATH_CALUDE_set_equality_l2658_265827

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l2658_265827


namespace NUMINAMATH_CALUDE_movie_night_kernels_calculation_l2658_265829

/-- Represents a person's popcorn preference --/
structure PopcornPreference where
  name : String
  cups_wanted : ℚ
  kernel_tablespoons : ℚ
  popcorn_cups : ℚ

/-- Calculates the tablespoons of kernels needed for a given preference --/
def kernels_needed (pref : PopcornPreference) : ℚ :=
  pref.kernel_tablespoons * (pref.cups_wanted / pref.popcorn_cups)

/-- The list of popcorn preferences for the movie night --/
def movie_night_preferences : List PopcornPreference := [
  { name := "Joanie", cups_wanted := 3, kernel_tablespoons := 3, popcorn_cups := 6 },
  { name := "Mitchell", cups_wanted := 4, kernel_tablespoons := 2, popcorn_cups := 4 },
  { name := "Miles and Davis", cups_wanted := 6, kernel_tablespoons := 4, popcorn_cups := 8 },
  { name := "Cliff", cups_wanted := 3, kernel_tablespoons := 1, popcorn_cups := 3 }
]

/-- The total tablespoons of kernels needed for the movie night --/
def total_kernels_needed : ℚ :=
  movie_night_preferences.map kernels_needed |>.sum

theorem movie_night_kernels_calculation :
  total_kernels_needed = 15/2 := by
  sorry

#eval total_kernels_needed

end NUMINAMATH_CALUDE_movie_night_kernels_calculation_l2658_265829


namespace NUMINAMATH_CALUDE_no_mems_are_veens_l2658_265805

universe u

def Mem : Type u := sorry
def En : Type u := sorry
def Veen : Type u := sorry

variable (is_mem : Mem → Prop)
variable (is_en : En → Prop)
variable (is_veen : Veen → Prop)

axiom all_mems_are_ens : ∀ (m : Mem), ∃ (e : En), is_mem m → is_en e
axiom no_ens_are_veens : ¬∃ (e : En) (v : Veen), is_en e ∧ is_veen v

theorem no_mems_are_veens : ¬∃ (m : Mem) (v : Veen), is_mem m ∧ is_veen v := by
  sorry

end NUMINAMATH_CALUDE_no_mems_are_veens_l2658_265805


namespace NUMINAMATH_CALUDE_industrial_lubricants_allocation_l2658_265850

theorem industrial_lubricants_allocation :
  let total_degrees : ℝ := 360
  let total_percentage : ℝ := 100
  let microphotonics : ℝ := 14
  let home_electronics : ℝ := 24
  let food_additives : ℝ := 15
  let genetically_modified_microorganisms : ℝ := 19
  let astrophysics_degrees : ℝ := 72
  let known_sectors := microphotonics + home_electronics + food_additives + genetically_modified_microorganisms
  let astrophysics_percentage := (astrophysics_degrees / total_degrees) * total_percentage
  let industrial_lubricants := total_percentage - known_sectors - astrophysics_percentage
  industrial_lubricants = 8 := by
sorry

end NUMINAMATH_CALUDE_industrial_lubricants_allocation_l2658_265850


namespace NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l2658_265802

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l2658_265802


namespace NUMINAMATH_CALUDE_infinite_divisibility_l2658_265838

theorem infinite_divisibility (p : Nat) (h_prime : Nat.Prime p) (h_mod : p % 4 = 1) (h_not_17 : p ≠ 17) :
  let n := p
  ∃ k : Nat, 3^((n - 2)^(n - 1) - 1) - 1 = 17 * n^2 * k := by
  sorry

end NUMINAMATH_CALUDE_infinite_divisibility_l2658_265838


namespace NUMINAMATH_CALUDE_unique_integer_solution_quadratic_l2658_265893

theorem unique_integer_solution_quadratic :
  ∃! a : ℤ, ∃ x : ℤ, x^2 + a*x + a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_quadratic_l2658_265893


namespace NUMINAMATH_CALUDE_count_lattice_points_l2658_265865

/-- The number of lattice points on the graph of x^2 - y^2 = 36 -/
def lattice_points_count : ℕ := 8

/-- A predicate that checks if a pair of integers satisfies x^2 - y^2 = 36 -/
def satisfies_equation (x y : ℤ) : Prop := x^2 - y^2 = 36

/-- The theorem stating that there are exactly 8 lattice points on the graph of x^2 - y^2 = 36 -/
theorem count_lattice_points :
  (∃! (s : Finset (ℤ × ℤ)), s.card = lattice_points_count ∧ 
    ∀ (p : ℤ × ℤ), p ∈ s ↔ satisfies_equation p.1 p.2) :=
by sorry

#check count_lattice_points

end NUMINAMATH_CALUDE_count_lattice_points_l2658_265865


namespace NUMINAMATH_CALUDE_base4_132_is_30_l2658_265892

/-- Converts a number from base 4 to decimal --/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The decimal representation of 132 in base 4 --/
def m : Nat := base4ToDecimal [2, 3, 1]

theorem base4_132_is_30 : m = 30 := by
  sorry

end NUMINAMATH_CALUDE_base4_132_is_30_l2658_265892


namespace NUMINAMATH_CALUDE_solution_set_f_max_integer_m_max_m_is_two_l2658_265890

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |1 - 2*x|

-- Theorem for part 1
theorem solution_set_f (x : ℝ) :
  f x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 :=
sorry

-- Theorem for part 2
theorem max_integer_m (a b : ℝ) (h1 : 0 < b) (h2 : b < 1/2) (h3 : 1/2 < a) (h4 : f a = 3 * f b) :
  ∀ m : ℤ, (a^2 + b^2 > m) → m ≤ 2 :=
sorry

-- Theorem to prove that 2 is the maximum integer satisfying the condition
theorem max_m_is_two (a b : ℝ) (h1 : 0 < b) (h2 : b < 1/2) (h3 : 1/2 < a) (h4 : f a = 3 * f b) :
  ∃ m : ℤ, (∀ n : ℤ, (a^2 + b^2 > n) → n ≤ m) ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_max_integer_m_max_m_is_two_l2658_265890


namespace NUMINAMATH_CALUDE_find_y_l2658_265833

theorem find_y (x : ℝ) (y : ℝ) (h1 : x^(3*y - 1) = 8) (h2 : x = 2) : y = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2658_265833


namespace NUMINAMATH_CALUDE_max_square_plots_l2658_265845

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available fencing -/
def availableFencing : ℕ := 2500

/-- Calculates the number of square plots given the side length -/
def numPlots (fd : FieldDimensions) (sideLength : ℕ) : ℕ :=
  (fd.width / sideLength) * (fd.length / sideLength)

/-- Calculates the required internal fencing given the side length -/
def requiredFencing (fd : FieldDimensions) (sideLength : ℕ) : ℕ :=
  fd.width * ((fd.length / sideLength) - 1) + fd.length * ((fd.width / sideLength) - 1)

/-- Theorem stating the maximum number of square plots -/
theorem max_square_plots (fd : FieldDimensions) 
    (h1 : fd.width = 30) 
    (h2 : fd.length = 60) : 
    ∃ (sideLength : ℕ), 
      sideLength > 0 ∧ 
      fd.width % sideLength = 0 ∧ 
      fd.length % sideLength = 0 ∧
      requiredFencing fd sideLength ≤ availableFencing ∧
      ∀ (s : ℕ), s > 0 → 
        fd.width % s = 0 → 
        fd.length % s = 0 → 
        requiredFencing fd s ≤ availableFencing → 
        numPlots fd s ≤ numPlots fd sideLength :=
  sorry

#eval numPlots ⟨30, 60⟩ 5  -- Should evaluate to 72

end NUMINAMATH_CALUDE_max_square_plots_l2658_265845


namespace NUMINAMATH_CALUDE_team_selection_count_l2658_265847

/-- The number of ways to select a team of 5 members from a group of 7 boys and 9 girls, 
    with at least 2 boys in the team -/
def select_team (num_boys num_girls : ℕ) : ℕ :=
  (num_boys.choose 2 * num_girls.choose 3) +
  (num_boys.choose 3 * num_girls.choose 2) +
  (num_boys.choose 4 * num_girls.choose 1) +
  (num_boys.choose 5 * num_girls.choose 0)

/-- Theorem stating that the number of ways to select the team is 3360 -/
theorem team_selection_count :
  select_team 7 9 = 3360 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l2658_265847


namespace NUMINAMATH_CALUDE_prob_two_consecutive_accurate_value_l2658_265822

/-- The accuracy rate of the weather forecast for each day -/
def accuracy_rate : ℝ := 0.8

/-- The probability of having at least two consecutive days with accurate forecasts
    out of three days, given the accuracy rate for each day -/
def prob_two_consecutive_accurate (p : ℝ) : ℝ :=
  p^3 + p^2 * (1 - p) + (1 - p) * p^2

/-- Theorem stating that the probability of having at least two consecutive days
    with accurate forecasts out of three days, given an accuracy rate of 0.8,
    is equal to 0.768 -/
theorem prob_two_consecutive_accurate_value :
  prob_two_consecutive_accurate accuracy_rate = 0.768 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_consecutive_accurate_value_l2658_265822


namespace NUMINAMATH_CALUDE_expression_evaluation_l2658_265842

theorem expression_evaluation : (96 / 6) * 3 / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2658_265842


namespace NUMINAMATH_CALUDE_fraction_equality_l2658_265823

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4*x + y) / (x - 4*y) = -3) : 
  (x + 4*y) / (4*x - y) = 39/37 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2658_265823


namespace NUMINAMATH_CALUDE_eighth_grade_girls_count_l2658_265875

theorem eighth_grade_girls_count :
  ∀ (N : ℕ), 
  (N > 0) →
  (∃ (boys girls : ℕ), 
    N = boys + girls ∧
    boys = girls + 1 ∧
    boys = (52 * N) / 100) →
  ∃ (girls : ℕ), girls = 12 :=
by sorry

end NUMINAMATH_CALUDE_eighth_grade_girls_count_l2658_265875


namespace NUMINAMATH_CALUDE_lcm_three_consecutive_naturals_l2658_265811

theorem lcm_three_consecutive_naturals (n : ℕ) :
  let lcm := Nat.lcm (Nat.lcm n (n + 1)) (n + 2)
  lcm = if Even (n + 1) then n * (n + 1) * (n + 2)
        else (n * (n + 1) * (n + 2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_lcm_three_consecutive_naturals_l2658_265811


namespace NUMINAMATH_CALUDE_frozen_fruit_sold_l2658_265891

/-- Given an orchard's fruit sales, calculate the amount of frozen fruit sold. -/
theorem frozen_fruit_sold (total_fruit : ℕ) (fresh_fruit : ℕ) (h1 : total_fruit = 9792) (h2 : fresh_fruit = 6279) :
  total_fruit - fresh_fruit = 3513 := by
  sorry

end NUMINAMATH_CALUDE_frozen_fruit_sold_l2658_265891


namespace NUMINAMATH_CALUDE_max_value_theorem_l2658_265879

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  3 * a * b * Real.sqrt 2 + 6 * b * c ≤ 4.5 ∧
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a'^2 + b'^2 + c'^2 = 1 ∧
    3 * a' * b' * Real.sqrt 2 + 6 * b' * c' = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2658_265879


namespace NUMINAMATH_CALUDE_sum_of_irrationals_not_always_irrational_student_claim_incorrect_l2658_265810

theorem sum_of_irrationals_not_always_irrational :
  ∃ (a b : ℝ), 
    (¬ ∃ (q : ℚ), a = ↑q) ∧ 
    (¬ ∃ (q : ℚ), b = ↑q) ∧ 
    (∃ (q : ℚ), a + b = ↑q) :=
by sorry

-- Given conditions
axiom sqrt_2_irrational : ¬ ∃ (q : ℚ), Real.sqrt 2 = ↑q
axiom sqrt_3_irrational : ¬ ∃ (q : ℚ), Real.sqrt 3 = ↑q
axiom sum_sqrt_2_3_irrational : ¬ ∃ (q : ℚ), Real.sqrt 2 + Real.sqrt 3 = ↑q

-- The statement to be proved
theorem student_claim_incorrect : 
  ¬ (∀ (a b : ℝ), (¬ ∃ (q : ℚ), a = ↑q) → (¬ ∃ (q : ℚ), b = ↑q) → (¬ ∃ (q : ℚ), a + b = ↑q)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_irrationals_not_always_irrational_student_claim_incorrect_l2658_265810


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2658_265895

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 2; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 16384; 0, -8192] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2658_265895


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_three_l2658_265857

theorem missing_digit_divisible_by_three :
  ∃ d : ℕ, d < 10 ∧ (43500 + d * 10 + 1) % 3 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_three_l2658_265857


namespace NUMINAMATH_CALUDE_num_male_students_l2658_265874

/-- Proves the number of male students in an algebra test given certain conditions -/
theorem num_male_students (total_avg : ℝ) (male_avg : ℝ) (female_avg : ℝ) (num_female : ℕ) :
  total_avg = 90 →
  male_avg = 87 →
  female_avg = 92 →
  num_female = 12 →
  ∃ (num_male : ℕ),
    num_male = 8 ∧
    (num_male : ℝ) * male_avg + (num_female : ℝ) * female_avg = (num_male + num_female : ℝ) * total_avg :=
by sorry

end NUMINAMATH_CALUDE_num_male_students_l2658_265874


namespace NUMINAMATH_CALUDE_probability_same_color_is_one_third_l2658_265855

/-- The set of available colors for sportswear -/
inductive Color
  | Red
  | White
  | Blue

/-- The probability of two athletes choosing the same color from three options -/
def probability_same_color : ℚ :=
  1 / 3

/-- Theorem stating that the probability of two athletes choosing the same color is 1/3 -/
theorem probability_same_color_is_one_third :
  probability_same_color = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_is_one_third_l2658_265855


namespace NUMINAMATH_CALUDE_total_boys_in_camp_l2658_265869

theorem total_boys_in_camp (total : ℕ) 
  (h1 : (total : ℚ) * (1 / 5) = (total : ℚ) * (20 / 100))
  (h2 : (total : ℚ) * (1 / 5) * (3 / 10) = (total : ℚ) * (1 / 5) * (30 / 100))
  (h3 : (total : ℚ) * (1 / 5) * (7 / 10) = 77) :
  total = 550 := by
sorry

end NUMINAMATH_CALUDE_total_boys_in_camp_l2658_265869


namespace NUMINAMATH_CALUDE_range_of_a_for_absolute_value_equation_l2658_265860

theorem range_of_a_for_absolute_value_equation (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = a * x + 1) ∧ 
  (∀ y : ℝ, y > 0 → |y| ≠ a * y + 1) → 
  a > -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_absolute_value_equation_l2658_265860


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l2658_265899

theorem largest_solution_of_equation (x : ℝ) :
  (3 * (9 * x^2 + 10 * x + 11) = x * (9 * x - 45)) →
  x ≤ (-1 / 2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l2658_265899


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l2658_265807

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = -3) : 
  Complex.abs (a + b + c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l2658_265807


namespace NUMINAMATH_CALUDE_vasya_numbers_l2658_265837

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vasya_numbers_l2658_265837


namespace NUMINAMATH_CALUDE_quilt_material_requirement_l2658_265889

theorem quilt_material_requirement (material_per_quilt : ℝ) : 
  (7 * material_per_quilt = 21) ∧ (12 * material_per_quilt = 36) :=
by sorry

end NUMINAMATH_CALUDE_quilt_material_requirement_l2658_265889


namespace NUMINAMATH_CALUDE_min_k_for_f_geq_3_solution_set_f_lt_3x_l2658_265819

-- Define the function f(x, k)
def f (x k : ℝ) : ℝ := |x - 3| + |x - 2| + k

-- Theorem for part I
theorem min_k_for_f_geq_3 :
  (∀ x : ℝ, f x 2 ≥ 3) ∧ (∀ k < 2, ∃ x : ℝ, f x k < 3) :=
sorry

-- Theorem for part II
theorem solution_set_f_lt_3x :
  {x : ℝ | f x 1 < 3 * x} = {x : ℝ | x > 6/5} :=
sorry

end NUMINAMATH_CALUDE_min_k_for_f_geq_3_solution_set_f_lt_3x_l2658_265819


namespace NUMINAMATH_CALUDE_congruence_solution_l2658_265824

theorem congruence_solution (n : ℤ) : 11 * 21 ≡ 16 [ZMOD 43] := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l2658_265824


namespace NUMINAMATH_CALUDE_hexagon_coloring_ways_l2658_265831

-- Define the colors
inductive Color
| Red
| Yellow
| Green

-- Define the hexagon grid
def HexagonGrid := List (List Color)

-- Define a function to check if two colors are different
def different_colors (c1 c2 : Color) : Prop :=
  c1 ≠ c2

-- Define a function to check if a coloring is valid
def valid_coloring (grid : HexagonGrid) : Prop :=
  -- Add conditions for valid coloring here
  sorry

-- Define the specific grid pattern with 8 hexagons
def specific_grid_pattern (grid : HexagonGrid) : Prop :=
  -- Add conditions for the specific grid pattern here
  sorry

-- Define the initial conditions (top-left and second-top hexagons are red)
def initial_conditions (grid : HexagonGrid) : Prop :=
  -- Add conditions for initial red hexagons here
  sorry

-- Theorem statement
theorem hexagon_coloring_ways :
  ∀ (grid : HexagonGrid),
    specific_grid_pattern grid →
    initial_conditions grid →
    valid_coloring grid →
    ∃! (n : Nat), n = 2 ∧ 
      ∃ (colorings : List HexagonGrid),
        colorings.length = n ∧
        ∀ c ∈ colorings, 
          specific_grid_pattern c ∧
          initial_conditions c ∧
          valid_coloring c :=
sorry

end NUMINAMATH_CALUDE_hexagon_coloring_ways_l2658_265831


namespace NUMINAMATH_CALUDE_compound_interest_repayment_l2658_265853

-- Define the initial loan amount in yuan
def initial_loan : ℝ := 100000

-- Define the annual interest rate
def interest_rate : ℝ := 0.07

-- Define the repayment function (in ten thousand yuan)
def repayment_amount (years : ℕ) : ℝ :=
  10 * (1 + interest_rate) ^ years

-- Define the total repayment after 5 years (in yuan)
def total_repayment_5_years : ℕ := 140255

-- Define the number of installments
def num_installments : ℕ := 5

-- Define the annual installment amount (in yuan)
def annual_installment : ℕ := 24389

theorem compound_interest_repayment :
  -- 1. Repayment function
  (∀ x : ℕ, repayment_amount x = 10 * (1 + interest_rate) ^ x) ∧
  -- 2. Total repayment after 5 years
  (repayment_amount 5 * 10000 = total_repayment_5_years) ∧
  -- 3. Annual installment calculation
  (annual_installment * (((1 + interest_rate) ^ num_installments - 1) / interest_rate) =
    total_repayment_5_years) :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_repayment_l2658_265853


namespace NUMINAMATH_CALUDE_envelope_addressing_equation_l2658_265814

theorem envelope_addressing_equation (x : ℝ) : x > 0 → (
  let rate1 := 500 / 8
  let rate2 := 500 / x
  let combined_rate := 500 / 2
  rate1 + rate2 = combined_rate
) ↔ 1/8 + 1/x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_envelope_addressing_equation_l2658_265814


namespace NUMINAMATH_CALUDE_candy_distribution_convergence_l2658_265843

/-- Represents the state of candy distribution among students -/
structure CandyState where
  numStudents : Nat
  candies : Fin numStudents → Nat

/-- Represents one round of candy distribution -/
def distributeCandy (state : CandyState) : CandyState :=
  sorry

/-- The teacher gives one candy to students with an odd number of candies -/
def teacherIntervention (state : CandyState) : CandyState :=
  sorry

/-- Checks if all students have the same number of candies -/
def allEqual (state : CandyState) : Bool :=
  sorry

/-- Main theorem: After a finite number of rounds, all students will have the same number of candies -/
theorem candy_distribution_convergence
  (initialState : CandyState)
  (h_even_initial : ∀ i, Even (initialState.candies i)) :
  ∃ n : Nat, allEqual (((teacherIntervention ∘ distributeCandy)^[n]) initialState) = true :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_convergence_l2658_265843


namespace NUMINAMATH_CALUDE_four_fours_l2658_265898

def four_digit_expr : ℕ → Prop :=
  fun n => ∃ (e : ℕ → ℕ → ℕ → ℕ → ℕ),
    (e 4 4 4 4 = n) ∧
    (∀ x y z w, e x y z w = n → x = 4 ∧ y = 4 ∧ z = 4 ∧ w = 4)

theorem four_fours :
  four_digit_expr 3 ∧
  four_digit_expr 4 ∧
  four_digit_expr 5 ∧
  four_digit_expr 6 := by sorry

end NUMINAMATH_CALUDE_four_fours_l2658_265898


namespace NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l2658_265826

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2*k + 1

theorem sum_of_special_primes_is_prime :
  ∀ A B : ℕ,
    is_prime A →
    is_prime B →
    is_prime (A - B) →
    is_prime (A + B) →
    A > B →
    B = 2 →
    is_odd A →
    is_odd (A - B) →
    is_odd (A + B) →
    (∃ k : ℕ, A = (A - B) + 2*k ∧ (A + B) = A + 2*k) →
    is_prime (A + B + (A - B) + B) :=
sorry

end NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l2658_265826


namespace NUMINAMATH_CALUDE_final_water_level_l2658_265873

/-- Represents the water level change in a reservoir over time -/
def waterLevelChange (initialLevel : Real) (riseRate : Real) (fallRate : Real) : Real :=
  let riseTime := 4  -- 8 a.m. to 12 p.m.
  let fallTime := 6  -- 12 p.m. to 6 p.m.
  initialLevel + riseTime * riseRate - fallTime * fallRate

/-- Theorem stating the final water level at 6 p.m. -/
theorem final_water_level (initialLevel : Real) (riseRate : Real) (fallRate : Real) :
  initialLevel = 45 ∧ riseRate = 0.6 ∧ fallRate = 0.3 →
  waterLevelChange initialLevel riseRate fallRate = 45.6 :=
by sorry

end NUMINAMATH_CALUDE_final_water_level_l2658_265873


namespace NUMINAMATH_CALUDE_square_sum_value_l2658_265854

theorem square_sum_value (x y : ℝ) (h1 : x - y = 12) (h2 : x * y = 9) : x^2 + y^2 = 162 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l2658_265854


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2658_265809

theorem sqrt_equation_solution (x : ℝ) (h : x > 0) : Real.sqrt ((3 / x) + 3) = 2 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2658_265809


namespace NUMINAMATH_CALUDE_smallest_a_parabola_l2658_265841

/-- The smallest possible value of 'a' for a parabola with specific conditions -/
theorem smallest_a_parabola : 
  ∀ (a b c : ℝ), 
  (∃ (x y : ℝ), y = a * x^2 + b * x + c ∧ x = 3/2 ∧ y = -1/4) →  -- vertex condition
  (a > 0) →  -- a is positive
  (∃ (n : ℤ), 2*a + b + 3*c = n) →  -- 2a + b + 3c is an integer
  (∀ (a' : ℝ), 
    (∃ (b' c' : ℝ), 
      (∃ (x y : ℝ), y = a' * x^2 + b' * x + c' ∧ x = 3/2 ∧ y = -1/4) ∧
      (a' > 0) ∧
      (∃ (n : ℤ), 2*a' + b' + 3*c' = n)) → 
    a ≤ a') →
  a = 3/23 := by
sorry

end NUMINAMATH_CALUDE_smallest_a_parabola_l2658_265841


namespace NUMINAMATH_CALUDE_line_intersects_both_axes_l2658_265813

/-- A line in the form Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ
  not_both_zero : A ≠ 0 ∨ B ≠ 0

/-- Predicate for a line intersecting both coordinate axes -/
def intersects_both_axes (l : Line) : Prop :=
  ∃ x y : ℝ, (l.A * x + l.C = 0) ∧ (l.B * y + l.C = 0)

/-- Theorem stating the condition for a line to intersect both coordinate axes -/
theorem line_intersects_both_axes (l : Line) : 
  intersects_both_axes l ↔ l.A ≠ 0 ∧ l.B ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_both_axes_l2658_265813


namespace NUMINAMATH_CALUDE_least_positive_angle_theta_l2658_265804

theorem least_positive_angle_theta (θ : Real) : 
  (θ > 0) → 
  (∀ φ, φ > 0 → φ < θ → Real.cos (15 * Real.pi / 180) ≠ Real.sin (35 * Real.pi / 180) + Real.sin φ) → 
  Real.cos (15 * Real.pi / 180) = Real.sin (35 * Real.pi / 180) + Real.sin θ → 
  θ = 55 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_least_positive_angle_theta_l2658_265804


namespace NUMINAMATH_CALUDE_polynomial_equality_l2658_265876

-- Define the polynomial Q
def Q (a b c d : ℝ) (x : ℝ) : ℝ := a + b * x + c * x^2 + d * x^3

-- State the theorem
theorem polynomial_equality (a b c d : ℝ) :
  (Q a b c d (-1) = 2) →
  (∀ x, Q a b c d x = 2 + x^2 - x^3) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2658_265876


namespace NUMINAMATH_CALUDE_rubiks_cube_purchase_l2658_265880

theorem rubiks_cube_purchase (price_A price_B total_cubes max_funding : ℕ)
  (h1 : price_A = 15)
  (h2 : price_B = 22)
  (h3 : total_cubes = 40)
  (h4 : max_funding = 776) :
  ∃ (x : ℕ), x = 15 ∧
    x ≤ total_cubes - x ∧
    x * price_A + (total_cubes - x) * price_B ≤ max_funding ∧
    ∀ (y : ℕ), y < x →
      y > total_cubes - y ∨
      y * price_A + (total_cubes - y) * price_B > max_funding :=
by sorry

end NUMINAMATH_CALUDE_rubiks_cube_purchase_l2658_265880


namespace NUMINAMATH_CALUDE_log_z_w_value_l2658_265844

theorem log_z_w_value (x y z w : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1) (hw : w > 0)
  (hlogx : Real.log w / Real.log x = 24)
  (hlogy : Real.log w / Real.log y = 40)
  (hlogxyz : Real.log w / Real.log (x * y * z) = 12) :
  Real.log w / Real.log z = 60 := by
  sorry

end NUMINAMATH_CALUDE_log_z_w_value_l2658_265844


namespace NUMINAMATH_CALUDE_largest_number_digit_sum_l2658_265839

def digits : Finset ℕ := {5, 6, 4, 7}

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  ∃ (a b c : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  n = 100 * a + 10 * b + c

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_number_digit_sum :
  ∃ (max_n : ℕ), is_valid_number max_n ∧
  ∀ (n : ℕ), is_valid_number n → n ≤ max_n ∧
  digit_sum max_n = 18 :=
sorry

end NUMINAMATH_CALUDE_largest_number_digit_sum_l2658_265839


namespace NUMINAMATH_CALUDE_system_solution_l2658_265846

theorem system_solution (a b c x y z : ℝ) 
  (eq1 : x + y = a) 
  (eq2 : y + z = b) 
  (eq3 : z + x = c) : 
  x = (a + c - b) / 2 ∧ 
  y = (a + b - c) / 2 ∧ 
  z = (b + c - a) / 2 := by
sorry


end NUMINAMATH_CALUDE_system_solution_l2658_265846


namespace NUMINAMATH_CALUDE_helium_lowest_liquefaction_temp_l2658_265866

-- Define the gases
inductive Gas : Type
| Oxygen
| Hydrogen
| Nitrogen
| Helium

-- Define the liquefaction temperature function
def liquefaction_temp : Gas → ℝ
| Gas.Oxygen => -183
| Gas.Hydrogen => -253
| Gas.Nitrogen => -195.8
| Gas.Helium => -268

-- Statement to prove
theorem helium_lowest_liquefaction_temp :
  ∀ g : Gas, liquefaction_temp Gas.Helium ≤ liquefaction_temp g :=
by sorry

end NUMINAMATH_CALUDE_helium_lowest_liquefaction_temp_l2658_265866


namespace NUMINAMATH_CALUDE_nested_radical_inequality_l2658_265870

theorem nested_radical_inequality (x : ℝ) (hx : x > 0) :
  Real.sqrt (2 * x * Real.sqrt ((2 * x + 1) * Real.sqrt ((2 * x + 2) * Real.sqrt (2 * x + 3)))) < (15 * x + 6) / 8 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_inequality_l2658_265870


namespace NUMINAMATH_CALUDE_product_of_numbers_with_sum_and_difference_l2658_265835

theorem product_of_numbers_with_sum_and_difference 
  (x y : ℝ) (sum_eq : x + y = 120) (diff_eq : x - y = 6) : x * y = 3591 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_sum_and_difference_l2658_265835


namespace NUMINAMATH_CALUDE_maze_paths_count_l2658_265888

/-- Represents a branching point in the maze -/
inductive BranchingPoint
  | Major
  | Minor

/-- Represents the structure of the maze -/
structure Maze where
  entrance : Unit
  exit : Unit
  initialChoices : Nat
  majorToMinor : Nat
  minorChoices : Nat

/-- Calculates the number of paths through the maze -/
def numberOfPaths (maze : Maze) : Nat :=
  maze.initialChoices * (maze.minorChoices ^ maze.majorToMinor)

/-- The specific maze from the problem -/
def problemMaze : Maze :=
  { entrance := ()
  , exit := ()
  , initialChoices := 2
  , majorToMinor := 3
  , minorChoices := 2
  }

theorem maze_paths_count :
  numberOfPaths problemMaze = 16 := by
  sorry

#eval numberOfPaths problemMaze

end NUMINAMATH_CALUDE_maze_paths_count_l2658_265888


namespace NUMINAMATH_CALUDE_cartesian_to_polar_coords_l2658_265852

/-- Given a point P with Cartesian coordinates (1, √3), prove that its polar coordinates are (2, π/3) -/
theorem cartesian_to_polar_coords :
  let x : ℝ := 1
  let y : ℝ := Real.sqrt 3
  let ρ : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  ρ = 2 ∧ θ = π / 3 := by sorry

end NUMINAMATH_CALUDE_cartesian_to_polar_coords_l2658_265852


namespace NUMINAMATH_CALUDE_probability_two_heads_in_three_flips_l2658_265801

/-- A fair coin has an equal probability of landing heads or tails. -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The number of coin flips. -/
def num_flips : ℕ := 3

/-- The number of desired heads. -/
def num_heads : ℕ := 2

/-- The probability of getting exactly k successes in n trials with probability p of success on each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1-p)^(n-k)

/-- The main theorem: the probability of getting exactly 2 heads in 3 flips of a fair coin is 0.375. -/
theorem probability_two_heads_in_three_flips (p : ℝ) (h : fair_coin p) :
  binomial_probability num_flips num_heads p = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_heads_in_three_flips_l2658_265801


namespace NUMINAMATH_CALUDE_common_chord_theorem_l2658_265817

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 3*x - 3*y + 3 = 0

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y = 0

/-- The equation of the line containing the common chord -/
def common_chord_line (x y : ℝ) : Prop := x + y - 3 = 0

/-- Theorem stating the equation of the common chord and its length -/
theorem common_chord_theorem :
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → common_chord_line x y) ∧
  (∃ a b c d : ℝ, C₁ a b ∧ C₂ a b ∧ C₁ c d ∧ C₂ c d ∧
    common_chord_line a b ∧ common_chord_line c d ∧
    ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = 6^(1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_theorem_l2658_265817


namespace NUMINAMATH_CALUDE_cubic_km_to_cubic_m_l2658_265849

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

/-- The number of cubic kilometers to convert -/
def cubic_km : ℝ := 5

/-- Theorem stating that 5 cubic kilometers is equal to 5,000,000,000 cubic meters -/
theorem cubic_km_to_cubic_m : 
  cubic_km * (km_to_m ^ 3) = 5000000000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_km_to_cubic_m_l2658_265849


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2658_265815

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 1 * a 4 * a 7 = 27) : 
  a 3 * a 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2658_265815


namespace NUMINAMATH_CALUDE_number_equality_l2658_265828

theorem number_equality (x : ℝ) : (0.4 * x = 0.25 * 80) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l2658_265828


namespace NUMINAMATH_CALUDE_red_balls_estimate_l2658_265816

/-- Estimates the number of red balls in a bag given the total number of balls,
    the number of draws, and the number of red balls drawn. -/
def estimate_red_balls (total_balls : ℕ) (total_draws : ℕ) (red_draws : ℕ) : ℕ :=
  (total_balls * red_draws) / total_draws

/-- Theorem stating that under the given conditions, the estimated number of red balls is 6. -/
theorem red_balls_estimate :
  let total_balls : ℕ := 20
  let total_draws : ℕ := 100
  let red_draws : ℕ := 30
  estimate_red_balls total_balls total_draws red_draws = 6 := by
  sorry

#eval estimate_red_balls 20 100 30

end NUMINAMATH_CALUDE_red_balls_estimate_l2658_265816


namespace NUMINAMATH_CALUDE_carlos_blocks_l2658_265820

theorem carlos_blocks (initial_blocks : ℕ) (given_blocks : ℕ) : 
  initial_blocks = 58 → given_blocks = 21 → initial_blocks - given_blocks = 37 := by
  sorry

end NUMINAMATH_CALUDE_carlos_blocks_l2658_265820


namespace NUMINAMATH_CALUDE_daily_step_goal_l2658_265877

def sunday_steps : ℕ := 9400
def monday_steps : ℕ := 9100
def tuesday_steps : ℕ := 8300
def wednesday_steps : ℕ := 9200
def thursday_steps : ℕ := 8900
def friday_saturday_avg : ℕ := 9050
def days_in_week : ℕ := 7

theorem daily_step_goal :
  (sunday_steps + monday_steps + tuesday_steps + wednesday_steps + thursday_steps + 
   2 * friday_saturday_avg) / days_in_week = 9000 := by
  sorry

end NUMINAMATH_CALUDE_daily_step_goal_l2658_265877


namespace NUMINAMATH_CALUDE_mario_garden_blossoms_l2658_265858

/-- Calculates the total number of blossoms in Mario's garden after a given number of weeks. -/
def total_blossoms (weeks : ℕ) : ℕ :=
  let hibiscus1 := 2 + 3 * weeks
  let hibiscus2 := 4 + 4 * weeks
  let hibiscus3 := 16 + 5 * weeks
  let rose1 := 3 + 2 * weeks
  let rose2 := 5 + 3 * weeks
  hibiscus1 + hibiscus2 + hibiscus3 + rose1 + rose2

/-- Theorem stating that the total number of blossoms in Mario's garden after 2 weeks is 64. -/
theorem mario_garden_blossoms : total_blossoms 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_mario_garden_blossoms_l2658_265858


namespace NUMINAMATH_CALUDE_picture_distribution_l2658_265883

theorem picture_distribution (total : ℕ) (first_album : ℕ) (num_albums : ℕ) :
  total = 35 →
  first_album = 14 →
  num_albums = 3 →
  (total - first_album) % num_albums = 0 →
  (total - first_album) / num_albums = 7 := by
  sorry

end NUMINAMATH_CALUDE_picture_distribution_l2658_265883


namespace NUMINAMATH_CALUDE_kiwi_weight_l2658_265840

theorem kiwi_weight (total_weight : ℝ) (apple_weight : ℝ) (orange_percent : ℝ) (strawberry_kiwi_percent : ℝ) (strawberry_kiwi_ratio : ℝ) :
  total_weight = 400 →
  apple_weight = 80 →
  orange_percent = 0.15 →
  strawberry_kiwi_percent = 0.40 →
  strawberry_kiwi_ratio = 3 / 5 →
  ∃ kiwi_weight : ℝ,
    kiwi_weight = 100 ∧
    kiwi_weight + (strawberry_kiwi_ratio * kiwi_weight) = strawberry_kiwi_percent * total_weight ∧
    kiwi_weight + (strawberry_kiwi_ratio * kiwi_weight) + (orange_percent * total_weight) + apple_weight = total_weight :=
by
  sorry

end NUMINAMATH_CALUDE_kiwi_weight_l2658_265840


namespace NUMINAMATH_CALUDE_president_savings_theorem_l2658_265878

/-- The amount saved by the president for his reelection campaign --/
def president_savings (total_funds friends_percentage family_percentage : ℝ) : ℝ :=
  let friends_contribution := friends_percentage * total_funds
  let remaining_after_friends := total_funds - friends_contribution
  let family_contribution := family_percentage * remaining_after_friends
  total_funds - friends_contribution - family_contribution

/-- Theorem stating the amount saved by the president given the campaign fund conditions --/
theorem president_savings_theorem :
  president_savings 10000 0.4 0.3 = 4200 := by
  sorry

end NUMINAMATH_CALUDE_president_savings_theorem_l2658_265878


namespace NUMINAMATH_CALUDE_notebook_purchase_problem_l2658_265812

theorem notebook_purchase_problem :
  ∀ (price_A price_B : ℝ) (quantity_A quantity_B : ℕ),
  -- Conditions
  (price_B = price_A + 1) →
  (110 / price_A = 120 / price_B) →
  (quantity_A + quantity_B = 100) →
  (quantity_B ≤ 3 * quantity_A) →
  -- Conclusions
  (price_A = 11) ∧
  (price_B = 12) ∧
  (∀ (total_cost : ℝ),
    total_cost = price_A * quantity_A + price_B * quantity_B →
    total_cost ≥ 1100) :=
by sorry

end NUMINAMATH_CALUDE_notebook_purchase_problem_l2658_265812


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2658_265856

theorem cube_volume_problem (a : ℝ) : 
  (a > 0) →  -- Ensure a is positive for a valid cube
  (a^3 - ((a + 1)^2 * (a - 2)) = 10) → 
  (a^3 = 216) := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2658_265856


namespace NUMINAMATH_CALUDE_value_of_a_l2658_265832

/-- Two circles centered at the origin with given properties -/
structure TwoCircles where
  -- Radius of the larger circle
  R : ℝ
  -- Radius of the smaller circle
  r : ℝ
  -- Point P on the larger circle
  P : ℝ × ℝ
  -- Point S on the smaller circle
  S : ℝ → ℝ × ℝ
  -- Distance between Q and R on x-axis
  QR_distance : ℝ
  -- Conditions
  center_origin : True
  P_on_larger : P.1^2 + P.2^2 = R^2
  S_on_smaller : ∀ a, (S a).1^2 + (S a).2^2 = r^2
  S_on_diagonal : ∀ a, (S a).1 = (S a).2
  QR_is_4 : QR_distance = 4
  R_eq_sqrt_104 : R = Real.sqrt 104
  r_eq_R_minus_4 : r = R - 4

/-- The theorem stating the value of a -/
theorem value_of_a (c : TwoCircles) : 
  ∃ a, c.S a = (a, a) ∧ a = Real.sqrt (60 - 4 * Real.sqrt 104) :=
sorry

end NUMINAMATH_CALUDE_value_of_a_l2658_265832


namespace NUMINAMATH_CALUDE_monthly_earnings_calculation_l2658_265871

/-- Proves that a person with given savings and earnings parameters earns a specific monthly amount -/
theorem monthly_earnings_calculation (savings_per_month : ℕ) 
                                     (car_cost : ℕ) 
                                     (total_earnings : ℕ) 
                                     (h1 : savings_per_month = 500)
                                     (h2 : car_cost = 45000)
                                     (h3 : total_earnings = 360000) : 
  (total_earnings / (car_cost / savings_per_month) : ℚ) = 4000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_earnings_calculation_l2658_265871


namespace NUMINAMATH_CALUDE_sum_of_squares_l2658_265862

theorem sum_of_squares (x y z : ℕ+) : 
  (x : ℕ) + y + z = 24 →
  Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10 →
  ∃! s : ℕ, s = x^2 + y^2 + z^2 ∧ s = 296 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2658_265862


namespace NUMINAMATH_CALUDE_find_number_l2658_265882

theorem find_number : ∃ x : ℤ, x + 12 - 27 = 24 ∧ x = 39 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2658_265882


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l2658_265896

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def Triangle.isSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t2.a = k * t1.a ∧
    t2.b = k * t1.b ∧
    t2.c = k * t1.c

theorem similar_triangle_perimeter 
  (t1 : Triangle) 
  (h1 : t1.isIsosceles)
  (h2 : t1.a = 16 ∧ t1.b = 16 ∧ t1.c = 8)
  (t2 : Triangle)
  (h3 : Triangle.isSimilar t1 t2)
  (h4 : min t2.a (min t2.b t2.c) = 40) :
  t2.perimeter = 200 := by
  sorry


end NUMINAMATH_CALUDE_similar_triangle_perimeter_l2658_265896


namespace NUMINAMATH_CALUDE_correct_operation_l2658_265861

theorem correct_operation (x y : ℝ) : 
  (2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y) ∧ 
  (x^3 * x^5 ≠ x^15) ∧ 
  (2 * x + 3 * y ≠ 5 * x * y) ∧ 
  ((x - 2)^2 ≠ x^2 - 4) := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2658_265861


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l2658_265867

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l2658_265867


namespace NUMINAMATH_CALUDE_fraction_simplification_l2658_265864

theorem fraction_simplification (x y z : ℝ) (h : x + 2*y + z ≠ 0) :
  (x^2 + y^2 - 4*z^2 + 2*x*y) / (x^2 + 4*y^2 - z^2 + 2*x*z) = (x + y - 2*z) / (x + z - 2*y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2658_265864
