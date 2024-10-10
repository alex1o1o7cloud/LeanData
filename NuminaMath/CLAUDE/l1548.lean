import Mathlib

namespace cubic_relation_l1548_154871

theorem cubic_relation (x A B : ℝ) (h1 : x^3 + 1/x^3 = A) (h2 : x - 1/x = B) :
  A / B = B^2 + 3 := by
  sorry

end cubic_relation_l1548_154871


namespace imaginary_part_of_z_l1548_154860

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) :
  z.im = -1/5 := by
  sorry

end imaginary_part_of_z_l1548_154860


namespace mitzel_allowance_left_l1548_154804

/-- Proves that the amount left in Mitzel's allowance is $26, given that she spent 35% of her allowance, which amounts to $14. -/
theorem mitzel_allowance_left (spent_percentage : ℝ) (spent_amount : ℝ) (total_allowance : ℝ) :
  spent_percentage = 0.35 →
  spent_amount = 14 →
  spent_amount = spent_percentage * total_allowance →
  total_allowance - spent_amount = 26 :=
by sorry

end mitzel_allowance_left_l1548_154804


namespace fourth_power_modulo_thirteen_l1548_154837

theorem fourth_power_modulo_thirteen (a d : ℤ) (h_pos : d > 0) 
  (h_div : d ∣ a^4 + a^3 + 2*a^2 - 4*a + 3) : 
  ∃ x : ℤ, d ≡ x^4 [ZMOD 13] := by
sorry

end fourth_power_modulo_thirteen_l1548_154837


namespace nearest_significant_place_is_ten_thousands_l1548_154891

/-- Represents the place value of a digit in a number -/
inductive DigitPlace
  | Hundreds
  | Thousands
  | TenThousands

/-- The given number -/
def givenNumber : ℝ := 3.02 * 10^6

/-- Determines if a digit place is significant for a given number -/
def isSignificantPlace (n : ℝ) (place : DigitPlace) : Prop :=
  match place with
  | DigitPlace.Hundreds => n ≥ 100 ∧ n < 1000
  | DigitPlace.Thousands => n ≥ 1000 ∧ n < 10000
  | DigitPlace.TenThousands => n ≥ 10000 ∧ n < 100000

/-- The nearest significant digit place for the given number -/
def nearestSignificantPlace : DigitPlace := DigitPlace.TenThousands

theorem nearest_significant_place_is_ten_thousands :
  isSignificantPlace givenNumber nearestSignificantPlace :=
by sorry

end nearest_significant_place_is_ten_thousands_l1548_154891


namespace rent_increase_proof_l1548_154888

theorem rent_increase_proof (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (increase_rate : ℝ) 
  (h1 : n = 4)
  (h2 : initial_avg = 800)
  (h3 : new_avg = 870)
  (h4 : increase_rate = 0.2) :
  ∃ (original_rent : ℝ), 
    (n * new_avg - n * initial_avg) / increase_rate = original_rent ∧ 
    original_rent = 1400 := by
  sorry

end rent_increase_proof_l1548_154888


namespace quadratic_rational_roots_l1548_154803

/-- Given positive prime numbers p and q, the equation x^2 + p^2x + q^3 = 0 has rational roots if and only if p = 3 and q = 2 -/
theorem quadratic_rational_roots (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ x : ℚ, x^2 + p^2*x + q^3 = 0) ↔ (p = 3 ∧ q = 2) := by
  sorry

end quadratic_rational_roots_l1548_154803


namespace last_four_average_l1548_154843

theorem last_four_average (total_count : ℕ) (total_avg : ℝ) (first_five_avg : ℝ) (middle_num : ℝ) :
  total_count = 10 →
  total_avg = 210 →
  first_five_avg = 40 →
  middle_num = 1100 →
  (5 * first_five_avg + middle_num + 4 * (total_count * total_avg - 5 * first_five_avg - middle_num) / 4) / total_count = total_avg →
  (total_count * total_avg - 5 * first_five_avg - middle_num) / 4 = 200 := by
sorry

end last_four_average_l1548_154843


namespace power_of_four_l1548_154831

theorem power_of_four (x : ℕ) : 5^29 * 4^x = 2 * 10^29 → x = 15 := by
  sorry

end power_of_four_l1548_154831


namespace sixth_power_sum_l1548_154895

theorem sixth_power_sum (x y a b : ℝ) (h1 : x + y = a) (h2 : x * y = b) :
  x^6 + y^6 = a^6 - 6*a^4*b + 9*a^2*b^2 - 2*b^3 := by
  sorry

end sixth_power_sum_l1548_154895


namespace sqrt_meaningful_range_l1548_154822

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 - x) ↔ x ≤ 2 := by
sorry

end sqrt_meaningful_range_l1548_154822


namespace train_speed_calculation_l1548_154897

/-- Calculates the speed of a train crossing a platform -/
theorem train_speed_calculation (train_length platform_length : Real) 
  (crossing_time : Real) (h1 : train_length = 240) 
  (h2 : platform_length = 240) (h3 : crossing_time = 27) : 
  ∃ (speed : Real), abs (speed - 64) < 0.01 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l1548_154897


namespace car_distance_traveled_l1548_154873

theorem car_distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 23 → time = 3 → distance = speed * time → distance = 69 := by
  sorry

end car_distance_traveled_l1548_154873


namespace square_difference_equality_l1548_154868

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end square_difference_equality_l1548_154868


namespace max_a_value_l1548_154898

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x

-- Define the theorem
theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, |f a (f a x)| ≤ 2) →
  a ≤ (3 + Real.sqrt 17) / 4 :=
by sorry

end max_a_value_l1548_154898


namespace blocks_used_in_susans_structure_l1548_154857

/-- Represents the dimensions and specifications of a rectangular structure --/
structure RectangularStructure where
  length : ℝ
  width : ℝ
  height : ℝ
  floorThickness : ℝ
  wallThickness : ℝ

/-- Calculates the number of one-foot cubical blocks used in the structure --/
def blocksUsed (s : RectangularStructure) : ℝ :=
  let totalVolume := s.length * s.width * s.height
  let internalLength := s.length - 2 * s.wallThickness
  let internalWidth := s.width - 2 * s.wallThickness
  let internalHeight := s.height - 2 * s.floorThickness
  let internalVolume := internalLength * internalWidth * internalHeight
  totalVolume - internalVolume

/-- Theorem stating that the number of blocks used in the given structure is 1068 --/
theorem blocks_used_in_susans_structure :
  let s : RectangularStructure := {
    length := 16,
    width := 12,
    height := 8,
    floorThickness := 2,
    wallThickness := 1.5
  }
  blocksUsed s = 1068 := by
  sorry

#eval blocksUsed {
  length := 16,
  width := 12,
  height := 8,
  floorThickness := 2,
  wallThickness := 1.5
}

end blocks_used_in_susans_structure_l1548_154857


namespace square_difference_equality_l1548_154832

theorem square_difference_equality : 1005^2 - 995^2 - 1004^2 + 996^2 = 4000 := by
  sorry

end square_difference_equality_l1548_154832


namespace parallel_vectors_k_value_l1548_154838

/-- Given two vectors a and b in ℝ², prove that if they are parallel and 
    a = (2, -1) and b = (k, 5/2), then k = -5. -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, -1) →
  b = (k, 5/2) →
  (∃ (t : ℝ), t ≠ 0 ∧ a = t • b) →
  k = -5 := by
  sorry

end parallel_vectors_k_value_l1548_154838


namespace original_triangle_area_l1548_154886

/-- 
Given a triangle whose dimensions are doubled to form a new triangle,
if the area of the new triangle is 32 square feet,
then the area of the original triangle is 8 square feet.
-/
theorem original_triangle_area (original : Real) (new : Real) : 
  (new = 32) → (new = 4 * original) → (original = 8) :=
by
  sorry

end original_triangle_area_l1548_154886


namespace triple_overlap_area_is_six_l1548_154864

/-- Represents a rectangular carpet with width and height in meters -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the layout of carpets in a hall -/
structure CarpetLayout where
  hall_width : ℝ
  hall_height : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area of triple overlap given a carpet layout -/
def tripleOverlapArea (layout : CarpetLayout) : ℝ :=
  sorry

/-- Theorem stating that the area of triple overlap is 6 square meters for the given layout -/
theorem triple_overlap_area_is_six (layout : CarpetLayout) 
  (h1 : layout.hall_width = 10 ∧ layout.hall_height = 10)
  (h2 : layout.carpet1.width = 6 ∧ layout.carpet1.height = 8)
  (h3 : layout.carpet2.width = 6 ∧ layout.carpet2.height = 6)
  (h4 : layout.carpet3.width = 5 ∧ layout.carpet3.height = 7)
  (h5 : ∀ c1 c2 : Carpet, c1 ≠ c2 → ¬ (c1.width + c2.width > layout.hall_width ∨ c1.height + c2.height > layout.hall_height)) :
  tripleOverlapArea layout = 6 := by
  sorry

end triple_overlap_area_is_six_l1548_154864


namespace three_digit_congruence_count_l1548_154852

theorem three_digit_congruence_count : 
  (Finset.filter (fun x => 100 ≤ x ∧ x < 1000 ∧ (2895 * x + 547) % 17 = 1613 % 17) 
    (Finset.range 1000)).card = 53 := by
  sorry

end three_digit_congruence_count_l1548_154852


namespace log_36_2_in_terms_of_a_b_l1548_154823

/-- Given lg 2 = a and lg 3 = b, prove that log_36 2 = (a + b) / b -/
theorem log_36_2_in_terms_of_a_b (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  (Real.log 2) / (Real.log 36) = (a + b) / b := by
  sorry

end log_36_2_in_terms_of_a_b_l1548_154823


namespace large_sphere_radius_twelve_small_to_one_large_l1548_154893

/-- The radius of a single sphere made from the same amount of material as multiple smaller spheres -/
theorem large_sphere_radius (n : ℕ) (r : ℝ) (h : n > 0) :
  (((n : ℝ) * (4 / 3 * Real.pi * r^3)) / (4 / 3 * Real.pi))^(1/3) = n^(1/3) * r :=
by sorry

/-- The radius of a single sphere made from the same amount of material as 12 spheres of radius 0.5 -/
theorem twelve_small_to_one_large :
  (((12 : ℝ) * (4 / 3 * Real.pi * (1/2)^3)) / (4 / 3 * Real.pi))^(1/3) = (3/2)^(1/3) :=
by sorry

end large_sphere_radius_twelve_small_to_one_large_l1548_154893


namespace cubic_equation_root_l1548_154806

theorem cubic_equation_root (a b : ℚ) : 
  ((-2 - 3 * Real.sqrt 3) ^ 3 + a * (-2 - 3 * Real.sqrt 3) ^ 2 + b * (-2 - 3 * Real.sqrt 3) + 54 = 0) → 
  a = 38 / 23 := by
sorry

end cubic_equation_root_l1548_154806


namespace sophie_owes_jordan_l1548_154863

theorem sophie_owes_jordan (price_per_window : ℚ) (windows_cleaned : ℚ) :
  price_per_window = 13/3 →
  windows_cleaned = 8/5 →
  price_per_window * windows_cleaned = 104/15 := by
  sorry

end sophie_owes_jordan_l1548_154863


namespace sum_of_reciprocal_equation_solutions_l1548_154826

theorem sum_of_reciprocal_equation_solutions : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, ∃ j k : ℕ, j > 0 ∧ k > 0 ∧ 1 / j + 1 / k = (1 : ℚ) / 4 ∧ n = j + k) ∧ 
  (∀ j k : ℕ, j > 0 → k > 0 → 1 / j + 1 / k = (1 : ℚ) / 4 → (j + k) ∈ S) ∧
  S.sum id = 59 :=
by sorry

end sum_of_reciprocal_equation_solutions_l1548_154826


namespace sticker_distribution_ways_l1548_154830

def distribute_stickers (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem sticker_distribution_ways :
  distribute_stickers 10 5 = 1001 := by
  sorry

end sticker_distribution_ways_l1548_154830


namespace rhizobia_cultivation_comparison_l1548_154870

structure Rhizobia where
  nitrogen_fixing : Bool
  aerobic : Bool

structure CultureBox where
  sterile : Bool
  gas_introduced : String

structure CultivationResult where
  nitrogen_fixation : ℕ
  colony_size : ℕ

def cultivate (box : CultureBox) (bacteria : Rhizobia) : CultivationResult :=
  sorry

theorem rhizobia_cultivation_comparison 
  (box : CultureBox) 
  (rhizobia : Rhizobia) 
  (h1 : box.sterile = true) 
  (h2 : rhizobia.nitrogen_fixing = true) 
  (h3 : rhizobia.aerobic = true) :
  let n2_result := cultivate { box with gas_introduced := "N₂" } rhizobia
  let air_result := cultivate { box with gas_introduced := "sterile air" } rhizobia
  n2_result.nitrogen_fixation < air_result.nitrogen_fixation ∧ 
  n2_result.colony_size < air_result.colony_size :=
sorry

end rhizobia_cultivation_comparison_l1548_154870


namespace f_properties_l1548_154827

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt (x^2 + 1) + x - 1) / (Real.sqrt (x^2 + 1) + x + 1)

theorem f_properties : 
  (∀ x ≠ 0, f (-x) = -f x) ∧ 
  (∀ x, ∃ y, f x = y) ∧
  (∀ y, f ⁻¹' {y} ≠ ∅ → -1 < y ∧ y < 1) := by
  sorry

end f_properties_l1548_154827


namespace sum_of_smallest_and_largest_prime_l1548_154836

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def primes_between_1_and_50 : Set ℕ := {n : ℕ | 1 < n ∧ n ≤ 50 ∧ is_prime n}

theorem sum_of_smallest_and_largest_prime :
  ∃ (p q : ℕ), p ∈ primes_between_1_and_50 ∧ q ∈ primes_between_1_and_50 ∧
  (∀ r ∈ primes_between_1_and_50, p ≤ r) ∧
  (∀ r ∈ primes_between_1_and_50, r ≤ q) ∧
  p + q = 49 :=
sorry

end sum_of_smallest_and_largest_prime_l1548_154836


namespace monotone_iff_bound_l1548_154845

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

/-- The derivative of f with respect to x -/
def f' (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 4*x + m

/-- f is monotonically increasing -/
def is_monotone_increasing (m : ℝ) : Prop :=
  ∀ x : ℝ, f' m x ≥ 0

theorem monotone_iff_bound (m : ℝ) :
  is_monotone_increasing m ↔ m ≥ 4/3 :=
sorry

end monotone_iff_bound_l1548_154845


namespace pyramid_x_value_l1548_154800

structure Pyramid where
  top : ℕ
  row2_left : ℕ
  row3_left : ℕ
  row4_left : ℕ
  row4_right : ℕ
  row5_left : ℕ
  row5_right : ℕ

def pyramid_sum (a b : ℕ) : ℕ := a + b

def calculate_x (pyr : Pyramid) : ℕ :=
  let p := pyramid_sum pyr.row2_left (pyr.top - pyr.row2_left)
  let q := p - pyr.row3_left
  let r := pyr.row2_left - q
  let s := r - pyr.row4_left
  let t := pyr.row4_left - pyr.row5_left
  s - t

theorem pyramid_x_value (pyr : Pyramid) 
  (h1 : pyr.top = 105)
  (h2 : pyr.row2_left = 47)
  (h3 : pyr.row3_left = 31)
  (h4 : pyr.row4_left = 13)
  (h5 : pyr.row4_right = 9)
  (h6 : pyr.row5_left = 9)
  (h7 : pyr.row5_right = 4) :
  calculate_x pyr = 3 := by
  sorry

end pyramid_x_value_l1548_154800


namespace pasture_consumption_l1548_154883

/-- Represents the pasture scenario with cows and grass -/
structure Pasture where
  /-- Amount of grass each cow eats per day -/
  daily_consumption : ℝ
  /-- Daily growth rate of the grass -/
  daily_growth : ℝ
  /-- Original amount of grass in the pasture -/
  initial_grass : ℝ

/-- Given the conditions, proves that 94 cows will consume all grass in 28 days -/
theorem pasture_consumption (p : Pasture) : 
  (p.initial_grass + 25 * p.daily_growth = 100 * 25 * p.daily_consumption) →
  (p.initial_grass + 35 * p.daily_growth = 84 * 35 * p.daily_consumption) →
  (p.initial_grass + 28 * p.daily_growth = 94 * 28 * p.daily_consumption) :=
by sorry

end pasture_consumption_l1548_154883


namespace complex_and_trig_problem_l1548_154866

/-- Given a complex number z and an angle θ, prove the magnitude of z and a trigonometric expression -/
theorem complex_and_trig_problem (z : ℂ) (θ : ℝ) : 
  θ = 4 * π / 3 →
  (∃ (x y : ℝ), z = x + y * I ∧ x + 3 * y = 0) →
  Complex.abs z = Real.sqrt 21 / 2 ∧
  (2 * Real.cos (θ / 2) ^ 2 - 1) / (Real.sqrt 2 * Real.sin (θ + π / 4)) = 2 / 3 := by
  sorry

end complex_and_trig_problem_l1548_154866


namespace smallest_n_for_integer_sum_l1548_154884

theorem smallest_n_for_integer_sum : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 6 + (1 : ℚ) / n = k) ∧
  (∀ (m : ℕ), m > 0 → m < n → 
    ¬∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 6 + (1 : ℚ) / m = k) ∧
  n = 4 :=
by sorry

end smallest_n_for_integer_sum_l1548_154884


namespace joyce_suitable_land_l1548_154819

/-- Given Joyce's property information, calculate the suitable land for growing vegetables -/
theorem joyce_suitable_land (previous_property : ℝ) (new_property_multiplier : ℝ) (non_arable_land : ℝ) :
  previous_property = 2 ∧ 
  new_property_multiplier = 8 ∧ 
  non_arable_land = 6 →
  previous_property * new_property_multiplier - non_arable_land = 10 := by
  sorry

#check joyce_suitable_land

end joyce_suitable_land_l1548_154819


namespace max_garden_area_optimal_garden_exists_l1548_154865

/-- Represents a rectangular garden with three sides fenced -/
structure Garden where
  length : ℝ
  width : ℝ
  fence_length : ℝ
  fence_constraint : fence_length = length + 2 * width

/-- The area of a rectangular garden -/
def garden_area (g : Garden) : ℝ := g.length * g.width

/-- Theorem stating the maximum area of the garden under given constraints -/
theorem max_garden_area :
  ∀ g : Garden,
  g.fence_length = 160 →
  garden_area g ≤ 3200 ∧
  (garden_area g = 3200 ↔ g.length = 80 ∧ g.width = 40) :=
by sorry

/-- Existence of the optimal garden -/
theorem optimal_garden_exists :
  ∃ g : Garden, g.fence_length = 160 ∧ garden_area g = 3200 ∧ g.length = 80 ∧ g.width = 40 :=
by sorry

end max_garden_area_optimal_garden_exists_l1548_154865


namespace function_properties_l1548_154862

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / 2) * Real.sin (2 * x) - Real.cos (2 * x)

theorem function_properties (a : ℝ) :
  f a (π / 8) = 0 →
  a = Real.sqrt 2 ∧
  (∀ x : ℝ, f a (x + π) = f a x) ∧
  (∀ x : ℝ, f a x ≤ Real.sqrt 2) ∧
  (∃ x : ℝ, f a x = Real.sqrt 2) :=
by sorry

end function_properties_l1548_154862


namespace cloth_coloring_problem_l1548_154816

theorem cloth_coloring_problem (men1 men2 days1 days2 length2 : ℕ) 
  (h1 : men1 = 4)
  (h2 : men2 = 6)
  (h3 : days1 = 2)
  (h4 : days2 = 1)
  (h5 : length2 = 36)
  (h6 : men1 * days2 * length2 = men2 * days1 * length1) :
  length1 = 48 :=
sorry

end cloth_coloring_problem_l1548_154816


namespace spinner_probability_l1548_154821

structure GameBoard where
  regions : Nat
  shaded_regions : Nat
  is_square : Bool
  equal_probability : Bool

def probability_shaded (board : GameBoard) : ℚ :=
  board.shaded_regions / board.regions

theorem spinner_probability (board : GameBoard) 
  (h1 : board.regions = 4)
  (h2 : board.shaded_regions = 3)
  (h3 : board.is_square = true)
  (h4 : board.equal_probability = true) :
  probability_shaded board = 3/4 := by
  sorry

end spinner_probability_l1548_154821


namespace shortest_path_length_l1548_154889

/-- The length of the shortest path from (0,0) to (20,21) avoiding a circle --/
theorem shortest_path_length (start : ℝ × ℝ) (end_point : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) : 
  start = (0, 0) →
  end_point = (20, 21) →
  center = (10, 10.5) →
  radius = 6 →
  ∃ (path_length : ℝ),
    path_length = 26.4 + 2 * Real.pi ∧
    ∀ (other_path : ℝ),
      (∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 ≥ radius^2 → 
        (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (x, y) = (start.1 + t * (end_point.1 - start.1), start.2 + t * (end_point.2 - start.2)))) →
      other_path ≥ path_length :=
by sorry

end shortest_path_length_l1548_154889


namespace rationalize_denominator_l1548_154817

theorem rationalize_denominator : 
  ∃ (A B C D E F G H I : ℤ),
    (1 : ℝ) / (Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 11) = 
      (A * Real.sqrt B + C * Real.sqrt D + E * Real.sqrt F + G * Real.sqrt H) / I ∧
    I > 0 ∧
    A = -6 ∧ B = 5 ∧ C = -8 ∧ D = 3 ∧ E = 3 ∧ F = 11 ∧ G = 1 ∧ H = 165 ∧ I = 51 :=
by sorry

end rationalize_denominator_l1548_154817


namespace quadratic_equation_transformation_l1548_154858

theorem quadratic_equation_transformation (p q : ℝ) :
  (∀ x, 4 * x^2 - p * x + q = 0 ↔ (x - 1/4)^2 = 33/16) →
  q / p = -4 := by
  sorry

end quadratic_equation_transformation_l1548_154858


namespace sum_of_divisible_by_four_l1548_154820

theorem sum_of_divisible_by_four : 
  (Finset.filter (fun n => n > 10 ∧ n < 30 ∧ n % 4 = 0) (Finset.range 30)).sum id = 100 := by
  sorry

end sum_of_divisible_by_four_l1548_154820


namespace probability_three_ones_two_twos_l1548_154869

def roll_probability : ℕ → ℚ
  | 1 => 1/6
  | 2 => 1/6
  | _ => 2/3

def num_rolls : ℕ := 5

theorem probability_three_ones_two_twos :
  (Nat.choose num_rolls 3) * (roll_probability 1)^3 * (roll_probability 2)^2 = 5/3888 := by
  sorry

end probability_three_ones_two_twos_l1548_154869


namespace sin_equality_proof_l1548_154874

theorem sin_equality_proof (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 → 
  Real.sin (n * π / 180) = Real.sin (670 * π / 180) → 
  n = -50 := by sorry

end sin_equality_proof_l1548_154874


namespace digit_sum_of_special_number_l1548_154808

theorem digit_sum_of_special_number (N : ℕ) :
  100 ≤ N ∧ N < 1000 ∧
  N % 10 = 7 ∧
  N % 11 = 7 ∧
  N % 12 = 7 →
  (N / 100) + ((N / 10) % 10) + (N % 10) = 19 := by
sorry

end digit_sum_of_special_number_l1548_154808


namespace cosine_square_expansion_l1548_154829

theorem cosine_square_expansion (z : ℝ) (h : z ≥ 3) :
  (3 - Real.cos (Real.sqrt (z^2 - 9)))^2 = 9 - 6 * Real.cos (Real.sqrt (z^2 - 9)) + (Real.cos (Real.sqrt (z^2 - 9)))^2 := by
  sorry

end cosine_square_expansion_l1548_154829


namespace log_inequality_l1548_154896

/-- Given 0 < a < b < 1 < c, prove that log_b(c) < log_a(c) < a^c -/
theorem log_inequality (a b c : ℝ) (ha : 0 < a) (hab : a < b) (hb1 : b < 1) (hc : 1 < c) :
  Real.log c / Real.log b < Real.log c / Real.log a ∧ Real.log c / Real.log a < a^c := by
  sorry

end log_inequality_l1548_154896


namespace largest_six_digit_with_product_60_l1548_154872

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def digit_product (n : ℕ) : ℕ := (n.digits 10).prod

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

theorem largest_six_digit_with_product_60 :
  ∃ M : ℕ, is_six_digit M ∧ 
           digit_product M = 60 ∧ 
           (∀ n : ℕ, is_six_digit n → digit_product n = 60 → n ≤ M) ∧
           digit_sum M = 15 := by
  sorry

end largest_six_digit_with_product_60_l1548_154872


namespace revenue_unchanged_with_price_increase_l1548_154811

theorem revenue_unchanged_with_price_increase
  (original_price original_demand : ℝ)
  (price_increase : ℝ)
  (demand_decrease : ℝ)
  (h1 : price_increase = 0.3)
  (h2 : demand_decrease = 0.2308) :
  original_price * original_demand ≤
  (original_price * (1 + price_increase)) * (original_demand * (1 - demand_decrease)) :=
by sorry

end revenue_unchanged_with_price_increase_l1548_154811


namespace quadratic_factorization_l1548_154815

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2*a*x + a = a * (x - 1)^2 := by
  sorry

end quadratic_factorization_l1548_154815


namespace exists_bijection_open_to_closed_unit_interval_l1548_154809

open Set Function Real

-- Define the open interval (0, 1)
def open_unit_interval : Set ℝ := Ioo 0 1

-- Define the closed interval [0, 1]
def closed_unit_interval : Set ℝ := Icc 0 1

-- Statement: There exists a bijective function from (0, 1) to [0, 1]
theorem exists_bijection_open_to_closed_unit_interval :
  ∃ f : ℝ → ℝ, Bijective f ∧ (∀ x, x ∈ open_unit_interval ↔ f x ∈ closed_unit_interval) :=
sorry

end exists_bijection_open_to_closed_unit_interval_l1548_154809


namespace necessary_not_sufficient_condition_l1548_154892

theorem necessary_not_sufficient_condition (x : ℝ) :
  (x^2 - x < 0) → (-1 < x ∧ x < 1) ∧
  ∃ y, -1 < y ∧ y < 1 ∧ ¬(y^2 - y < 0) := by
  sorry

end necessary_not_sufficient_condition_l1548_154892


namespace parker_dumbbells_l1548_154881

/-- Given an initial number of dumbbells, the weight of each dumbbell, 
    and the total weight being used, calculate the number of additional dumbbells needed. -/
def additional_dumbbells (initial_count : ℕ) (weight_per_dumbbell : ℕ) (total_weight : ℕ) : ℕ :=
  ((total_weight - initial_count * weight_per_dumbbell) / weight_per_dumbbell)

/-- Theorem stating that given 4 initial dumbbells of 20 pounds each, 
    and a total weight of 120 pounds, the number of additional dumbbells needed is 2. -/
theorem parker_dumbbells : 
  additional_dumbbells 4 20 120 = 2 := by
  sorry

end parker_dumbbells_l1548_154881


namespace negation_of_universal_proposition_l1548_154844

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 < 0) ↔ (∃ x : ℝ, x^2 - x + 2 ≥ 0) :=
by sorry

end negation_of_universal_proposition_l1548_154844


namespace complex_power_modulus_l1548_154824

theorem complex_power_modulus : Complex.abs ((2 - 3 * Complex.I) ^ 5) = 13 ^ (5/2) := by
  sorry

end complex_power_modulus_l1548_154824


namespace pizzas_bought_l1548_154842

theorem pizzas_bought (cost_per_pizza total_paid : ℕ) (h1 : cost_per_pizza = 8) (h2 : total_paid = 24) :
  total_paid / cost_per_pizza = 3 := by
sorry

end pizzas_bought_l1548_154842


namespace min_adventurers_l1548_154814

structure AdventurerGroup where
  R : Finset Nat  -- Adventurers with rubies
  E : Finset Nat  -- Adventurers with emeralds
  S : Finset Nat  -- Adventurers with sapphires
  D : Finset Nat  -- Adventurers with diamonds

def AdventurerGroup.valid (g : AdventurerGroup) : Prop :=
  (g.R.card = 13) ∧
  (g.E.card = 9) ∧
  (g.S.card = 15) ∧
  (g.D.card = 6) ∧
  (∀ x ∈ g.S, (x ∈ g.E ∨ x ∈ g.D) ∧ ¬(x ∈ g.E ∧ x ∈ g.D)) ∧
  (∀ x ∈ g.E, (x ∈ g.R ∨ x ∈ g.S) ∧ ¬(x ∈ g.R ∧ x ∈ g.S))

theorem min_adventurers (g : AdventurerGroup) (h : g.valid) :
  (g.R ∪ g.E ∪ g.S ∪ g.D).card = 22 :=
sorry

end min_adventurers_l1548_154814


namespace video_recorder_wholesale_cost_l1548_154846

theorem video_recorder_wholesale_cost :
  ∀ (wholesale : ℝ),
  let retail := wholesale * 1.20
  let employee_price := retail * 0.75
  employee_price = 180 →
  wholesale = 200 :=
by
  sorry

end video_recorder_wholesale_cost_l1548_154846


namespace person_speed_l1548_154885

/-- The speed of a person crossing a street -/
theorem person_speed (distance : ℝ) (time : ℝ) (h1 : distance = 600) (h2 : time = 5) :
  (distance / 1000) / (time / 60) = 7.2 := by
  sorry

end person_speed_l1548_154885


namespace unique_number_with_properties_l1548_154875

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def all_digits_odd (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d % 2 = 1

theorem unique_number_with_properties : ∃! n : ℕ,
  n < 200 ∧
  all_digits_odd n ∧
  ∃ a b : ℕ, is_two_digit a ∧ is_two_digit b ∧ n = a * b :=
by sorry

end unique_number_with_properties_l1548_154875


namespace subgroup_normal_iff_power_commute_l1548_154833

variables {G : Type*} [Group G]

theorem subgroup_normal_iff_power_commute : 
  (∀ (H : Subgroup G), H.Normal) ↔ 
  (∀ (a b : G), ∃ (m : ℤ), (a * b) ^ m = b * a) :=
by sorry

end subgroup_normal_iff_power_commute_l1548_154833


namespace store_distance_l1548_154841

def walking_speed : ℝ := 2
def running_speed : ℝ := 10
def average_time_minutes : ℝ := 56

theorem store_distance : 
  ∃ (distance : ℝ),
    (distance / walking_speed + distance / running_speed + distance / running_speed) / 3 = average_time_minutes / 60 ∧
    distance = 4 := by
  sorry

end store_distance_l1548_154841


namespace total_selling_price_usd_l1548_154834

/-- Calculate the total selling price in USD for three articles given their purchase prices,
    exchange rates, and profit percentages. -/
theorem total_selling_price_usd 
  (purchase_price_eur : ℝ) (purchase_price_gbp : ℝ) (purchase_price_usd : ℝ)
  (initial_exchange_rate_eur : ℝ) (initial_exchange_rate_gbp : ℝ)
  (new_exchange_rate_eur : ℝ) (new_exchange_rate_gbp : ℝ)
  (profit_percent_1 : ℝ) (profit_percent_2 : ℝ) (profit_percent_3 : ℝ)
  (h1 : purchase_price_eur = 600)
  (h2 : purchase_price_gbp = 450)
  (h3 : purchase_price_usd = 750)
  (h4 : initial_exchange_rate_eur = 1.1)
  (h5 : initial_exchange_rate_gbp = 1.3)
  (h6 : new_exchange_rate_eur = 1.15)
  (h7 : new_exchange_rate_gbp = 1.25)
  (h8 : profit_percent_1 = 0.08)
  (h9 : profit_percent_2 = 0.10)
  (h10 : profit_percent_3 = 0.15) :
  let selling_price_1 := purchase_price_eur * (1 + profit_percent_1) * new_exchange_rate_eur
  let selling_price_2 := purchase_price_gbp * (1 + profit_percent_2) * new_exchange_rate_gbp
  let selling_price_3 := purchase_price_usd * (1 + profit_percent_3)
  selling_price_1 + selling_price_2 + selling_price_3 = 2225.85 := by
  sorry


end total_selling_price_usd_l1548_154834


namespace wedding_catering_calculation_l1548_154807

/-- Calculates the total number of items needed for a wedding reception -/
theorem wedding_catering_calculation 
  (bridgette_guests : ℕ) 
  (alex_guests : ℕ) 
  (extra_plates : ℕ) 
  (tomatoes_per_salad : ℕ) 
  (asparagus_regular : ℕ) 
  (asparagus_large : ℕ) 
  (large_portion_percent : ℚ) 
  (blueberries_per_dessert : ℕ) 
  (raspberries_per_dessert : ℕ) 
  (blackberries_per_dessert : ℕ) 
  (h1 : bridgette_guests = 84)
  (h2 : alex_guests = (2 * bridgette_guests) / 3)
  (h3 : extra_plates = 10)
  (h4 : tomatoes_per_salad = 5)
  (h5 : asparagus_regular = 8)
  (h6 : asparagus_large = 12)
  (h7 : large_portion_percent = 1/10)
  (h8 : blueberries_per_dessert = 15)
  (h9 : raspberries_per_dessert = 8)
  (h10 : blackberries_per_dessert = 10) :
  ∃ (cherry_tomatoes asparagus_spears blueberries raspberries blackberries : ℕ),
    cherry_tomatoes = 750 ∧ 
    asparagus_spears = 1260 ∧ 
    blueberries = 2250 ∧ 
    raspberries = 1200 ∧ 
    blackberries = 1500 := by
  sorry


end wedding_catering_calculation_l1548_154807


namespace log_properties_l1548_154847

theorem log_properties :
  (Real.log 5 + Real.log 2 = 1) ∧
  (Real.log 5 / Real.log 2 = Real.log 5 / Real.log 2) := by
  sorry

end log_properties_l1548_154847


namespace expression_equality_l1548_154848

theorem expression_equality : (-3)^4 + (-3)^3 + 3^3 + 3^4 = 162 := by
  sorry

end expression_equality_l1548_154848


namespace no_solution_iff_p_equals_seven_l1548_154890

theorem no_solution_iff_p_equals_seven :
  ∀ p : ℝ, (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - p) / (x - 8)) ↔ p = 7 := by
  sorry

end no_solution_iff_p_equals_seven_l1548_154890


namespace q_completes_in_four_days_l1548_154851

-- Define the work as a positive real number
variable (W : ℝ) (hW : W > 0)

-- Define the time taken by p and q together
def combined_time : ℝ := 20

-- Define the time p worked alone
def p_alone_time : ℝ := 4

-- Define the total time of work
def total_time : ℝ := 10

-- Define q's time to complete the work alone
def q_alone_time : ℝ := 4

-- Theorem statement
theorem q_completes_in_four_days :
  (W / combined_time + W / q_alone_time) * (total_time - p_alone_time) = W * (1 - p_alone_time / combined_time) :=
sorry

end q_completes_in_four_days_l1548_154851


namespace chord_length_l1548_154894

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 4*y + 6 = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x - y - 5 = 0

-- Theorem statement
theorem chord_length :
  ∃ (chord_length : ℝ),
    (∀ (x y : ℝ), circle_equation x y → line_equation x y → 
      ∃ (x1 y1 x2 y2 : ℝ), 
        circle_equation x1 y1 ∧ circle_equation x2 y2 ∧
        line_equation x1 y1 ∧ line_equation x2 y2 ∧
        (x2 - x1)^2 + (y2 - y1)^2 = chord_length^2) ∧
    chord_length = Real.sqrt 6 :=
by
  sorry

end chord_length_l1548_154894


namespace nested_fraction_equality_l1548_154879

theorem nested_fraction_equality : 
  1 + 4 / (5 + 6 / 7) = 69 / 41 := by sorry

end nested_fraction_equality_l1548_154879


namespace special_function_property_l1548_154840

-- Define a monotonic function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the monotonicity of f
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- Define the special property of f
def SpecialProperty (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ x₁ x₂, f (x * x₁ + x * x₂) = f x + f x₁ + f x₂

theorem special_function_property 
  (h_monotonic : Monotonic f) 
  (h_exists : ∃ x, SpecialProperty f x) :
  (f 1 + f 0 = 0) ∧ (∃ x, SpecialProperty f x ∧ x = 1) :=
by sorry

end special_function_property_l1548_154840


namespace xavier_success_probability_l1548_154810

theorem xavier_success_probability 
  (p_yvonne : ℝ) 
  (p_zelda : ℝ) 
  (p_xavier_and_yvonne_not_zelda : ℝ) 
  (h1 : p_yvonne = 1/2) 
  (h2 : p_zelda = 5/8) 
  (h3 : p_xavier_and_yvonne_not_zelda = 0.0625) :
  ∃ p_xavier : ℝ, 
    p_xavier_and_yvonne_not_zelda = p_xavier * p_yvonne * (1 - p_zelda) ∧ 
    p_xavier = 1/3 :=
sorry

end xavier_success_probability_l1548_154810


namespace not_all_same_graph_l1548_154850

-- Define the three equations
def equation_I (x y : ℝ) : Prop := y = x - 2
def equation_II (x y : ℝ) : Prop := y = (x^2 - 4) / (x + 2)
def equation_III (x y : ℝ) : Prop := (x + 2) * y = x^2 - 4

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq1 x y ↔ eq2 x y

-- Theorem stating that the three equations do not all represent the same graph
theorem not_all_same_graph :
  ¬(same_graph equation_I equation_II ∧ same_graph equation_II equation_III ∧ same_graph equation_I equation_III) :=
by sorry

end not_all_same_graph_l1548_154850


namespace f_extrema_l1548_154812

def f (x : ℝ) := x^2 - 2*x - 1

theorem f_extrema :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-3) 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 2, f x = max) ∧
    (∀ x ∈ Set.Icc (-3) 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3) 2, f x = min) ∧
    max = 14 ∧ min = -2 := by
  sorry

end f_extrema_l1548_154812


namespace square_sum_of_specific_conditions_l1548_154877

theorem square_sum_of_specific_conditions (x y : ℕ+) 
  (h1 : x * y + x + y = 110)
  (h2 : x^2 * y + x * y^2 = 1540) : 
  x^2 + y^2 = 620 := by
sorry

end square_sum_of_specific_conditions_l1548_154877


namespace linear_relationship_scaling_l1548_154861

/-- Given a linear relationship between x and y, prove that if an increase of 4 in x
    corresponds to an increase of 10 in y, then an increase of 12 in x
    will result in an increase of 30 in y. -/
theorem linear_relationship_scaling (f : ℝ → ℝ) (h : ∀ x, f (x + 4) - f x = 10) :
  ∀ x, f (x + 12) - f x = 30 := by
sorry

end linear_relationship_scaling_l1548_154861


namespace sequence_inequality_l1548_154859

theorem sequence_inequality (a : ℕ → ℕ) 
  (h0 : ∀ n, a n > 0)
  (h1 : a 1 > a 0)
  (h2 : ∀ k, 2 ≤ k → k ≤ 100 → a k = 3 * a (k-1) - 2 * a (k-2)) :
  a 100 > 2^99 := by
  sorry

end sequence_inequality_l1548_154859


namespace halloween_costume_payment_l1548_154876

theorem halloween_costume_payment (last_year_cost deposit_percentage price_increase : ℝ) 
  (h1 : last_year_cost = 250)
  (h2 : deposit_percentage = 0.1)
  (h3 : price_increase = 0.4) : 
  (1 + price_increase) * last_year_cost * (1 - deposit_percentage) = 315 :=
by sorry

end halloween_costume_payment_l1548_154876


namespace quadratic_is_square_of_binomial_l1548_154849

theorem quadratic_is_square_of_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 4*x^2 - 12*x + a = (2*x + b)^2) → a = 9 := by
  sorry

end quadratic_is_square_of_binomial_l1548_154849


namespace complex_fraction_simplification_l1548_154801

def complex_i : ℂ := Complex.I

theorem complex_fraction_simplification :
  (1 - complex_i) / (1 + complex_i)^2 = -1/2 - complex_i/2 := by sorry

end complex_fraction_simplification_l1548_154801


namespace max_y_value_l1548_154856

theorem max_y_value (x y : ℤ) (h : x * y + 6 * x + 3 * y = 6) : 
  y ≤ 18 ∧ ∃ (x₀ y₀ : ℤ), x₀ * y₀ + 6 * x₀ + 3 * y₀ = 6 ∧ y₀ = 18 :=
by sorry

end max_y_value_l1548_154856


namespace students_in_neither_subject_l1548_154802

/- Given: -/
def total_students : ℕ := 120
def chemistry_students : ℕ := 75
def biology_students : ℕ := 50
def both_subjects : ℕ := 15

/- Theorem to prove -/
theorem students_in_neither_subject : 
  total_students - (chemistry_students + biology_students - both_subjects) = 10 := by
  sorry

end students_in_neither_subject_l1548_154802


namespace initial_number_proof_l1548_154854

/-- The number that needs to be added to get a multiple of 412 -/
def addend : ℝ := 342.00000000007276

/-- The divisor -/
def divisor : ℕ := 412

/-- The initial number we're trying to find -/
def initial_number : ℝ := 69.99999999992724

theorem initial_number_proof :
  ∃ (n : ℕ), (initial_number + addend : ℝ) = n * divisor :=
by sorry

#check initial_number_proof

end initial_number_proof_l1548_154854


namespace emma_harry_weight_l1548_154825

/-- Given the weights of pairs of students, prove the combined weight of Emma and Harry -/
theorem emma_harry_weight
  (emma_fiona : ℝ)
  (fiona_george : ℝ)
  (george_harry : ℝ)
  (h_emma_fiona : emma_fiona = 280)
  (h_fiona_george : fiona_george = 260)
  (h_george_harry : george_harry = 290) :
  ∃ (emma harry : ℝ),
    emma + harry = 310 ∧
    ∃ (fiona george : ℝ),
      emma + fiona = emma_fiona ∧
      fiona + george = fiona_george ∧
      george + harry = george_harry :=
sorry

end emma_harry_weight_l1548_154825


namespace point_on_hyperbola_l1548_154839

/-- The x-coordinate of point A on the hyperbola y = -4/x with y-coordinate 4 -/
def a : ℝ := sorry

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y = -4 / x

theorem point_on_hyperbola : 
  hyperbola a 4 → a = -1 := by sorry

end point_on_hyperbola_l1548_154839


namespace sum_last_two_digits_fibonacci_factorial_l1548_154813

def fibonacci_factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 5
  | 5 => 8
  | 6 => 13
  | 7 => 21
  | 8 => 34
  | 9 => 55
  | 10 => 89
  | 11 => 144
  | 12 => 233
  | _ => 0

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def last_two_digits (n : ℕ) : ℕ := n % 100

def modified_term (n : ℕ) : ℕ := last_two_digits (factorial (fibonacci_factorial n) + 2)

def sum_last_two_digits (n : ℕ) : ℕ := 
  (List.range n).map modified_term |> List.sum

theorem sum_last_two_digits_fibonacci_factorial : sum_last_two_digits 13 = 14 := by
  sorry

end sum_last_two_digits_fibonacci_factorial_l1548_154813


namespace quadratic_roots_to_coefficients_l1548_154805

theorem quadratic_roots_to_coefficients (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = -2) → 
  b = 1 ∧ c = -2 := by
sorry

end quadratic_roots_to_coefficients_l1548_154805


namespace two_digit_special_number_l1548_154878

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def digit_product (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem two_digit_special_number (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ b = 9 →
  10 * a + b = digit_product (10 * a + b) + digit_sum (10 * a + b) →
  is_prime (digit_sum (10 * a + b)) →
  a = 2 ∨ a = 4 ∨ a = 8 := by sorry

end two_digit_special_number_l1548_154878


namespace perpendicular_lines_imply_a_eq_neg_three_l1548_154899

/-- Two lines are perpendicular if the sum of the products of their coefficients is zero -/
def are_perpendicular (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * a₂ + b₁ * b₂ = 0

/-- The first line: ax + 3y + 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 3 * y + 1 = 0

/-- The second line: 2x + (a+1)y + 1 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  2 * x + (a + 1) * y + 1 = 0

/-- Theorem: If the lines are perpendicular, then a = -3 -/
theorem perpendicular_lines_imply_a_eq_neg_three (a : ℝ) :
  are_perpendicular a 3 2 (a + 1) → a = -3 := by
  sorry

end perpendicular_lines_imply_a_eq_neg_three_l1548_154899


namespace D_72_equals_38_l1548_154887

/-- D(n) represents the number of ways to factor n into integers greater than 1, counting permutations -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem: D(72) equals 38 -/
theorem D_72_equals_38 : D 72 = 38 := by sorry

end D_72_equals_38_l1548_154887


namespace point_slope_problem_l1548_154828

/-- Given two points A(2, b) and B(3, -2) on a line with slope -1, prove that b = -1 -/
theorem point_slope_problem (b : ℝ) : 
  (let A : ℝ × ℝ := (2, b)
   let B : ℝ × ℝ := (3, -2)
   (B.2 - A.2) / (B.1 - A.1) = -1) → b = -1 := by
sorry

end point_slope_problem_l1548_154828


namespace team_a_remaining_days_l1548_154882

/-- The number of days Team A needs to complete the project alone initially -/
def team_a_initial_days : ℚ := 24

/-- The number of days Team B needs to complete the project alone initially -/
def team_b_initial_days : ℚ := 18

/-- The number of days Team A needs to complete the project after receiving 6 people from Team B -/
def team_a_after_transfer_days : ℚ := 18

/-- The number of days Team B needs to complete the project after transferring 6 people to Team A -/
def team_b_after_transfer_days : ℚ := 24

/-- The number of days Team B works alone -/
def team_b_alone_days : ℚ := 6

/-- The number of days both teams work together -/
def teams_together_days : ℚ := 4

/-- The efficiency of one person per day -/
def efficiency_per_person : ℚ := 1 / 432

/-- The number of people in Team A -/
def team_a_people : ℚ := team_a_initial_days / efficiency_per_person

/-- The number of people in Team B -/
def team_b_people : ℚ := team_b_initial_days / efficiency_per_person

/-- The theorem stating that Team A needs 26/3 more days to complete the project -/
theorem team_a_remaining_days : 
  ∃ (m : ℚ), 
    (team_a_people * efficiency_per_person * (team_b_alone_days + teams_together_days) + 
     team_b_people * teams_together_days * efficiency_per_person + 
     team_a_people * m * efficiency_per_person = 1) ∧ 
    m = 26 / 3 := by
  sorry

end team_a_remaining_days_l1548_154882


namespace gwen_recycling_problem_l1548_154835

/-- Represents the problem of calculating unrecycled bags. -/
def unrecycled_bags_problem (total_bags : ℕ) (points_per_bag : ℕ) (total_points : ℕ) : Prop :=
  let recycled_bags := total_points / points_per_bag
  total_bags - recycled_bags = 2

/-- Theorem stating the solution to Gwen's recycling problem. -/
theorem gwen_recycling_problem :
  unrecycled_bags_problem 4 8 16 := by
  sorry

end gwen_recycling_problem_l1548_154835


namespace sally_pens_theorem_l1548_154853

/-- The number of pens Sally takes home -/
def pens_taken_home (initial_pens : ℕ) (num_students : ℕ) (pens_per_student : ℕ) : ℕ :=
  let pens_given := num_students * pens_per_student
  let pens_remaining := initial_pens - pens_given
  pens_remaining / 2

theorem sally_pens_theorem :
  pens_taken_home 342 44 7 = 17 := by
  sorry

#eval pens_taken_home 342 44 7

end sally_pens_theorem_l1548_154853


namespace wire_length_problem_l1548_154880

theorem wire_length_problem (shorter_piece longer_piece total_length : ℝ) :
  shorter_piece = 14 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 49 := by
  sorry

end wire_length_problem_l1548_154880


namespace union_of_sets_l1548_154867

theorem union_of_sets : 
  let M : Set ℕ := {1, 2, 4, 5}
  let N : Set ℕ := {2, 3, 4}
  M ∪ N = {1, 2, 3, 4, 5} := by
sorry

end union_of_sets_l1548_154867


namespace area_triangle_PF1F2_is_sqrt15_l1548_154818

/-- The ellipse with equation x²/9 + y²/5 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 9) + (p.2^2 / 5) = 1}

/-- The foci of the ellipse -/
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

/-- A point on the ellipse satisfying the given condition -/
def P : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of triangle PF₁F₂ is √15 -/
theorem area_triangle_PF1F2_is_sqrt15 
  (h_P_on_ellipse : P ∈ Ellipse)
  (h_PF1_eq_2PF2 : distance P F1 = 2 * distance P F2) :
  (1/2) * distance F1 F2 * Real.sqrt (distance P F1 ^ 2 - (distance F1 F2 / 2) ^ 2) = Real.sqrt 15 := by
  sorry

end area_triangle_PF1F2_is_sqrt15_l1548_154818


namespace abc_order_l1548_154855

theorem abc_order : 
  let a : ℝ := (3/5) ^ (2/5)
  let b : ℝ := (2/5) ^ (3/5)
  let c : ℝ := (2/5) ^ (2/5)
  b < c ∧ c < a := by sorry

end abc_order_l1548_154855
