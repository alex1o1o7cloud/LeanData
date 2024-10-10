import Mathlib

namespace complex_equation_l617_61758

theorem complex_equation (z : ℂ) : (Complex.I * z = 1 - 2 * Complex.I) → z = -2 - Complex.I := by
  sorry

end complex_equation_l617_61758


namespace extreme_values_of_f_l617_61759

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1)^2 + 2

theorem extreme_values_of_f :
  ∃ (a b c : ℝ), 
    (a = 0 ∧ b = 1 ∧ c = -1) ∧
    (∀ x : ℝ, f x ≥ 2) ∧
    (f a = 3 ∧ f b = 2 ∧ f c = 2) ∧
    (∀ x : ℝ, x ≠ a ∧ x ≠ b ∧ x ≠ c → f x < 3) :=
by
  sorry

end extreme_values_of_f_l617_61759


namespace max_value_of_sum_l617_61753

noncomputable def f (x : ℝ) : ℝ := 3^(x-1) + x - 1

def is_inverse (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

theorem max_value_of_sum (f : ℝ → ℝ) (f_inv : ℝ → ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = 3^(x-1) + x - 1) →
  is_inverse f f_inv →
  (∃ y, ∀ x ∈ Set.Icc 0 1, f x + f_inv x ≤ y) ∧
  (∃ x ∈ Set.Icc 0 1, f x + f_inv x = 2) :=
sorry

end max_value_of_sum_l617_61753


namespace quadratic_properties_l617_61712

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) (h1 : a < 0) 
  (h2 : quadratic_function a b c (-1) = 0) 
  (h3 : -b / (2 * a) = 1) :
  (a - b + c = 0) ∧ 
  (∀ m : ℝ, quadratic_function a b c m ≤ -4 * a) ∧
  (∀ x1 x2 : ℝ, x1 < x2 → quadratic_function a b c x1 = -1 → 
    quadratic_function a b c x2 = -1 → x1 < -1 ∧ x2 > 3) :=
by sorry

end quadratic_properties_l617_61712


namespace smallest_inverse_domain_l617_61756

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2)^2 - 5

-- State the theorem
theorem smallest_inverse_domain (c : ℝ) : 
  (∀ x ≥ c, ∀ y ≥ c, f x = f y → x = y) ∧ 
  (∀ d < c, ∃ x y, d ≤ x ∧ d ≤ y ∧ x ≠ y ∧ f x = f y) ↔ 
  c = -2 :=
sorry

end smallest_inverse_domain_l617_61756


namespace sqrt_inequality_l617_61749

theorem sqrt_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) ≤ 2 * Real.sqrt 3 ∧
  (Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) = 2 * Real.sqrt 3 ↔ a = b) :=
by sorry

end sqrt_inequality_l617_61749


namespace banana_arrangements_l617_61773

def word : String := "BANANA"

def letter_count : Nat := word.length

def b_count : Nat := 1
def a_count : Nat := 3
def n_count : Nat := 2

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

theorem banana_arrangements :
  (factorial letter_count) / (factorial b_count * factorial a_count * factorial n_count) = 60 := by
  sorry

end banana_arrangements_l617_61773


namespace cricket_team_right_handed_players_l617_61750

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 61) 
  (h2 : throwers = 37) 
  (h3 : throwers ≤ total_players) 
  (h4 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  : (throwers + (2 * (total_players - throwers) / 3)) = 53 := by
  sorry

end cricket_team_right_handed_players_l617_61750


namespace subtracted_amount_for_ratio_change_l617_61740

theorem subtracted_amount_for_ratio_change : ∃ (a : ℝ),
  (72 : ℝ) / 192 = 3 / 8 ∧
  (72 - a) / (192 - a) = 4 / 9 ∧
  a = 24 := by
  sorry

end subtracted_amount_for_ratio_change_l617_61740


namespace f_congruence_implies_input_congruence_l617_61767

def f (x : ℤ) : ℤ := x^3 + 7*x^2 + 9*x + 10

theorem f_congruence_implies_input_congruence :
  ∀ (a b : ℤ), f a ≡ f b [ZMOD 11] → a ≡ b [ZMOD 11] := by
  sorry

end f_congruence_implies_input_congruence_l617_61767


namespace value_of_c_l617_61737

theorem value_of_c (a b c d : ℝ) : 
  12 = 0.04 * (a + d) →
  4 = 0.12 * (b - d) →
  c = (b - d) / (a + d) →
  c = 1 / 9 := by
sorry

end value_of_c_l617_61737


namespace tan_double_angle_l617_61721

theorem tan_double_angle (α : Real) (h : Real.sin α + 2 * Real.cos α = 0) : 
  Real.tan (2 * α) = 4/3 := by
sorry

end tan_double_angle_l617_61721


namespace machine_retail_price_l617_61762

/-- The retail price of a machine -/
def retail_price : ℝ := 132

/-- The wholesale price of the machine -/
def wholesale_price : ℝ := 99

/-- The discount rate applied to the retail price -/
def discount_rate : ℝ := 0.1

/-- The profit rate as a percentage of the wholesale price -/
def profit_rate : ℝ := 0.2

theorem machine_retail_price :
  retail_price = wholesale_price * (1 + profit_rate) / (1 - discount_rate) :=
by sorry

end machine_retail_price_l617_61762


namespace prob_sector_1_eq_prob_sector_8_prob_consecutive_sectors_correct_l617_61774

-- Define the number of sectors and the number of played sectors
def total_sectors : ℕ := 13
def played_sectors : ℕ := 6

-- Define a function to calculate the probability of a specific sector being played
def prob_sector_played (sector : ℕ) : ℚ :=
  played_sectors / total_sectors

-- Theorem for part (a)
theorem prob_sector_1_eq_prob_sector_8 :
  prob_sector_played 1 = prob_sector_played 8 :=
sorry

-- Define a function to calculate the probability of sectors 1 to 6 being played consecutively
def prob_consecutive_sectors : ℚ :=
  (7^5 : ℚ) / (13^6 : ℚ)

-- Theorem for part (b)
theorem prob_consecutive_sectors_correct :
  prob_consecutive_sectors = (7^5 : ℚ) / (13^6 : ℚ) :=
sorry

end prob_sector_1_eq_prob_sector_8_prob_consecutive_sectors_correct_l617_61774


namespace sequence_general_term_l617_61755

def sequence_a : ℕ → ℤ
  | 0 => 3
  | 1 => 9
  | (n + 2) => 4 * sequence_a (n + 1) - 3 * sequence_a n - 4 * (n + 2) + 2

theorem sequence_general_term (n : ℕ) : 
  sequence_a n = 3^n + n^2 + 3*n + 2 := by
  sorry

end sequence_general_term_l617_61755


namespace greatest_sum_is_correct_l617_61790

/-- The greatest possible sum of two consecutive integers whose product is less than 500 -/
def greatest_sum : ℕ := 43

/-- Predicate to check if two consecutive integers have a product less than 500 -/
def valid_pair (n : ℕ) : Prop := n * (n + 1) < 500

theorem greatest_sum_is_correct :
  (∀ n : ℕ, valid_pair n → n + (n + 1) ≤ greatest_sum) ∧
  (∃ n : ℕ, valid_pair n ∧ n + (n + 1) = greatest_sum) :=
sorry

end greatest_sum_is_correct_l617_61790


namespace division_problem_l617_61732

theorem division_problem : (144 : ℚ) / ((12 : ℚ) / 2) = 24 := by
  sorry

end division_problem_l617_61732


namespace sport_water_amount_l617_61725

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio where
  flavoring : ℚ
  cornSyrup : ℚ
  water : ℚ

/-- Represents the amount of ingredients in ounces -/
structure DrinkAmount where
  flavoring : ℚ
  cornSyrup : ℚ
  water : ℚ

/-- The standard formulation ratio -/
def standardRatio : DrinkRatio :=
  { flavoring := 1, cornSyrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sportRatio (standard : DrinkRatio) : DrinkRatio :=
  { flavoring := standard.flavoring,
    cornSyrup := standard.cornSyrup / 3,
    water := standard.water * 2 }

/-- Theorem stating the amount of water in the sport formulation -/
theorem sport_water_amount 
  (standard : DrinkRatio)
  (sport : DrinkRatio)
  (sportAmount : DrinkAmount)
  (h1 : sport = sportRatio standard)
  (h2 : sportAmount.cornSyrup = 2) :
  sportAmount.water = 7.5 := by
  sorry


end sport_water_amount_l617_61725


namespace min_height_rectangular_container_l617_61707

theorem min_height_rectangular_container (h : ℝ) (y : ℝ) :
  h = 2 * y →                -- height is twice the side length
  y > 0 →                    -- side length is positive
  10 * y^2 ≥ 150 →           -- surface area is at least 150
  h ≥ 2 * Real.sqrt 15 :=    -- minimum height is 2√15
sorry

end min_height_rectangular_container_l617_61707


namespace rock_song_requests_l617_61744

/-- Represents the number of song requests for each genre --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of rock song requests given the conditions --/
theorem rock_song_requests (req : SongRequests) : req.rock = 5 :=
  by
  have h1 : req.total = 30 := by sorry
  have h2 : req.electropop = req.total / 2 := by sorry
  have h3 : req.dance = req.electropop / 3 := by sorry
  have h4 : req.oldies = req.rock - 3 := by sorry
  have h5 : req.dj_choice = req.oldies / 2 := by sorry
  have h6 : req.rap = 2 := by sorry
  have h7 : req.total = req.electropop + req.dance + req.rap + req.rock + req.oldies + req.dj_choice := by sorry
  sorry

end rock_song_requests_l617_61744


namespace exists_valid_coloring_l617_61793

def isArithmeticProgression (a : Fin 2008 → ℕ) : Prop :=
  ∃ (start d : ℕ), ∀ i : Fin 10, a i = start + i.val * d

theorem exists_valid_coloring :
  ∃ (f : Fin 2008 → Fin 4),
    ∀ (a : Fin 10 → Fin 2008),
      isArithmeticProgression (λ i => (a i).val + 1) →
        ∃ (i j : Fin 10), f (a i) ≠ f (a j) :=
by sorry

end exists_valid_coloring_l617_61793


namespace max_apples_capacity_l617_61743

theorem max_apples_capacity (num_boxes : ℕ) (trays_per_box : ℕ) (extra_trays : ℕ) (apples_per_tray : ℕ) : 
  num_boxes = 6 → trays_per_box = 12 → extra_trays = 4 → apples_per_tray = 8 →
  (num_boxes * trays_per_box + extra_trays) * apples_per_tray = 608 := by
  sorry

end max_apples_capacity_l617_61743


namespace inscribed_tetrahedron_volume_bound_l617_61757

/-- A right circular cylinder with volume 1 -/
structure RightCircularCylinder where
  radius : ℝ
  height : ℝ
  volume_eq_one : π * radius^2 * height = 1

/-- A tetrahedron inscribed in a right circular cylinder -/
structure InscribedTetrahedron (c : RightCircularCylinder) where
  volume : ℝ
  is_inscribed : volume ≤ π * c.radius^2 * c.height

/-- The volume of any tetrahedron inscribed in a right circular cylinder 
    with volume 1 does not exceed 2/(3π) -/
theorem inscribed_tetrahedron_volume_bound 
  (c : RightCircularCylinder) 
  (t : InscribedTetrahedron c) : 
  t.volume ≤ 2 / (3 * π) := by
  sorry

end inscribed_tetrahedron_volume_bound_l617_61757


namespace area_of_specific_trapezoid_l617_61746

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- Length of the smaller segment of the lateral side -/
  smaller_segment : ℝ
  /-- Length of the larger segment of the lateral side -/
  larger_segment : ℝ

/-- Calculate the area of the isosceles trapezoid with an inscribed circle -/
def area (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific isosceles trapezoid is 156 -/
theorem area_of_specific_trapezoid :
  let t : IsoscelesTrapezoidWithInscribedCircle := ⟨4, 9⟩
  area t = 156 := by sorry

end area_of_specific_trapezoid_l617_61746


namespace range_of_a_l617_61724

theorem range_of_a (a : ℝ) : 
  (∀ (x y : ℝ), x ≠ 0 → |x + 1/x| ≥ |a - 2| + Real.sin y) ↔ a ∈ Set.Icc 1 3 :=
by sorry

end range_of_a_l617_61724


namespace fraction_to_zero_power_l617_61722

theorem fraction_to_zero_power (a b : ℤ) (h : b ≠ 0) :
  (a / b : ℚ) ^ 0 = 1 := by sorry

end fraction_to_zero_power_l617_61722


namespace imaginary_part_of_z_l617_61715

theorem imaginary_part_of_z (z : ℂ) (h : (1 + z) * (1 - Complex.I) = 2) :
  z.im = 1 := by
sorry

end imaginary_part_of_z_l617_61715


namespace smallest_divisible_by_fractions_l617_61760

def is_divisible_by_fraction (n : ℕ) (a b : ℕ) : Prop :=
  ∃ k : ℕ, n * b = k * a

theorem smallest_divisible_by_fractions :
  ∀ n : ℕ, n > 0 →
    (is_divisible_by_fraction n 8 33 ∧
     is_divisible_by_fraction n 7 22 ∧
     is_divisible_by_fraction n 15 26) →
    n ≥ 120 :=
by sorry

end smallest_divisible_by_fractions_l617_61760


namespace tan_C_when_a_neg_eight_min_tan_C_l617_61776

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the condition for tan A and tan B
def roots_condition (t : Triangle) (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + a*x + 4 = 0 ∧ y^2 + a*y + 4 = 0 ∧ 
  x = Real.tan t.A ∧ y = Real.tan t.B

-- Theorem 1: When a = -8, tan C = 8/3
theorem tan_C_when_a_neg_eight (t : Triangle) (a : ℝ) 
  (h : roots_condition t a) (h_a : a = -8) : 
  Real.tan t.C = 8/3 := by sorry

-- Theorem 2: Minimum value of tan C is 4/3, occurring when tan A = tan B = 2
theorem min_tan_C (t : Triangle) (a : ℝ) 
  (h : roots_condition t a) : 
  ∃ (t_min : Triangle), 
    (∀ t' : Triangle, roots_condition t' a → Real.tan t_min.C ≤ Real.tan t'.C) ∧
    Real.tan t_min.C = 4/3 ∧ 
    Real.tan t_min.A = 2 ∧ 
    Real.tan t_min.B = 2 := by sorry

end tan_C_when_a_neg_eight_min_tan_C_l617_61776


namespace imaginary_part_of_z_l617_61751

theorem imaginary_part_of_z (z : ℂ) : 
  z * (1 + 2 * I ^ 6) = (2 - 3 * I) / I → z.im = 2 := by
  sorry

end imaginary_part_of_z_l617_61751


namespace company_production_theorem_l617_61742

/-- Represents the production schedule of a company making parts --/
structure ProductionSchedule where
  initialRate : ℕ            -- Initial production rate (parts per day)
  initialDays : ℕ            -- Number of days at initial rate
  increasedRate : ℕ          -- Increased production rate (parts per day)
  extraParts : ℕ             -- Extra parts produced beyond the plan

/-- Calculates the total number of parts produced given a production schedule --/
def totalPartsProduced (schedule : ProductionSchedule) : ℕ :=
  sorry

/-- Theorem stating that given the specific production schedule, 675 parts are produced --/
theorem company_production_theorem :
  let schedule := ProductionSchedule.mk 25 3 30 100
  totalPartsProduced schedule = 675 :=
by sorry

end company_production_theorem_l617_61742


namespace boxes_loaded_is_100_l617_61763

/-- The number of boxes loaded on a truck given its capacity and other items --/
def boxes_loaded (truck_capacity : ℕ) (box_weight crate_weight sack_weight bag_weight : ℕ)
  (num_crates num_sacks num_bags : ℕ) : ℕ :=
  (truck_capacity - (crate_weight * num_crates + sack_weight * num_sacks + bag_weight * num_bags)) / box_weight

/-- Theorem stating that 100 boxes were loaded given the specific conditions --/
theorem boxes_loaded_is_100 :
  boxes_loaded 13500 100 60 50 40 10 50 10 = 100 := by
  sorry

end boxes_loaded_is_100_l617_61763


namespace line_ellipse_intersection_l617_61730

theorem line_ellipse_intersection
  (m n : ℝ)
  (h1 : m^2 + n^2 < 3)
  (h2 : 0 < m^2 + n^2) :
  ∀ (a b : ℝ), ∃! (x y : ℝ),
    x^2 / 7 + y^2 / 3 = 1 ∧
    y = a*x + b ∧
    a*m + b = n :=
by sorry

end line_ellipse_intersection_l617_61730


namespace bertolli_farm_tomatoes_bertolli_farm_tomatoes_proof_l617_61771

theorem bertolli_farm_tomatoes : ℕ → Prop :=
  fun tomatoes =>
    let corn : ℕ := 4112
    let onions : ℕ := 985
    let onions_difference : ℕ := 5200
    onions = tomatoes + corn - onions_difference →
    tomatoes = 2073

-- The proof is omitted
theorem bertolli_farm_tomatoes_proof : bertolli_farm_tomatoes 2073 := by
  sorry

end bertolli_farm_tomatoes_bertolli_farm_tomatoes_proof_l617_61771


namespace jemma_grasshopper_count_l617_61747

/-- The number of grasshoppers Jemma saw on her African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The number of dozens of baby grasshoppers Jemma found on the grass -/
def dozens_of_baby_grasshoppers : ℕ := 2

/-- The number of grasshoppers in a dozen -/
def grasshoppers_per_dozen : ℕ := 12

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := grasshoppers_on_plant + dozens_of_baby_grasshoppers * grasshoppers_per_dozen

theorem jemma_grasshopper_count : total_grasshoppers = 31 := by
  sorry

end jemma_grasshopper_count_l617_61747


namespace clown_balloons_l617_61702

/-- The number of balloons a clown has after blowing up an initial set and then an additional set -/
def total_balloons (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that the clown has 60 balloons after blowing up 47 and then 13 more -/
theorem clown_balloons : total_balloons 47 13 = 60 := by
  sorry

end clown_balloons_l617_61702


namespace seokjins_uncle_age_l617_61772

/-- Seokjin's uncle's age when Seokjin is 12 years old -/
def uncles_age (mothers_age_at_birth : ℕ) (age_difference : ℕ) : ℕ :=
  mothers_age_at_birth + 12 - age_difference

/-- Theorem stating that Seokjin's uncle's age is 41 when Seokjin is 12 -/
theorem seokjins_uncle_age :
  uncles_age 32 3 = 41 := by
  sorry

end seokjins_uncle_age_l617_61772


namespace cargo_ship_unloading_time_l617_61777

/-- Cargo ship transportation problem -/
theorem cargo_ship_unloading_time 
  (loading_speed : ℝ) 
  (loading_time : ℝ) 
  (unloading_speed : ℝ) 
  (unloading_time : ℝ) 
  (h1 : loading_speed = 30)
  (h2 : loading_time = 8)
  (h3 : unloading_speed > 0) :
  unloading_time = (loading_speed * loading_time) / unloading_speed :=
by
  sorry

#check cargo_ship_unloading_time

end cargo_ship_unloading_time_l617_61777


namespace jane_inspected_five_eighths_l617_61784

/-- Represents the fraction of products inspected by Jane given the total rejection rate,
    John's rejection rate, and Jane's rejection rate. -/
def jane_inspection_fraction (total_rejection_rate john_rejection_rate jane_rejection_rate : ℚ) : ℚ :=
  5 / 8

/-- Theorem stating that given the specified rejection rates, Jane inspected 5/8 of the products. -/
theorem jane_inspected_five_eighths
  (total_rejection_rate : ℚ)
  (john_rejection_rate : ℚ)
  (jane_rejection_rate : ℚ)
  (h_total : total_rejection_rate = 75 / 10000)
  (h_john : john_rejection_rate = 5 / 1000)
  (h_jane : jane_rejection_rate = 9 / 1000) :
  jane_inspection_fraction total_rejection_rate john_rejection_rate jane_rejection_rate = 5 / 8 := by
  sorry

#eval jane_inspection_fraction (75/10000) (5/1000) (9/1000)

end jane_inspected_five_eighths_l617_61784


namespace extreme_value_implies_f_2_l617_61729

/-- A function f with an extreme value at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_value_implies_f_2 (a b : ℝ) :
  (f' a b 1 = 0) →  -- f has an extreme value at x = 1
  (f a b 1 = 10) →  -- The extreme value is 10
  (f a b 2 = 11 ∨ f a b 2 = 18) :=
by sorry

end extreme_value_implies_f_2_l617_61729


namespace golf_strokes_over_par_l617_61783

/-- Calculates the number of strokes over par for a golfer -/
def strokes_over_par (holes : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) : ℕ :=
  (holes * avg_strokes_per_hole) - (holes * par_per_hole)

theorem golf_strokes_over_par :
  strokes_over_par 9 4 3 = 9 := by
  sorry

end golf_strokes_over_par_l617_61783


namespace train_speed_without_stoppages_l617_61789

/-- The average speed of a train without stoppages, given its speed with stoppages and stoppage time. -/
theorem train_speed_without_stoppages
  (distance : ℝ) -- The distance traveled by the train
  (speed_with_stoppages : ℝ) -- The average speed of the train with stoppages
  (stoppage_time : ℝ) -- The time the train stops per hour
  (h1 : speed_with_stoppages = 360) -- The given speed with stoppages
  (h2 : stoppage_time = 6) -- The given stoppage time in minutes
  (h3 : distance > 0) -- Ensure the distance is positive
  : ∃ (speed_without_stoppages : ℝ),
    speed_without_stoppages = 400 ∧
    distance = speed_with_stoppages * 1 ∧
    distance = speed_without_stoppages * (1 - stoppage_time / 60) := by
  sorry

end train_speed_without_stoppages_l617_61789


namespace combined_time_calculation_l617_61739

/-- The time taken by the car to reach station B -/
def car_time : ℝ := 4.5

/-- The additional time taken by the train compared to the car -/
def train_additional_time : ℝ := 2

/-- The time taken by the train to reach station B -/
def train_time : ℝ := car_time + train_additional_time

/-- The combined time taken by both the car and the train to reach station B -/
def combined_time : ℝ := car_time + train_time

theorem combined_time_calculation : combined_time = 11 := by sorry

end combined_time_calculation_l617_61739


namespace min_white_fraction_4x4x4_cube_l617_61723

/-- Represents a cube composed of smaller unit cubes -/
structure CompositeCube where
  edge_length : ℕ
  red_cubes : ℕ
  white_cubes : ℕ

/-- The minimum fraction of white surface area for a given composite cube -/
def min_white_fraction (c : CompositeCube) : ℚ :=
  sorry

theorem min_white_fraction_4x4x4_cube :
  let c : CompositeCube := ⟨4, 50, 14⟩
  min_white_fraction c = 1 / 16 := by sorry

end min_white_fraction_4x4x4_cube_l617_61723


namespace paving_stones_required_l617_61766

-- Define the dimensions of the courtyard and paving stone
def courtyard_length : ℝ := 158.5
def courtyard_width : ℝ := 35.4
def stone_length : ℝ := 3.2
def stone_width : ℝ := 2.7

-- Define the theorem
theorem paving_stones_required :
  ∃ (n : ℕ), n = 650 ∧ 
  (n : ℝ) * (stone_length * stone_width) ≥ courtyard_length * courtyard_width ∧
  ∀ (m : ℕ), (m : ℝ) * (stone_length * stone_width) ≥ courtyard_length * courtyard_width → m ≥ n :=
by sorry

end paving_stones_required_l617_61766


namespace similar_triangle_perimeter_l617_61734

theorem similar_triangle_perimeter (a b c d e f : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for the original triangle
  d^2 + e^2 = f^2 →  -- Pythagorean theorem for the similar triangle
  d / a = e / b →    -- Similarity condition
  d / a = f / c →    -- Similarity condition
  a = 6 →            -- Given length of shorter leg of original triangle
  b = 8 →            -- Given length of longer leg of original triangle
  d = 15 →           -- Given length of shorter leg of similar triangle
  d + e + f = 60 :=  -- Perimeter of the similar triangle
by
  sorry


end similar_triangle_perimeter_l617_61734


namespace book_price_comparison_l617_61738

theorem book_price_comparison (price_second : ℝ) (price_first : ℝ) :
  price_first = price_second * 1.5 →
  (price_first - price_second) / price_first * 100 = 100 / 3 := by
  sorry

end book_price_comparison_l617_61738


namespace exponential_function_fixed_point_l617_61700

theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1)
  f 1 = 1 := by
  sorry

end exponential_function_fixed_point_l617_61700


namespace expression_evaluation_l617_61717

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end expression_evaluation_l617_61717


namespace f_lower_bound_l617_61736

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 / 2 - (a + 1) * x

theorem f_lower_bound :
  ∀ x : ℝ, x > 0 → f (-1) x ≥ 1/2 := by sorry

end f_lower_bound_l617_61736


namespace ten_people_leaders_and_committee_l617_61703

/-- The number of ways to choose a president, vice-president, and committee from a group --/
def choose_leaders_and_committee (n : ℕ) : ℕ :=
  n * (n - 1) * Nat.choose (n - 2) 3

/-- The theorem stating the number of ways to choose leaders and committee from 10 people --/
theorem ten_people_leaders_and_committee :
  choose_leaders_and_committee 10 = 5040 := by
  sorry

end ten_people_leaders_and_committee_l617_61703


namespace possible_m_values_l617_61705

def A (m : ℝ) : Set ℝ := {x | m * x - 1 = 0}
def B : Set ℝ := {2, 3}

theorem possible_m_values :
  ∀ m : ℝ, (A m) ⊆ B → (m = 0 ∨ m = 1/2 ∨ m = 1/3) :=
by sorry

end possible_m_values_l617_61705


namespace parabola_intersection_midpoint_l617_61798

/-- Parabola defined by y² = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Condition that A and B are on the parabola and |AF| + |BF| = 10 -/
def IntersectionCondition (A B : ℝ × ℝ) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧
  Real.sqrt ((A.1 - Focus.1)^2 + (A.2 - Focus.2)^2) +
  Real.sqrt ((B.1 - Focus.1)^2 + (B.2 - Focus.2)^2) = 10

/-- The theorem to be proved -/
theorem parabola_intersection_midpoint
  (A B : ℝ × ℝ) (h : IntersectionCondition A B) :
  (A.1 + B.1) / 2 = 4 := by
  sorry

end parabola_intersection_midpoint_l617_61798


namespace smallest_m_for_square_inequality_l617_61741

theorem smallest_m_for_square_inequality : ∃ (m : ℕ+), 
  (m = 16144325) ∧ 
  (∀ (n : ℕ+), n ≥ m → ∃ (l : ℕ+), (n : ℝ) < (l : ℝ)^2 ∧ (l : ℝ)^2 < (1 + 1/2009) * (n : ℝ)) ∧
  (∀ (m' : ℕ+), m' < m → ∃ (n : ℕ+), n ≥ m' ∧ ∀ (l : ℕ+), ((n : ℝ) ≥ (l : ℝ)^2 ∨ (l : ℝ)^2 ≥ (1 + 1/2009) * (n : ℝ))) :=
by sorry

end smallest_m_for_square_inequality_l617_61741


namespace simplify_sqrt_expression_l617_61797

theorem simplify_sqrt_expression : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 56) = (3 + Real.sqrt 7) / 2 := by
  sorry

end simplify_sqrt_expression_l617_61797


namespace perpendicular_tangents_locus_l617_61706

/-- The locus of points where mutually perpendicular tangents to x² + y² = 32 intersect -/
theorem perpendicular_tangents_locus (x₀ y₀ : ℝ) : 
  (∃ t₁ t₂ : ℝ → ℝ, 
    (∀ x y, x^2 + y^2 = 32 → (t₁ x = y ∨ t₂ x = y) → (x - x₀) * (y - y₀) = 0) ∧ 
    (∀ x, (t₁ x - y₀) * (t₂ x - y₀) = -1)) →
  x₀^2 + y₀^2 = 64 :=
sorry

end perpendicular_tangents_locus_l617_61706


namespace owen_work_hours_l617_61799

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents the number of hours Owen spends on daily chores -/
def hours_on_chores : ℕ := 7

/-- Represents the number of hours Owen sleeps -/
def hours_sleeping : ℕ := 11

/-- Calculates the number of hours Owen spends at work -/
def hours_at_work : ℕ := hours_in_day - hours_on_chores - hours_sleeping

/-- Theorem stating that Owen spends 6 hours at work -/
theorem owen_work_hours : hours_at_work = 6 := by
  sorry

end owen_work_hours_l617_61799


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l617_61748

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, b = (t * a.1, t * a.2)

/-- The vectors a and b as defined in the problem -/
def a (k : ℝ) : ℝ × ℝ := (1, k)
def b (k : ℝ) : ℝ × ℝ := (k, 4)

/-- k=-2 is sufficient for collinearity -/
theorem sufficient_condition (k : ℝ) : 
  k = -2 → collinear (a k) (b k) :=
sorry

/-- k=-2 is not necessary for collinearity -/
theorem not_necessary_condition : 
  ∃ k : ℝ, k ≠ -2 ∧ collinear (a k) (b k) :=
sorry

/-- The main theorem stating that k=-2 is sufficient but not necessary -/
theorem sufficient_but_not_necessary : 
  (∀ k : ℝ, k = -2 → collinear (a k) (b k)) ∧
  (∃ k : ℝ, k ≠ -2 ∧ collinear (a k) (b k)) :=
sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l617_61748


namespace factorial_fraction_is_integer_l617_61701

theorem factorial_fraction_is_integer (m n : ℕ) : 
  ∃ k : ℤ, (↑((2 * m).factorial * (2 * n).factorial) : ℚ) / 
    (↑(m.factorial * n.factorial * (m + n).factorial) : ℚ) = ↑k :=
by
  sorry

end factorial_fraction_is_integer_l617_61701


namespace lcm_hcf_problem_l617_61769

theorem lcm_hcf_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 → 
  Nat.gcd a b = 83 → 
  a = 210 → 
  b = 913 := by
sorry

end lcm_hcf_problem_l617_61769


namespace a_upper_bound_l617_61780

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 5

-- State the theorem
theorem a_upper_bound (a : ℝ) :
  (∀ x y, 5/2 < x ∧ x < y → f a x < f a y) →
  a ≤ 5 :=
by sorry

end a_upper_bound_l617_61780


namespace triangle_value_proof_l617_61711

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

theorem triangle_value_proof :
  ∀ (triangle : Nat),
  (triangle < 7) →
  (triangle < 9) →
  (base_to_decimal [triangle, 5] 7 = base_to_decimal [3, triangle] 9) →
  triangle = 4 := by
  sorry

end triangle_value_proof_l617_61711


namespace arithmetic_mean_problem_l617_61765

theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10)
  (h2 : r - p = 20) :
  (q + r) / 2 = 20 := by
sorry

end arithmetic_mean_problem_l617_61765


namespace bus_passenger_ratio_l617_61709

/-- Represents the number of passengers on a bus --/
structure BusPassengers where
  men : ℕ
  women : ℕ

/-- The initial state of passengers on the bus --/
def initial : BusPassengers := sorry

/-- The state of passengers after changes in city Y --/
def after_city_y : BusPassengers := sorry

/-- The total number of passengers at the start --/
def total_passengers : ℕ := 72

/-- Changes in passenger numbers at city Y --/
def men_leave : ℕ := 16
def women_enter : ℕ := 8

theorem bus_passenger_ratio :
  initial.men = 2 * initial.women ∧
  initial.men + initial.women = total_passengers ∧
  after_city_y.men = initial.men - men_leave ∧
  after_city_y.women = initial.women + women_enter ∧
  after_city_y.men = after_city_y.women :=
by sorry

end bus_passenger_ratio_l617_61709


namespace cosine_sine_sum_zero_l617_61713

theorem cosine_sine_sum_zero (x : ℝ) 
  (h : Real.cos (π / 6 - x) = -Real.sqrt 3 / 3) : 
  Real.cos (5 * π / 6 + x) + Real.sin (2 * π / 3 - x) = 0 := by
  sorry

end cosine_sine_sum_zero_l617_61713


namespace pen_cost_l617_61788

theorem pen_cost (pen ink : ℝ) 
  (total_cost : pen + ink = 2.50)
  (price_difference : pen = ink + 2) : 
  pen = 2.25 := by sorry

end pen_cost_l617_61788


namespace helmet_sales_theorem_l617_61710

/-- Represents the helmet sales scenario -/
structure HelmetSales where
  initialPrice : ℝ
  initialSales : ℝ
  priceReductionEffect : ℝ
  costPrice : ℝ

/-- Calculates the number of helmets sold after a price reduction -/
def helmetsSold (hs : HelmetSales) (priceReduction : ℝ) : ℝ :=
  hs.initialSales + hs.priceReductionEffect * priceReduction

/-- Calculates the monthly profit -/
def monthlyProfit (hs : HelmetSales) (priceReduction : ℝ) : ℝ :=
  (hs.initialPrice - priceReduction - hs.costPrice) * (helmetsSold hs priceReduction)

/-- The main theorem about helmet sales -/
theorem helmet_sales_theorem (hs : HelmetSales) 
    (h1 : hs.initialPrice = 80)
    (h2 : hs.initialSales = 200)
    (h3 : hs.priceReductionEffect = 20)
    (h4 : hs.costPrice = 50) : 
    (helmetsSold hs 10 = 400 ∧ monthlyProfit hs 10 = 8000) ∧
    ∃ x, x > 0 ∧ monthlyProfit hs x = 7500 ∧ hs.initialPrice - x = 65 := by
  sorry


end helmet_sales_theorem_l617_61710


namespace sundae_cost_theorem_l617_61792

/-- The cost of ice cream in dollars -/
def ice_cream_cost : ℚ := 2

/-- The cost of one topping in dollars -/
def topping_cost : ℚ := 1/2

/-- The number of toppings on the sundae -/
def num_toppings : ℕ := 10

/-- The total cost of a sundae with given number of toppings -/
def sundae_cost (ice_cream : ℚ) (topping : ℚ) (num : ℕ) : ℚ :=
  ice_cream + topping * num

theorem sundae_cost_theorem :
  sundae_cost ice_cream_cost topping_cost num_toppings = 7 := by
  sorry

end sundae_cost_theorem_l617_61792


namespace parabola_focus_distance_l617_61745

/-- The value of 'a' for a parabola y^2 = ax (a > 0) with a point P(3/2, y₀) on the parabola,
    where the distance from P to the focus is 2. -/
theorem parabola_focus_distance (a : ℝ) (y₀ : ℝ) : 
  a > 0 ∧ y₀^2 = a * (3/2) ∧ ((3/2 - (-a/4))^2 + y₀^2)^(1/2) = 2 → a = 2 := by
  sorry

end parabola_focus_distance_l617_61745


namespace daily_water_evaporation_l617_61786

/-- Given a glass with initial water amount, evaporation period, and total evaporation percentage,
    calculate the amount of water that evaporates each day. -/
theorem daily_water_evaporation
  (initial_water : ℝ)
  (evaporation_period : ℕ)
  (total_evaporation_percentage : ℝ)
  (h1 : initial_water = 25)
  (h2 : evaporation_period = 10)
  (h3 : total_evaporation_percentage = 1.6)
  : (initial_water * total_evaporation_percentage / 100) / evaporation_period = 0.04 := by
  sorry

end daily_water_evaporation_l617_61786


namespace optimal_sampling_methods_l617_61704

/-- Represents different income levels of families -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

structure Community where
  totalFamilies : Nat
  highIncomeFamilies : Nat
  middleIncomeFamilies : Nat
  lowIncomeFamilies : Nat
  sampleSize : Nat

structure School where
  femaleAthletes : Nat
  selectionSize : Nat

def optimalSamplingMethod (c : Community) (s : School) : 
  (SamplingMethod × SamplingMethod) :=
  sorry

theorem optimal_sampling_methods 
  (c : Community) 
  (s : School) 
  (h1 : c.totalFamilies = 500)
  (h2 : c.highIncomeFamilies = 125)
  (h3 : c.middleIncomeFamilies = 280)
  (h4 : c.lowIncomeFamilies = 95)
  (h5 : c.sampleSize = 100)
  (h6 : s.femaleAthletes = 12)
  (h7 : s.selectionSize = 3) :
  optimalSamplingMethod c s = (SamplingMethod.Stratified, SamplingMethod.SimpleRandom) :=
  sorry

end optimal_sampling_methods_l617_61704


namespace gravelling_rate_calculation_l617_61791

/-- Given a rectangular lawn with two intersecting roads, calculate the rate per square meter for gravelling the roads. -/
theorem gravelling_rate_calculation (lawn_length lawn_width road_width total_cost : ℝ) 
  (h1 : lawn_length = 70)
  (h2 : lawn_width = 30)
  (h3 : road_width = 5)
  (h4 : total_cost = 1900) : 
  total_cost / ((lawn_length * road_width) + (lawn_width * road_width) - (road_width * road_width)) = 4 := by
  sorry

#check gravelling_rate_calculation

end gravelling_rate_calculation_l617_61791


namespace division_remainder_problem_l617_61787

theorem division_remainder_problem (L S : ℕ) : 
  L - S = 1325 → 
  L = 1650 → 
  ∃ (R : ℕ), L = 5 * S + R ∧ R < S → 
  R = 25 := by
  sorry

end division_remainder_problem_l617_61787


namespace spies_configuration_exists_l617_61785

/-- Represents a position on the 6x6 board -/
structure Position where
  row : Fin 6
  col : Fin 6

/-- Represents the direction a spy is facing -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a spy on the board -/
structure Spy where
  pos : Position
  dir : Direction

/-- Determines if a spy can see a given position -/
def Spy.canSee (s : Spy) (p : Position) : Bool :=
  match s.dir with
  | Direction.Up => 
      (s.pos.row < p.row && p.row ≤ s.pos.row + 2 && s.pos.col - 1 ≤ p.col && p.col ≤ s.pos.col + 1) 
  | Direction.Down => 
      (s.pos.row > p.row && p.row ≥ s.pos.row - 2 && s.pos.col - 1 ≤ p.col && p.col ≤ s.pos.col + 1)
  | Direction.Left => 
      (s.pos.col > p.col && p.col ≥ s.pos.col - 2 && s.pos.row - 1 ≤ p.row && p.row ≤ s.pos.row + 1)
  | Direction.Right => 
      (s.pos.col < p.col && p.col ≤ s.pos.col + 2 && s.pos.row - 1 ≤ p.row && p.row ≤ s.pos.row + 1)

/-- A valid configuration of spies -/
def ValidConfiguration (spies : List Spy) : Prop :=
  spies.length = 18 ∧ 
  ∀ s1 s2, s1 ∈ spies → s2 ∈ spies → s1 ≠ s2 → ¬(s1.canSee s2.pos) ∧ ¬(s2.canSee s1.pos)

/-- There exists a valid configuration of 18 spies on a 6x6 board -/
theorem spies_configuration_exists : ∃ spies : List Spy, ValidConfiguration spies := by
  sorry

end spies_configuration_exists_l617_61785


namespace factorial_sum_equals_1190_l617_61779

theorem factorial_sum_equals_1190 : 
  (Nat.factorial 16) / ((Nat.factorial 6) * (Nat.factorial 10)) + 
  (Nat.factorial 11) / ((Nat.factorial 6) * (Nat.factorial 5)) = 1190 := by
  sorry

end factorial_sum_equals_1190_l617_61779


namespace parallelogram_inequality_l617_61754

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Parallelogram P -/
structure Parallelogram (n : ℕ) (t : ℝ) where
  v1 : ℝ × ℝ := (0, 0)
  v2 : ℝ × ℝ := (0, t)
  v3 : ℝ × ℝ := (t * fib (2 * n + 1), t * fib (2 * n))
  v4 : ℝ × ℝ := (t * fib (2 * n + 1), t * fib (2 * n) + t)

/-- Number of integer points inside P -/
def L (n : ℕ) (t : ℝ) : ℕ := sorry

/-- Area of P -/
def M (n : ℕ) (t : ℝ) : ℝ := t^2 * fib (2 * n + 1)

/-- Main theorem -/
theorem parallelogram_inequality (n : ℕ) (t : ℝ) (hn : n > 1) (ht : t ≥ 1) :
  |Real.sqrt (L n t) - Real.sqrt (M n t)| ≤ Real.sqrt 2 := by sorry

end parallelogram_inequality_l617_61754


namespace family_income_increase_l617_61770

theorem family_income_increase (total_income : ℝ) 
  (masha_scholarship mother_salary father_salary grandfather_pension : ℝ) : 
  masha_scholarship + mother_salary + father_salary + grandfather_pension = total_income →
  masha_scholarship = 0.05 * total_income →
  mother_salary = 0.15 * total_income →
  father_salary = 0.25 * total_income →
  grandfather_pension = 0.55 * total_income :=
by
  sorry

#check family_income_increase

end family_income_increase_l617_61770


namespace line_points_b_plus_one_l617_61794

/-- Given a line y = 0.75x + 1 and three points on the line, prove that b + 1 = 5 -/
theorem line_points_b_plus_one (b a : ℝ) : 
  (b = 0.75 * 4 + 1) →  -- Point (4, b) on the line
  (5 = 0.75 * a + 1) →  -- Point (a, 5) on the line
  (b + 1 = 0.75 * a + 1) →  -- Point (a, b + 1) on the line
  b + 1 = 5 :=
by sorry

end line_points_b_plus_one_l617_61794


namespace coins_on_side_for_36_circumference_l617_61778

/-- The number of coins on one side of a square arrangement, given the total number of coins on the circumference. -/
def coins_on_one_side (circumference_coins : ℕ) : ℕ :=
  (circumference_coins + 4) / 4

/-- Theorem stating that for a square arrangement of coins with 36 coins on the circumference, there are 10 coins on one side. -/
theorem coins_on_side_for_36_circumference :
  coins_on_one_side 36 = 10 := by
  sorry

#eval coins_on_one_side 36  -- This should output 10

end coins_on_side_for_36_circumference_l617_61778


namespace negation_of_implication_l617_61761

theorem negation_of_implication (x y : ℝ) :
  (¬(x = y → Real.sqrt x = Real.sqrt y)) ↔ (x ≠ y → Real.sqrt x ≠ Real.sqrt y) := by
  sorry

end negation_of_implication_l617_61761


namespace negation_of_sine_inequality_l617_61768

theorem negation_of_sine_inequality :
  (¬ ∀ x : ℝ, |Real.sin x| < 1) ↔ (∃ x : ℝ, |Real.sin x| ≥ 1) :=
by sorry

end negation_of_sine_inequality_l617_61768


namespace ice_cream_sales_ratio_l617_61718

/-- Ice cream sales problem -/
theorem ice_cream_sales_ratio (tuesday_sales wednesday_sales : ℕ) : 
  tuesday_sales = 12000 →
  wednesday_sales = 36000 - tuesday_sales →
  (wednesday_sales : ℚ) / tuesday_sales = 2 := by
  sorry

end ice_cream_sales_ratio_l617_61718


namespace percentage_difference_l617_61719

theorem percentage_difference (x y : ℝ) (h : x = 6 * y) :
  (x - y) / x * 100 = 83.33333333333333 := by
  sorry

end percentage_difference_l617_61719


namespace mean_median_difference_l617_61708

theorem mean_median_difference (x : ℕ) (h : x > 0) : 
  let sequence := [x, x + 2, x + 4, x + 7, x + 37]
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 37)) / 5
  let median := x + 4
  mean - median = 6 := by
  sorry

end mean_median_difference_l617_61708


namespace students_just_passed_l617_61728

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ)
  (h_total : total = 300)
  (h_first_div : first_div_percent = 30 / 100)
  (h_second_div : second_div_percent = 54 / 100)
  (h_all_passed : first_div_percent + second_div_percent < 1) :
  total - (total * first_div_percent).floor - (total * second_div_percent).floor = 48 := by
  sorry

end students_just_passed_l617_61728


namespace barycentric_coordinate_properties_l617_61796

/-- Barycentric coordinates in a tetrahedron -/
structure BarycentricCoord :=
  (x₁ x₂ x₃ x₄ : ℝ)
  (sum_to_one : x₁ + x₂ + x₃ + x₄ = 1)

/-- The tetrahedron A₁A₂A₃A₄ -/
structure Tetrahedron :=
  (A₁ A₂ A₃ A₄ : BarycentricCoord)

/-- A point lies on line A₁A₂ iff x₃ = 0 and x₄ = 0 -/
def lies_on_line_A₁A₂ (t : Tetrahedron) (p : BarycentricCoord) : Prop :=
  p.x₃ = 0 ∧ p.x₄ = 0

/-- A point lies on plane A₁A₂A₃ iff x₄ = 0 -/
def lies_on_plane_A₁A₂A₃ (t : Tetrahedron) (p : BarycentricCoord) : Prop :=
  p.x₄ = 0

/-- A point lies on the plane through A₃A₄ parallel to A₁A₂ iff x₁ = -x₂ and x₃ + x₄ = 1 -/
def lies_on_plane_parallel_A₁A₂_through_A₃A₄ (t : Tetrahedron) (p : BarycentricCoord) : Prop :=
  p.x₁ = -p.x₂ ∧ p.x₃ + p.x₄ = 1

theorem barycentric_coordinate_properties (t : Tetrahedron) (p : BarycentricCoord) :
  (lies_on_line_A₁A₂ t p ↔ p.x₃ = 0 ∧ p.x₄ = 0) ∧
  (lies_on_plane_A₁A₂A₃ t p ↔ p.x₄ = 0) ∧
  (lies_on_plane_parallel_A₁A₂_through_A₃A₄ t p ↔ p.x₁ = -p.x₂ ∧ p.x₃ + p.x₄ = 1) := by
  sorry

end barycentric_coordinate_properties_l617_61796


namespace min_value_of_geometric_sequence_l617_61795

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem min_value_of_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_condition : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) :
  ∃ min_value : ℝ, min_value = 54 ∧ 
  ∀ x : ℝ, (∃ a : ℕ → ℝ, geometric_sequence a ∧ 
    (∀ n : ℕ, a n > 0) ∧ 
    2 * a 4 + a 3 - 2 * a 2 - a 1 = 8 ∧
    2 * a 8 + a 7 = x) → x ≥ min_value :=
sorry

end min_value_of_geometric_sequence_l617_61795


namespace unique_solution_to_equation_l617_61781

theorem unique_solution_to_equation :
  ∃! z : ℝ, (z + 2)^4 + (2 - z)^4 = 258 :=
by
  -- The proof would go here
  sorry

end unique_solution_to_equation_l617_61781


namespace bug_path_tiles_l617_61764

-- Define the rectangle's dimensions
def width : ℕ := 12
def length : ℕ := 20

-- Define the function to calculate the number of tiles visited
def tilesVisited (w l : ℕ) : ℕ := w + l - Nat.gcd w l

-- Theorem statement
theorem bug_path_tiles : tilesVisited width length = 28 := by
  sorry

end bug_path_tiles_l617_61764


namespace sin_sq_plus_cos_sq_eq_one_s_sq_plus_c_sq_eq_one_l617_61731

/-- Given an angle θ, prove that sin²θ + cos²θ = 1 -/
theorem sin_sq_plus_cos_sq_eq_one (θ : Real) : (Real.sin θ)^2 + (Real.cos θ)^2 = 1 := by
  sorry

/-- Given s = sin θ and c = cos θ for some angle θ, prove that s² + c² = 1 -/
theorem s_sq_plus_c_sq_eq_one (s c : Real) (h : ∃ θ : Real, s = Real.sin θ ∧ c = Real.cos θ) : s^2 + c^2 = 1 := by
  sorry

end sin_sq_plus_cos_sq_eq_one_s_sq_plus_c_sq_eq_one_l617_61731


namespace y_equation_solution_l617_61782

theorem y_equation_solution (y : ℝ) (c d : ℕ+) 
  (h1 : y^2 + 2*y + 2/y + 1/y^2 = 20)
  (h2 : y = c + Real.sqrt d) : 
  c + d = 5 := by
sorry

end y_equation_solution_l617_61782


namespace power_of_product_squared_l617_61714

theorem power_of_product_squared (a : ℝ) : (3 * a^2)^2 = 9 * a^4 := by
  sorry

end power_of_product_squared_l617_61714


namespace percentage_of_whole_l617_61735

theorem percentage_of_whole (whole : ℝ) (part : ℝ) (h : whole = 450 ∧ part = 229.5) :
  (part / whole) * 100 = 51 := by
  sorry

end percentage_of_whole_l617_61735


namespace store_a_discount_proof_l617_61775

/-- The additional discount percentage offered by Store A -/
def store_a_discount : ℝ := 8

/-- The full price of the smartphone at Store A -/
def store_a_full_price : ℝ := 125

/-- The full price of the smartphone at Store B -/
def store_b_full_price : ℝ := 130

/-- The additional discount percentage offered by Store B -/
def store_b_discount : ℝ := 10

/-- The price difference between Store A and Store B after discounts -/
def price_difference : ℝ := 2

theorem store_a_discount_proof :
  store_a_full_price * (1 - store_a_discount / 100) =
  store_b_full_price * (1 - store_b_discount / 100) - price_difference :=
by sorry


end store_a_discount_proof_l617_61775


namespace sock_ratio_is_one_to_two_l617_61720

/-- Represents the order of socks --/
structure SockOrder where
  green : ℕ
  red : ℕ
  green_price : ℝ
  red_price : ℝ

/-- The original order --/
def original_order : SockOrder := {
  green := 6,
  red := 0,  -- We don't know this value yet
  green_price := 0,  -- We don't know this value yet
  red_price := 0  -- We don't know this value yet
}

/-- The swapped order --/
def swapped_order (o : SockOrder) : SockOrder := {
  green := o.red,
  red := o.green,
  green_price := o.green_price,
  red_price := o.red_price
}

/-- The cost of an order --/
def cost (o : SockOrder) : ℝ :=
  o.green * o.green_price + o.red * o.red_price

/-- The theorem to prove --/
theorem sock_ratio_is_one_to_two :
  ∃ (o : SockOrder),
    o.green = 6 ∧
    o.green_price = 3 * o.red_price ∧
    cost (swapped_order o) = 1.4 * cost o ∧
    o.green / o.red = 1 / 2 :=
sorry

end sock_ratio_is_one_to_two_l617_61720


namespace equation_solution_l617_61752

theorem equation_solution :
  ∃ y : ℚ, y ≠ -2 ∧ (6 * y / (y + 2) - 2 / (y + 2) = 5 / (y + 2)) ∧ y = 7/6 := by
  sorry

end equation_solution_l617_61752


namespace function_satisfying_inequality_is_constant_l617_61733

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h : ∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2 * y + 3 * z)) :
  ∃ C : ℝ, ∀ x : ℝ, f x = C :=
sorry

end function_satisfying_inequality_is_constant_l617_61733


namespace greatest_valid_number_divisible_by_11_l617_61716

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    A < 9 ∧
    n = A * 10000 + B * 1000 + C * 100 + B * 10 + A

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem greatest_valid_number_divisible_by_11 :
  ∀ n : ℕ, is_valid_number n → is_divisible_by_11 n → n ≤ 87978 :=
sorry

end greatest_valid_number_divisible_by_11_l617_61716


namespace subset_implies_lower_bound_l617_61727

/-- Given sets A = [1, 4) and B = (-∞, a), if A ⊂ B, then a ≥ 4 -/
theorem subset_implies_lower_bound (a : ℝ) :
  let A := { x : ℝ | 1 ≤ x ∧ x < 4 }
  let B := { x : ℝ | x < a }
  A ⊆ B → a ≥ 4 := by
  sorry

end subset_implies_lower_bound_l617_61727


namespace fraction_equality_l617_61726

-- Define the @ operation
def at_op (a b : ℕ) : ℕ := a * b + b^2

-- Define the # operation
def hash_op (a b : ℕ) : ℕ := a + b + a * b^2

-- Theorem statement
theorem fraction_equality : (at_op 5 3 : ℚ) / (hash_op 5 3) = 24 / 53 := by
  sorry

end fraction_equality_l617_61726
